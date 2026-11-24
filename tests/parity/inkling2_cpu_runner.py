"""Isolated registered-path CPU test; native CUDA extension and Triton are stubs."""
import os
import sys
import types
import torch
from safetensors.torch import load_file


class Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return lambda *a, **k: a[0] if a and callable(a[0]) else (lambda f: f)


def install_test_stubs():
    triton = Stub('triton')
    triton.__path__, triton.__version__ = [], '3.0'
    triton.KernelInterface = type('KernelInterface', (), {})
    triton.Config = lambda *a, **k: None
    language = Stub('triton.language')
    language.constexpr = object()
    triton.language = language
    sys.modules.update({'triton': triton, 'triton.language': language})
    ext = types.ModuleType('exllamav3.ext')
    ext.exllamav3_ext = Stub('ext_impl')
    ext.exllamav3_ext.exl3_moe_cpu_core_order = lambda: ([], 0)
    def update(k, v, kc, vc, table, lengths):
        offsets = torch.arange(k.shape[1]) + lengths[0]
        pages, rows = table[0, offsets // kc.shape[1]], offsets % kc.shape[1]
        kc[pages, rows] = k[0]
        vc[pages, rows] = v[0]
    def stage(arena, width, tables, positions, rope, batch, ids, pages, offsets):
        assert len(tables) == 1 and not rope
        table = tables[0].flatten()[:width]
        arena.zero_()
        arena[:len(table)] = table
        arena[width] = positions[0]
        if ids:
            batch[0, 0] = pages[0][0, offsets[0]] = ids[0][0, positions[0]]
    ext.exllamav3_ext.generator_stage_decode = stage
    ext.exllamav3_ext.paged_kv_cache_update = update
    sys.modules['exllamav3.ext'] = ext


def check_ring_indices():
    from types import SimpleNamespace
    from exllamav3.architecture.inkling2_exl import InklingWindowCache
    from exllamav3.constants import PAGE_SIZE
    window, pages, slot = 640, 3, 1
    ring = InklingWindowCache(None, SimpleNamespace(window=window), 0, 6 * PAGE_SIZE, lambda: None)
    ring.table = torch.arange(2 * pages, dtype=torch.int32).view(2, pages).repeat(1, 2)
    workspace = {}
    pos = torch.empty(1, dtype=torch.int32)
    for past in (1, 639, 767, 768, 895, 1024, 2049):
        ctx = dict(lengths=pos.fill_(past), window_workspace=workspace)
        base, length, table, skip, write, write_table = ring.layout(ctx, past, 1, slot)
        logical = torch.arange(max(0, past - window + 1), past)
        offset = logical - base
        expected_pages = slot * pages + logical // PAGE_SIZE % pages
        assert torch.equal(table[0, offset // PAGE_SIZE], expected_pages)
        assert length.item() == past - base and skip == 0
        assert write_table[0, write.long() // PAGE_SIZE].item() == slot * pages + past // PAGE_SIZE % pages
        assert write.item() % PAGE_SIZE == past % PAGE_SIZE
    assert len(workspace) == 1


@torch.inference_mode()
def check_graph_body(model, cache, ids):
    """Use the capture body's static buffers with the registered CPU oracle."""
    from exllamav3.architecture.inkling2_graph import DecodeGraph
    from exllamav3.modules.arch_specific.inkling2_dspark import VisitTaps
    from exllamav3.cache.recurrent_util import advance_recurrent_states

    visits = len(model.config.layer_map)
    eager, staged = cache.get_new_state(), cache.get_new_state()
    tables = [torch.tensor([row], dtype=torch.int32) for row in ((0, 1), (2, 3))]
    entries = {}
    def params(state, table):
        return dict(cache=cache, attn_mode='flash_attn', cache_seqlens=table.new_full((1,), state.position),
                    recurrent_states=[state], block_table=table,
                    inkling_tap_collector=VisitTaps(visits, (1, visits)))

    try:
        for state, table in zip((eager, staged), tables):
            model.forward(ids[:, :253], params(state, table))
        for width in (1, 1, 8, 3, 7, 2, 4, 6, 5, 8, 3):
            pos = eager.position
            # Reassign the two previously unused global pages just before wrap.
            if pos == 255:
                tables[0][0, 1], tables[1][0, 1] = 3, 1
            tokens = ids[:, pos:pos + width]
            expected_params, actual_params = params(eager, tables[0]), params(staged, tables[1])
            actual_params['_inkling_record_conv'] = expected_params['_inkling_record_conv'] = width > 1
            expected = model.forward(tokens, expected_params)
            x = model.prepare_inputs(tokens, actual_params)
            ctx = actual_params['_inkling']
            if width not in entries:
                entries[width] = DecodeGraph(x, ctx)
            entry = entries[width]
            entry.stage(x, ctx)
            actual, taps = entry.body(model.forward_ls, actual_params)
            advance_recurrent_states(tokens, actual_params, model)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(taps.result(), expected_params['inkling_tap_collector'].result(), rtol=0, atol=0)
            if width > 1:
                for key, planes in staged.verify_inputs.items():
                    for actual_plane, expected_plane in zip(planes, eager.verify_inputs[key], strict=True):
                        torch.testing.assert_close(actual_plane, expected_plane, rtol=0, atol=0)
            for layer in cache.recurrent_layers.values():
                for tensor in layer.states:
                    torch.testing.assert_close(tensor[staged.slot], tensor[eager.slot], rtol=0, atol=0)
        # Reject a capture before replay, then keep using eager for this
        # signature. The request must advance once and produce a fresh output.
        import warnings
        from contextlib import nullcontext
        from unittest.mock import Mock, patch
        entry = entries[1]
        entry.stream = Mock()
        entry.warmed = True
        failure = Mock()
        failure.capture_begin.side_effect = RuntimeError('test capture failure')
        with (patch.object(torch.cuda, 'current_stream', return_value=entry.stream),
              patch.object(torch.cuda, 'stream', return_value=nullcontext()),
              patch.object(torch.cuda, 'CUDAGraph', return_value=failure),
              warnings.catch_warnings(record=True) as notices):
            warnings.simplefilter('always')
            for _ in range(2):
                pos = staged.position
                eq, p = params(eager, tables[0]), params(staged, tables[1])
                token = ids[:, pos:pos + 1]
                expected = model.forward(token, eq)
                x = model.prepare_inputs(token, p)
                actual = entry.run(model.forward_ls, x, p)
                advance_recurrent_states(x, p, model)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert entry.failed and failure.capture_begin.call_count == 1
            assert any('test capture failure' in str(w.message) for w in notices)
    finally:
        eager.free()
        staged.free()
    print('registered graph body: eager logits, taps and recurrent slots identical across page and ring wrap')


@torch.inference_mode()
def check_chunked_prefill(model, cache, ids):
    from exllamav3.generator.inkling2 import Inkling2DecodeRequest

    with (Inkling2DecodeRequest(model, cache, (0, 2)) as chunked,
          Inkling2DecodeRequest(model, cache, (3, 1)) as serial):
        for lo, hi in ((0, 37), (37, 269)):
            chunk, features = chunked._forward(ids[:, lo:hi])
            chunked._commit(chunk, features)
            rows = []
            for pos in range(lo, hi):
                logits, features = serial._forward(ids[:, pos:pos + 1])
                serial._commit(logits, features)
                rows.append(logits)
            torch.testing.assert_close(chunk, torch.cat(rows), rtol=1e-4, atol=1e-5)
            assert chunked.position == serial.position == hi
            for layer in cache.recurrent_layers.values():
                for plane in layer.states:
                    torch.testing.assert_close(plane[chunked._state.slot], plane[serial._state.slot],
                                               rtol=1e-4, atol=1e-5)
            for layer in cache.layers.values():
                for tensor in layer.get_tensors():
                    a, b = [tensor[list(req.pages)].flatten(0, 1)[:hi] for req in (chunked, serial)]
                    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    print('chunked prefill: 269 logits and final conv/ring/global KV match single-token forwards')


def check_dspark(model, cache, draft, prompt):
    from exllamav3.generator.inkling2 import Inkling2DecodeRequest
    from exllamav3.modules.arch_specific.inkling2_dspark import VisitTaps

    visits = len(model.config.layer_map)
    indices = draft.draft_config.target_layer_ids
    taps, expected_taps = VisitTaps(visits, indices), VisitTaps(visits, indices)
    output = model.forward(prompt[:, :6], dict(inkling_tap_collector=taps, last_tokens_only=1))
    with torch.inference_mode():
        expected = model.loop(model.embed_norm(model.embedding(prompt[:, :6])), tap_collector=expected_taps)
    torch.testing.assert_close(output, expected[:, -1:])
    torch.testing.assert_close(taps.result(), expected_taps.result())

    def finish(request, tokens, count):
        while len(tokens) < count:
            result, _ = request.step(count - len(tokens))
            tokens.extend(result.tokens)
        request.flush()

    def cache_tensors(request, *, committed=False):
        for layer in cache.layers.values():
            for tensor in layer.get_tensors():
                rows = tensor[list(request.pages)].flatten(0, 1)
                yield rows[:request.position] if committed else rows
        for layer in cache.recurrent_layers.values():
            for tensor in layer.states:
                yield tensor[request._state.slot]

    # A multi-page prefill writes every tap and retains just the window's
    # trailing KV, even when the initial prompt wraps the physical ring.
    long_prompt = torch.cat((prompt, prompt[:, :30]), 1)
    with Inkling2DecodeRequest(model, cache, (3, 1), draft=draft) as request:
        request.prefill(long_prompt)
        all_taps = VisitTaps(visits, indices)
        expected = model.forward(long_prompt, dict(inkling_tap_collector=all_taps, last_tokens_only=1))
        torch.testing.assert_close(request._logits, expected[0, -1, :request.vocab_size], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(request.features, all_taps.result().to(request.features.dtype))

    total = 24
    with Inkling2DecodeRequest(model, cache, (0, 2)) as plain, \
         Inkling2DecodeRequest(model, cache, (1, 3), draft=draft) as spec:
        plain.prefill(prompt)
        spec.prefill(prompt)
        # Exercise the real static draft body, its masked padding, and KV
        # append on the same bucket with two different anchor positions.
        anchor = torch.tensor([2])
        with torch.inference_mode():
            for n in (1, 7, 8, 249, 252, 253):
                features = spec.features[:, :n]
                kwargs = dict(num_proposals=4)
                eager = draft.propose(anchor, features, spec._positions, **kwargs)
                staged = draft.propose(anchor, features, spec._positions, state=spec._state, **kwargs)
                torch.testing.assert_close(staged[0], eager[0])
                torch.testing.assert_close(staged[1], eager[1], rtol=.01, atol=.02)
                assert spec._state.draft_graph.length == n
        expected_ids, actual_ids = [], []
        finish(plain, expected_ids, total)
        first, proposed = spec.step(total)
        actual_ids.extend(first.tokens)
        assert proposed == 0 and actual_ids == expected_ids[:1]
        # Exercise the draft residency assertion on the production step path.
        draft.draft.register_buffer('_test_off_device', torch.empty(1, device='meta'))
        try:
            spec.step(total - 1)
        except AssertionError as exc:
            assert 'DSpark tensor on meta' in str(exc)
        else:
            raise AssertionError('draft residency guard missed a misplaced buffer')
        finally:
            del draft.draft._test_off_device
        ring_heads = [(conv, [plane[spec._state.slot, 0, :5].clone() for plane in conv.states[4:]])
                      for conv in cache.recurrent_layers.values() if conv.window_cache]
        assert ring_heads and all(conv.window_cache.attention.window == 16 for conv, _ in ring_heads)
        kv_tail = [[t[3, :5].clone() for t in layer.get_tensors()] for layer in cache.layers.values()]

        calls, forward, propose = [], model.forward, draft.propose
        def trace(ids, params):
            calls.append(ids.shape[1])
            return forward(ids, params)

        def controlled_proposals(*args, **kwargs):
            ids, logits, probs = propose(*args, **kwargs)
            start = len(actual_ids)
            ids = torch.tensor([expected_ids[start:start + ids.shape[1]]])
            if start == 1:
                ids[0, 2] = (ids[0, 2] + 1) % model.config.unpadded_vocab_size
            return ids, logits, probs

        model.forward, draft.propose = trace, controlled_proposals
        try:
            result, proposed = spec.step(total - len(actual_ids))
            assert calls == [8] and proposed == 7 and result.accepted == 2
            actual_ids.extend(result.tokens)
            # The verification crossed onto page 3. Its rejected rows must be
            # restored even though the accepted prefix ends on page 1.
            assert spec.position == 256
            for layer, saved in zip(cache.layers.values(), kv_tail):
                for kv, old in zip(layer.get_tensors(), saved):
                    torch.testing.assert_close(kv[3, :5], old, rtol=0, atol=0)
            for conv, saved in ring_heads:
                for plane, old in zip(conv.states[4:], saved, strict=True):
                    torch.testing.assert_close(plane[spec._state.slot, 0, :5], old, rtol=0, atol=0)
            calls.clear()
            result, proposed = spec.step(total - len(actual_ids))
            assert calls == [8] and result.accepted == proposed == 7
            actual_ids.extend(result.tokens)
        finally:
            model.forward, draft.propose = forward, propose

        finish(spec, actual_ids, total)
        assert actual_ids == expected_ids and len(actual_ids) == total
        assert spec.features.shape[1] == spec.position == plain.position == prompt.numel() + total
        assert max(actual_ids) < model.config.unpadded_vocab_size
        # Only allocated global rows through the committed prefix belong to a request.
        for actual, expected in zip(cache_tensors(spec, committed=True), cache_tensors(plain, committed=True), strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        complete = torch.cat((prompt, torch.tensor([actual_ids])), dim=1)
        complete_taps = VisitTaps(visits, indices)
        model.forward(complete, dict(inkling_tap_collector=complete_taps, last_tokens_only=1))
        torch.testing.assert_close(spec.features, complete_taps.result().to(spec.features.dtype))

    with Inkling2DecodeRequest(model, cache, (1, 3), draft=draft, temperature=.8, seed=11) as spec:
        spec.prefill(prompt)
        first, _ = spec.step(total)
        before = [tensor.clone() for tensor in cache_tensors(spec)]
        features, rng = spec.features.clone(), spec.rng.get_state()

        class Interrupted(Exception):
            pass

        def interrupt(ids, params):
            forward(ids, params)
            raise Interrupted

        model.forward = interrupt
        try:
            spec.step(total - 1)
            raise AssertionError('verification did not reach the target forward')
        except Interrupted:
            pass
        finally:
            model.forward = forward
        assert spec.position == prompt.numel() and torch.equal(spec.rng.get_state(), rng)
        assert torch.equal(spec.features, features)
        assert all(torch.equal(a, b) for a, b in zip(cache_tensors(spec), before, strict=True))
        tokens = list(first.tokens)
        finish(spec, tokens, total)
        assert len(tokens) == total and spec.position == prompt.numel() + total
    print('registered DSpark: 24 greedy tokens match, cross-page and SWA ring-wrap rollback and sampled retry passed')
    return expected_ids


def check_server_generator(model, cache, draft, draft_cache, prompt, expected):
    from contextlib import ExitStack
    from types import SimpleNamespace
    from unittest.mock import patch
    from exllamav3 import Generator, Job
    from exllamav3.generator.sampler import CustomSampler, SS_MinP, SS_Sample_mn, SS_Temperature, SS_TopK, SS_TopP
    from exllamav3.modules.arch_specific.inkling2_dspark import VisitTaps

    def pageable(factory):
        def allocate(*args, **kwargs):
            kwargs.pop('pin_memory', None)
            return factory(*args, **kwargs)
        return allocate

    # Use the real job loop with a CPU argmax sampler and pageable staging.
    vocab = model.config.unpadded_vocab_size
    tokenizer = SimpleNamespace(actual_vocab_size=vocab, get_id_to_piece_list=lambda _: list(map(str, range(vocab))))
    sampler = SimpleNamespace(reqs_past_ids=False, forward=lambda logits, *a, **k: logits.argmax(-1))
    taps = VisitTaps(len(model.config.layer_map), draft.draft_config.target_layer_ids)
    tokens = torch.cat((prompt, torch.tensor([expected])), dim=1)
    model.forward(tokens, dict(inkling_tap_collector=taps, last_tokens_only=1))
    expected_features = taps.result().to(torch.bfloat16)
    propose = draft.propose

    def check_features(anchor, features, positions, **kwargs):
        start = features.shape[1]
        torch.testing.assert_close(features, expected_features[:, :start])
        assert anchor.item() == tokens[0, start].item()
        return propose(anchor, features, positions, **kwargs)

    def run(generator, job):
        generator.enqueue(job)
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                if result['stage'] == 'error':
                    raise result['error']
        return job.sequences[0].sequence_ids.torch()[0, prompt.shape[1]:].tolist()

    with ExitStack() as stack:
        for name in ('empty', 'zeros'):
            stack.enter_context(patch.object(torch, name, pageable(getattr(torch, name))))
        stack.enter_context(patch.object(Generator, '_sync_verify_pass', lambda self, device: None))
        generator = Generator(model, cache, tokenizer, draft_model=draft, draft_cache=draft_cache,
                              max_batch_size=1, pipelined_prefill=False, enable_defrag=False,
                              max_chunk_size=256, recurrent_checkpoint_interval_pp=256,
                              reuse_inkling_graphs=True)
        stack.callback(generator.close)
        job = Job(input_ids=prompt, max_new_tokens=len(expected), sampler=sampler, stop_conditions=[])
        with patch.object(draft, 'propose', check_features):
            assert run(generator, job) == expected
        assert job.rejected_draft_tokens > 0

        sampled_sampler = CustomSampler([SS_Temperature(.8), SS_MinP(.04), SS_TopK(12), SS_TopP(.95), SS_Sample_mn()])
        job = Job(input_ids=prompt, max_new_tokens=len(expected), sampler=sampled_sampler, seed=318)
        # This fixed seed exercises both rejection outcomes on the full-draft
        # trajectory. Exercise the adaptive schedule independently below.
        with (patch.object(generator, 'dspark_adaptive', False),
              patch.object(job, 'sample_dspark', wraps=job.sample_dspark) as verifier):
            assert len(run(generator, job)) == len(expected)
        assert job.dspark_rule == 'rejection' and verifier.call_count > 0
        assert job.accepted_draft_tokens > 0 and job.rejected_draft_tokens > 0

        def run_greedy(ids, count):
            job = Job(input_ids=ids, max_new_tokens=count, sampler=sampler, stop_conditions=[])
            run(generator, job)
            return job

        # Keep the first two pages, then replace the suffix. The draft must see
        # the same tap rows as a full forward of the edited prompt.
        prefix = prompt.repeat(1, 3)[:, :512]
        # Page two is outside the original prompt's tail checkpoint window.
        original = torch.cat((prefix, prompt.repeat(1, 3)[:, :541]), dim=1)
        followup = torch.cat((prefix, prompt[:, :29].flip([1])), dim=1)
        stack.enter_context(patch.object(draft, 'propose', propose))
        assert run_greedy(original, 2).cached_pages == 0
        expected_taps = VisitTaps(len(model.config.layer_map), draft.draft_config.target_layer_ids)
        model.forward(followup, dict(inkling_tap_collector=expected_taps, last_tokens_only=1))
        expected_features = expected_taps.result().to(torch.bfloat16)
        tokens = followup
        first_drafts = []

        def check_first_draft(anchor, features, positions, **kwargs):
            if features.shape[1] == followup.shape[1] - 1:
                first_drafts.append(anchor.item())
                return check_features(anchor, features, positions, **kwargs)
            return propose(anchor, features, positions, **kwargs)

        stack.enter_context(patch.object(draft, 'propose', check_first_draft))
        cached = run_greedy(followup, 8)
        assert cached.cached_pages == 2 and cached.last_recurrent_checkpoint_pos == 512
        generator.pagetable.reset_page_table()
        cold = run_greedy(followup, 8)
        assert cold.cached_pages == 0
        assert len(first_drafts) == 2
        assert torch.equal(cached.sequences[0].sequence_ids.torch(), cold.sequences[0].sequence_ids.torch())
        # Force an expensive verifier without relying on host scheduling. A short
        # interval exercises multiple pauses and probes on a real greedy job.
        from exllamav3.generator.draft_confidence import DSparkGain
        modes = []
        record = DSparkGain.record

        def measured(gain, drafted, emitted, seconds):
            modes.append(drafted)
            record(gain, drafted, emitted, 100. if drafted else 1.)

        generator.dspark_probe_interval = 6
        with patch.object(DSparkGain, 'record', measured):
            guarded = run_greedy(prompt, len(expected))
        assert guarded.sequences[0].sequence_ids.torch()[0, prompt.shape[1]:].tolist() == expected
        assert any(modes) and False in modes and True in modes[modes.index(False):]
        generator.dspark_adaptive = False
        unguarded = run_greedy(prompt, len(expected))
        assert unguarded.dspark_gain is None
        assert unguarded.sequences[0].sequence_ids.torch()[0, prompt.shape[1]:].tolist() == expected
    print('Tabby generator: greedy parity, sampled rejection, two-page restore and cached taps passed')


def check_server_sampling_distribution():
    from types import SimpleNamespace
    from unittest.mock import patch
    from exllamav3 import ComboSampler, Job
    from exllamav3.generator.sampler import custom

    # CPU Gumbel replacement for the plain, eager sampler. The native kernel
    # has the same categorical law and its own RNG implementation.
    def gumbel(probs, output, seed):
        rng = torch.Generator().manual_seed(seed)
        noise = -torch.rand(probs.shape, generator=rng).log().neg().log()
        output.copy_(probs.log() + noise)

    with patch.object(custom, 'fused_sampler_enable', False):
        sampler = ComboSampler(temperature=.6, top_k=4, top_p=.93, min_p=.04)
        assert not ComboSampler(rep_p=1.1).supports_speculative_sampling
        assert not ComboSampler(logit_bias={0: 1.}).supports_speculative_sampling

    target = torch.tensor([[2.2, 2.0, 1.8, .8, -2.0],
                           [.8, 2.0, 1.6, 1.9, -2.0],
                           [2.0, 1.6, 1.9, .8, -2.0],
                           [1.8, .8, 2.0, 1.9, -2.0],
                           [.8, 1.9, 2.0, 1.6, -2.0]])
    draft = torch.tensor([[1.7, 2.1, .6, 1.9, -2.0]])
    prompt = torch.tensor([[0]])
    job = Job(prompt, sampler=sampler, seed=640)
    job.generator = SimpleNamespace(tokenizer=SimpleNamespace(actual_vocab_size=target.shape[-1]), recurrent_cache=None)
    assert job.dspark_sampling_rule() == 'rejection'
    job.set_banned_strings(['test'])
    assert job.dspark_sampling_rule().startswith('token-match')
    job.set_banned_strings([])
    for _ in range(70):
        job.dspark_draft_window(7, .4)
        job.record_dspark_acceptance(7, 2, 1)
    assert job.dspark_draft_window(7, .4) == 2
    job.new_tokens = 1
    q = job.dspark_distribution(draft)
    rng = job.dspark_rng(draft.device)
    plain_rng = torch.Generator().manual_seed(268)
    samples = 6000
    counts = torch.zeros((2, target.shape[-1], target.shape[-1]))
    with patch.object(custom.ext, 'gumbel_noise_log', gumbel):
        for seed in torch.randint(2**31, (samples, 2), generator=plain_rng).tolist():
            proposal = torch.multinomial(q, 1, generator=rng)
            p = torch.stack((target[0], target[proposal.item()]))[None]
            result = job.sample_dspark(p, proposal, q[None])
            first, *bonus = result.tokens
            second = bonus[0] if bonus else torch.multinomial(job.dspark_distribution(target[first]), 1, generator=rng).item()
            counts[0, first, second] += 1
            first = sampler.forward(target[0:1].clone(), rand_u32=seed[0]).item()
            second = sampler.forward(target[first:first + 1].clone(), rand_u32=seed[1]).item()
            counts[1, first, second] += 1
    target_probs = sampler.distribution(target)
    expected = target_probs[0, :, None] * target_probs
    torch.testing.assert_close(counts / samples, expected.expand_as(counts), rtol=0, atol=.025)
    print('Tabby job: 6000 filtered two-token continuations per rule match the target law')


def check_fused_sampling_boundary():
    from unittest.mock import patch
    from exllamav3 import ComboSampler
    from exllamav3.generator.sampler import custom

    with patch.object(custom, 'fused_sampler_enable', True):
        sampler = ComboSampler(temperature=.6, min_p=.1, top_k=4, top_p=.65, temp_last=True)
    step, = sampler.steps
    stride = 2 * 1024 * 12 + 64

    def selected_boundary(state):
        # A top-P crossing half way through the tied bin at depth 0.5.
        hist = torch.zeros((2, stride), dtype=torch.uint8)
        ctrl = hist[:, -64:].view(torch.int32)
        ctrl[:, 0] = torch.tensor(2.).view(torch.int32)
        ctrl[:, 4] = ctrl[:, 8] = 16
        ctrl[:, 10] = torch.tensor(.5).log().view(torch.int32)
        step.histograms[(state.in_logits.device, state.bsz)] = hist

    logits = torch.tensor([[2., 1.5, 1.5, 1.25, -1.], [1.5, 2., 1.25, 1.5, -1.]])
    expected = torch.tensor([[2. / .6, 1.5 / .6, 1.5 / .6, -torch.inf, -torch.inf],
                             [1.5 / .6, 2. / .6, -torch.inf, 1.5 / .6, -torch.inf]])
    expected[logits == 1.5] += torch.tensor(.5).log()
    with (patch.object(step, 'run', selected_boundary),
          patch.object(custom.ext, 'FUSED_SAMPLER_HIST_STRIDE', stride)):
        torch.testing.assert_close(sampler.distribution(logits), expected.softmax(-1))


def main():
    # TorchDispatchMode imports Dynamo. Let Torch discover its real CPU
    # environment before installing the extension-only import stubs.
    from torch import _dynamo
    install_test_stubs()
    check_fused_sampling_boundary()
    check_server_sampling_distribution()
    from exllamav3 import Config, Model, Cache
    from exllamav3.cache.fp16 import CacheLayer_fp16
    from exllamav3.generator.job import Job
    from types import SimpleNamespace
    class RecordingSampler:
        def forward(self, logits, *args, **kwargs):
            assert torch.isneginf(logits[..., 5:]).all()
            return logits.argmax(-1)
    fake_job = Job(torch.tensor([[0]]), sampler=RecordingSampler(), seed=1)
    fake_job.generator = SimpleNamespace(tokenizer=SimpleNamespace(actual_vocab_size=5))
    fake_job.current_device_ids = fake_job.device_logit_mask = None
    logits = torch.tensor([[[0., 1., 2., 3., 4., 99., 100.]]])
    assert Job.receive_logits(fake_job, logits)[0].item() == 4
    logits[..., 5:] = -torch.inf
    fake_job._check_sample_logits(logits)
    from exllamav3.generator.job import NonFiniteLogits
    for column, invalid in ((4, -torch.inf), (5, torch.inf), (6, torch.nan)):
        bad = logits.clone()
        bad[..., column] = invalid
        try:
            fake_job._check_sample_logits(bad)
        except NonFiniteLogits:
            pass
        else:
            raise AssertionError(f"invalid logit escaped detection at column {column}")
    dense = load_file(sys.argv[2])
    expected = load_file(sys.argv[3])['logits']
    class DenseHandle:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
        def forward(self, x, params): return torch.nn.functional.linear(x, dense[self.key])
    class FloatCache(CacheLayer_fp16):
        def alloc(self, device):
            super().alloc(device)
            self.k, self.v = self.k.float(), self.v.float()
    config = Config.from_directory(sys.argv[1])
    config.inkling_linear_factory = DenseHandle
    model = Model.from_config(config)
    profile_only = sys.argv[4:] == ['--profile-prefill']
    cache = Cache(model, max_num_tokens=2048 if profile_only else 1280, max_batch_size=2,
                  max_history=7, layer_type=FloatCache)
    model.load(device='cpu')
    if profile_only:
        from tools.inkling2_generate import profile_prefill
        profile_prefill(model, cache, torch.arange(2048).remainder(config.unpadded_vocab_size)[None])
        return
    os.environ['INKLING_ASSERT_NO_HTOD'] = '1'
    from exllamav3.architecture.inkling2_loop import _NoHostTensor
    # Intercept an actual CPU->CUDA request before PyTorch can dispatch it.
    try:
        with _NoHostTensor():
            torch.ops.aten._to_copy.default(torch.ones(1), device=torch.device('cuda'))
    except AssertionError as exc:
        assert 'CPU tensor' in str(exc)
    else:
        raise AssertionError('guard allowed an HtoD transfer')
    # A misplaced buffer must fail the same checked path used by generator forwards.
    model.loop.register_buffer('_test_off_device', torch.empty(1, device='meta'))
    try:
        model.forward(torch.tensor([[1]]), {'attn_mode': 'flash_attn_nc'})
    except AssertionError as exc:
        assert '_test_off_device' in str(exc)
    else:
        raise AssertionError('device-residency guard missed a misplaced buffer')
    del model.loop._test_off_device
    assert len(cache.layers) == len(cache.recurrent_layers) == 8
    assert cache.initialized
    ids = torch.tensor([[1, 3, 2, 4, 5, 6]])
    params = dict(attn_mode='flash_attn', cache=cache, past_len=0, batch_shape=(1, 256))
    whole = model.forward(ids, params)
    torch.testing.assert_close(whole, expected, rtol=1e-4, atol=1e-5)
    state = params['recurrent_states'][0]
    state.free()
    params = dict(attn_mode='flash_attn', cache=cache, past_len=0, batch_shape=(1, 256))
    outputs = []
    for i in range(ids.shape[1]):
        params['past_len'] = i
        outputs.append(model.forward(ids[:, i:i+1], params))
    torch.testing.assert_close(torch.cat(outputs, 1), expected, rtol=1e-4, atol=1e-5)
    params['recurrent_states'][0].free()

    # Interleave two independently paged requests and compare 16 greedy steps
    # to no-cache recomputation for each request. This does not run Generator.
    requests = []
    for page, prefix in enumerate((ids[:, :3], ids[:, 3:])):
        requests.append(dict(page=page, tokens=prefix.clone(), pending=prefix.clone(),
                             state=cache.get_new_state(), position=0))
    for _ in range(16):
        for request in requests:
            tokens = request['tokens']
            reference = model.forward(tokens, {'attn_mode': 'flash_attn_nc'})[:, -1].argmax(-1)
            p = dict(attn_mode='flash_attn', cache=cache,
                     cache_seqlens=torch.tensor([request['position']], dtype=torch.int32),
                     block_table=torch.tensor([[request['page']]], dtype=torch.int32),
                     recurrent_states=[request['state']])
            got = model.forward(request['pending'], p)[:, -1].argmax(-1)
            assert torch.equal(got, reference)
            request['position'] += request['pending'].shape[1]
            request['pending'] = got[:, None]
            request['tokens'] = torch.cat((tokens, got[:, None]), 1)
    for request in requests: request['state'].free()
    # 16-token local window, one 256-row ring page per slot, two shuffled
    # global pages. Cross the physical wrap with a live prefix, then fork a
    # checkpoint after the wrap and compare logits against full recomputation.
    ids = torch.arange(292).remainder(29)[None]
    whole = model.forward(ids, {'attn_mode': 'flash_attn_nc'})
    tables = torch.tensor([[1, 0]], dtype=torch.int32)
    request = cache.get_new_state()
    for key, layer in cache.layers.items():
        if key[0] in (0, 1, 2):
            assert layer.pages == 1
            assert layer.storage.k.shape[0] == cache.num_slots
            assert cache.recurrent_layers[key].states[4].shape[:2] == (cache.num_slots, 1)
        else:
            assert layer.k.shape[0] == 5
    def cached(state, end):
        start = state.position
        p = dict(attn_mode='flash_attn', cache=cache, recurrent_states=[state],
                 cache_seqlens=torch.tensor([start], dtype=torch.int32), block_table=tables)
        return model.forward(ids[:, start:end], p)
    ring_logits = []
    for end in (13, 251, 259, 271):
        start = request.position
        ring_logits.append(cached(request, end))
        torch.testing.assert_close(ring_logits[-1], whole[:, start:end], rtol=1e-4, atol=1e-5)
    saved = request.stash()
    fork = cache.new_from_stashed(saved, request.position)
    for state in (request, fork):
        torch.testing.assert_close(cached(state, 292), whole[:, 271:], rtol=1e-4, atol=1e-5)
        state.free()

    draft = Model.from_config(config, component='mtp')
    assert draft.caps.get('dspark_draft')
    draft.attach_to(model)
    draft_cache = Cache(draft, max_num_tokens=cache.max_num_tokens, max_batch_size=1)
    draft.load()
    check_ring_indices()
    check_chunked_prefill(model, cache, ids)
    check_graph_body(model, cache, ids.repeat(1, 2))
    expected_ids = check_dspark(model, cache, draft, ids[:, :253])
    from tools.inkling2_generate import generate, profile_generation, profile_prefill
    profile_prefill(model, cache, ids[:, :32])
    summary = {}
    with profile_generation(True, model, summary=summary) as measure:
        emitted, _, _ = generate(model, cache, ids[:, :253], 5, 0., measure=measure)
    assert emitted == expected_ids[:5] and summary['plain'] > 0
    check_server_generator(model, cache, draft, draft_cache, ids[:, :253], expected_ids)
    # Run the same page and ring rollback assertions with variable verification widths.
    mapped_ids = list(range(1, draft.draft.vocab_size))
    draft.draft.prepare_proposal_vocab(mapped_ids, model.loop.lm_head.weight, draft.folded_head)
    assert check_dspark(model, cache, draft, ids[:, :253]) == expected_ids

    # Repeat the same chunk schedule with full retention in every visit.
    for key, conv in cache.recurrent_layers.items():
        if conv.window_cache:
            kv = FloatCache(config, conv.module, id(cache), cache.max_num_tokens)
            kv.alloc('cpu')
            cache.layers[key] = kv
            conv.window_cache = None
    request = cache.get_new_state()
    for end, trimmed in zip((13, 251, 259, 271), ring_logits):
        retained = cached(request, end)
        torch.testing.assert_close(trimmed, retained, rtol=1e-4, atol=1e-5)
        assert torch.equal(trimmed.argmax(-1), retained.argmax(-1))
    request.free()

    print('registered config/model/load/cache/forward: passed with CPU stubs')


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
