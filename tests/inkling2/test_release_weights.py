"""CPU coverage for the release's combined safetensors text and EXL3 groups."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('inkling_release_weights_test', ROOT / 'exllamav3/architecture/inkling2_weights.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Handle:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def release(tmp_path):
    groups = {'0.self_attn.q_proj.linear': 3.5, '0.experts.gate_up_proj.0': 4}
    weights = {'model.llm.embed.weight': torch.randn(7, 16)}
    for name, bits in groups.items():
        weights.update({name + '.trellis': torch.zeros(1, 1, int(bits * 16), dtype=torch.int16),
                        name + '.suh': torch.ones(16, dtype=torch.float16),
                        name + '.svh': torch.ones(16, dtype=torch.float16),
                        name + '.mul1': torch.tensor(1, dtype=torch.int32)})
    save_file(weights, tmp_path / 'model.safetensors')
    save_file({'irrelevant': torch.ones(1)}, tmp_path / 'model-vision.safetensors')
    (tmp_path / 'quantization_config.json').write_text(json.dumps({'bits': groups}))
    config = SimpleNamespace(infer_params=SimpleNamespace(no_reconstruct=False))
    return module.Inkling2ReleaseWeights(tmp_path, config, 'cpu', Handle)


def test_release_combined_file_and_physical_handle_reuse(tmp_path):
    reader = release(tmp_path)
    assert reader.config.infer_params.no_reconstruct
    assert reader.tensor('model.llm.embed.weight').shape == (7, 16)
    key = '0.self_attn.q_proj.linear'
    first = reader.linear(key, 16, 16)
    assert first is reader.linear(key, 16, 16)
    assert first.config.infer_params.no_reconstruct
    assert first.trellis.shape == (1, 1, 56)
    assert reader.linear('0.experts.gate_up_proj.0', 16, 16) is not first
    with pytest.raises(ValueError, match='inconsistent'):
        reader.linear(key, 32, 16)
    with pytest.raises(KeyError, match='irrelevant'):
        reader.tensor('irrelevant')


def test_invalid_rate_and_missing_group_fail_closed(tmp_path):
    reader = release(tmp_path)
    reader.bits['0.self_attn.q_proj.linear'] = 3
    with pytest.raises(ValueError, match='bit rate'):
        reader.linear('0.self_attn.q_proj.linear', 16, 16)
    with pytest.raises(KeyError, match='missing release tensor'):
        reader.linear('0.experts.down_proj.0', 16, 16)


def test_real_t2_pack_header():
    path = ROOT.parent / 'packs/T2/layer-06.exl3.safetensors'
    if not path.exists():
        pytest.skip('real layer-06 pack unavailable')
    with __import__('safetensors').safe_open(path, framework='pt', device='cpu') as f:
        names = set(f.keys())
        key = '6.self_attn.q_proj.linear'
        assert {f'{key}.{field}' for field in ('trellis', 'suh', 'svh', 'mul1')} <= names
        trellis = f.get_slice(key + '.trellis')
        assert len(trellis.get_shape()) == 3


def test_build_text_from_release_and_resume(tmp_path, monkeypatch):
    """Load an exact-layout synthetic release through the same reader/builder."""
    import sys
    import types
    for name, path in [('inkling_fake', ROOT / 'exllamav3'),
                       ('inkling_fake.architecture', ROOT / 'exllamav3/architecture'),
                       ('inkling_fake.modules', ROOT / 'exllamav3/modules'),
                       ('inkling_fake.modules.arch_specific', ROOT / 'exllamav3/modules/arch_specific')]:
        pkg = types.ModuleType(name); pkg.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, pkg)
    spec = importlib.util.spec_from_file_location('inkling_fake.architecture.inkling2_weights',
                                                  ROOT / 'exllamav3/architecture/inkling2_weights.py')
    builder = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, builder)
    spec.loader.exec_module(builder)
    from inkling_fake.modules.arch_specific.inkling2_loop import RequestState
    torch.manual_seed(120)
    raw = {'architectures': ['Inkling2LoopForConditionalGeneration'], 'text_config': dict(num_hidden_layers=4, hidden_size=16, n_routed_experts=2,
        loop_config=dict(core_start=1, core_end=3, num_passes=3, gate_rank=4),
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        swa_num_attention_heads=2, swa_num_key_value_heads=2, swa_head_dim=8,
        local_layer_ids=[0, 1, 2], intermediate_size=16, num_experts_per_tok=1, vocab_size=29, unpadded_vocab_size=27,
        n_layers_per_attn_res_block=2, sconv_kernel_size=3, rel_extent=16, sliding_window_size=16,
        qk_l2_clip_threshold=1.3, rel_l2_clip_threshold=1.1)}
    raw['mtp_config'] = dict(hidden_size=16, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, intermediate_size=16, local_layer_ids=[0], sliding_window_size=16, block_size=8, markov_rank=4,
        target_layer_ids=[1, 3, 6, 8], rms_norm_eps=1e-6, rope_theta=10000., max_position_embeddings=1024,
        markov_head_type='rnn_hidden')
    w, rates, dense = {}, {}, {}
    def add(key, *shape): w['model.llm.' + key] = torch.randn(*shape) * .1
    def packed(group, inp, out):
        dense[group] = torch.randn(out, inp) * .1
        rates[group] = 3.5
        w[group + '.trellis'] = torch.zeros(inp // 16, out // 16, 56, dtype=torch.int16)
        w[group + '.suh'] = torch.ones(inp, dtype=torch.float16)
        w[group + '.svh'] = torch.ones(out, dtype=torch.float16)
        w[group + '.mul1'] = torch.tensor(1, dtype=torch.int32)
    add('embed.weight', 29, 16); add('embed_norm.weight', 16)
    add('unembed.weight', 29, 16); add('norm.weight', 16)
    add('attn_res.output_query', 16)
    add('middle_cycle_cross_pass_queries', 4, 16); add('middle_cycle_cross_pass_kv_queries', 4, 16)
    for i in range(4):
        if i: add(f'attn_res.attn_queries.{i-1}', 3 if i == 3 else 1, 16)
        add(f'attn_res.mlp_queries.{i}', 16)
        prefix = f'layers.{i}.'
        for suffix in ('q_proj.linear', 'k_proj.linear', 'v_proj.linear', 'o_proj.linear', 'o_proj.gate_proj'):
            packed(f'{i}.self_attn.{suffix}', 16, 16)
        for e in range(2):
            packed(f'{i}.experts.gate_up_proj.{e}', 16, 32)
            packed(f'{i}.experts.down_proj.{e}', 16, 16)
        for suffix, shape in [('attn.wr_du.weight', (6, 16)), ('attn.attn_sink_proj.weight', (2, 16)),
                              ('attn.q_l2_clip.scale', (2, 8)), ('attn.rel_l2_clip.scale', (2, 3)),
                              ('attn.rel_logits_proj.proj', (3, 16)), ('mlp.gate.weight', (2, 16)),
                              ('mlp.gate.bias', (2,)), ('attn_sconv.weight', (16, 1, 3)),
                              ('mlp_sconv.weight', (16, 1, 3)), ('mlp_norm.weight', (16,))]:
            add(prefix + suffix, *shape)
        if i == 3:
            for s in ('q', 'k', 'v'): add(prefix + s + '_input_norm.weight', 16)
            for s in ('k', 'v'):
                add(prefix + f'attn.{s}_sconv.static_w', 3, 16)
                add(prefix + f'attn.{s}_sconv.weight_proj', 6, 16)
        else:
            add(prefix + 'attn_norm.weight', 16)
            for s in ('k', 'v'): add(prefix + f'attn.{s}_sconv.weight', 16, 1, 3)
        if i in (1, 2):
            for s in ('attn', 'k', 'v', 'mlp'):
                add(prefix + f'controllers.{s}.down_weight', 4, 32)
                add(prefix + f'controllers.{s}.forget_weight', 16, 4)
                add(prefix + f'controllers.{s}.gain_weight', 16, 4)
    save_file(w, tmp_path / 'model.safetensors')
    (tmp_path / 'config.json').write_text(json.dumps(raw))
    (tmp_path / 'quantization_config.json').write_text(json.dumps({'bits': rates}))
    class DenseHandle(Handle):
        def forward(self, x, params): return torch.nn.functional.linear(x, dense[self.key])
    config = SimpleNamespace(infer_params=SimpleNamespace(no_reconstruct=False))
    reader = builder.Inkling2ReleaseWeights(tmp_path, config, 'cpu', DenseHandle)
    embedding, norm, model = builder.build_text_components(raw, reader, dtype=torch.float32)
    torch.testing.assert_close(embedding.weight, w['model.llm.embed.weight'])
    q = w['model.llm.middle_cycle_cross_pass_queries'].view(2, 2, 16)
    kv = w['model.llm.middle_cycle_cross_pass_kv_queries'].view(2, 2, 16)
    torch.testing.assert_close(model.message_query, torch.stack((q[:, 0], kv[:, 0], kv[:, 1], q[:, 1]), 1))
    ids = torch.tensor([[1, 3, 2, 4, 5, 6]])
    x = norm(embedding(ids))
    torch.set_num_threads(1)
    parity_spec = importlib.util.spec_from_file_location('inkling_release_parity_helper',
                                                        ROOT / 'tests/parity/test_inkling2_loop.py')
    parity = importlib.util.module_from_spec(parity_spec)
    monkeypatch.setitem(sys.modules, parity_spec.name, parity)
    parity_spec.loader.exec_module(parity)
    ref_attention = parity.attn_test.load_reference()
    ref_moe = parity.moe_test.oracle()
    refs = []
    def weight(value): return torch.nn.Parameter(value, requires_grad=False)
    for i, source in enumerate(model.layers):
        full = i == 3
        cfg = SimpleNamespace(hidden_size=16, layer_types=['hybrid' if full else 'hybrid_sliding'],
            swa_head_dim=8, head_dim=8, swa_num_attention_heads=2, num_attention_heads=2,
            swa_num_key_value_heads=2, num_key_value_heads=2, sliding_window_size=16, rel_extent=16,
            attention_dropout=0., d_rel=3, qk_l2_clip_threshold=1.3, rel_l2_clip_threshold=1.1,
            log_scaling_alpha=.1, log_scaling_n_floor=8192, sconv_kernel_size=3,
            rms_norm_eps=1e-6, _attn_implementation='eager', n_routed_experts=2,
            num_local_experts=2, moe_intermediate_size=16, num_experts_per_tok=1,
            scoring_func='sigmoid', routed_scaling_factor=8., mlp_gate_softcap=4., mlp_up_softcap=25.)
        attn = ref_attention['Inkling2LoopFullAttention' if full else 'Inkling2LoopSlidingAttention'](cfg, 0)
        pre = f'model.llm.layers.{i}.'
        for proj in ('q', 'k', 'v'):
            getattr(attn, proj + '_proj').linear.weight = weight(dense[f'{i}.self_attn.{proj}_proj.linear'])
        attn.o_proj.linear.weight = weight(dense[f'{i}.self_attn.o_proj.linear'])
        attn.o_proj.gate_proj.weight = weight(dense[f'{i}.self_attn.o_proj.gate_proj'])
        attn.q_proj.norm.scale = weight(w[pre + 'attn.q_l2_clip.scale'])
        attn.relative_position_bias.linear.weight = weight(w[pre + 'attn.wr_du.weight'])
        attn.relative_position_bias.norm.scale = weight(w[pre + 'attn.rel_l2_clip.scale'])
        attn.relative_position_bias.proj = weight(w[pre + 'attn.rel_logits_proj.proj'])
        attn.sink_proj.weight = weight(w[pre + 'attn.attn_sink_proj.weight'])
        for proj in ('k', 'v'):
            conv = getattr(attn, proj + '_proj').sconv
            key = pre + f'attn.{proj}_sconv.'
            if full:
                conv.static_weight = weight(w[key + 'static_w'])
                conv.weight_proj.weight = weight(w[key + 'weight_proj'])
            else: conv.conv1d.weight = weight(w[key + 'weight'])
        moe = ref_moe['Inkling2LoopSparseMoeBlock'](cfg)
        moe.gate.weight = weight(w[pre + 'mlp.gate.weight'])
        moe.gate.e_score_correction_bias = weight(w[pre + 'mlp.gate.bias'])
        moe.experts.gate_up_proj = weight(torch.stack([dense[f'{i}.experts.gate_up_proj.{e}'] for e in range(2)]))
        moe.experts.down_proj = weight(torch.stack([dense[f'{i}.experts.down_proj.{e}'] for e in range(2)]))
        refs.append((attn, moe, (weight(w[pre + 'attn_sconv.weight']), weight(w[pre + 'mlp_sconv.weight']))))
    with torch.no_grad():
        whole = model(x, RequestState())
        expected, _ = parity.reference(model, refs, x)
        delta = (whole - expected).abs()
        cosine = torch.nn.functional.cosine_similarity(whole.flatten(), expected.flatten(), dim=0)
        top1 = (whole.argmax(-1) == expected.argmax(-1)).float().mean()
        print(f'loaded logits max={delta.max().item():.9g} mean={delta.mean().item():.9g} '
              f'cosine={cosine.item():.9g} top1={top1.item():.6f}')
        torch.testing.assert_close(whole, expected, rtol=1e-4, atol=1e-5)
        state = RequestState()
        chunks = torch.cat([model(x[:, :3], state), model(x[:, 3:], state)], 1)
    torch.testing.assert_close(chunks, whole, rtol=1e-4, atol=1e-5)
    assert len(reader._handles) == 4 * (5 + 2 * 2)
    assert len(state.visits) == 8
    with pytest.raises(ValueError, match='batch size 1'):
        model(x.expand(2, -1, -1), RequestState())
    long_x = norm(embedding(torch.arange(1026).remainder(29)[None]))
    with torch.no_grad():
        long_whole = model(long_x, RequestState())
        state = RequestState()
        long_chunks = torch.cat((model(long_x[:, :1023], state),
                                 model(long_x[:, 1023:], state)), 1)
    torch.testing.assert_close(long_chunks, long_whole, rtol=1e-4, atol=1e-5)

    # Test the ExLlama block bridge with paged tensors and checkpointable visit
    # states. The production CUDA cache update is replaced by a CPU equivalent.
    module_stub = types.ModuleType('inkling_fake.modules.module')
    class BaseModule:
        def __init__(self, config, key, qmap):
            self.config, self.key, self.qmap = config, key, qmap
            self.device, self.caps, self.modules = None, {}, []
        def unload(self): self.device = None
    module_stub.Module = BaseModule
    monkeypatch.setitem(sys.modules, module_stub.__name__, module_stub)
    short_stub = types.ModuleType('inkling_fake.modules.short_conv')
    short_stub.ShortConvState = type('ShortConvState', (), {})
    monkeypatch.setitem(sys.modules, short_stub.__name__, short_stub)
    cache_stub = types.ModuleType('inkling_fake.cache'); cache_stub.__path__ = []
    monkeypatch.setitem(sys.modules, cache_stub.__name__, cache_stub)
    recurrent_stub = types.ModuleType('inkling_fake.cache.recurrent')
    recurrent_stub.host_copy = lambda value: value.detach().cpu().clone()
    monkeypatch.setitem(sys.modules, recurrent_stub.__name__, recurrent_stub)
    constants_stub = types.ModuleType('inkling_fake.constants'); constants_stub.PAGE_SIZE = 256
    monkeypatch.setitem(sys.modules, constants_stub.__name__, constants_stub)
    bridge_spec = importlib.util.spec_from_file_location('inkling_fake.architecture.inkling2_exl',
                                                         ROOT / 'exllamav3/architecture/inkling2_exl.py')
    bridge = importlib.util.module_from_spec(bridge_spec)
    monkeypatch.setitem(sys.modules, bridge_spec.name, bridge)
    bridge_spec.loader.exec_module(bridge)
    owner = SimpleNamespace(config=SimpleNamespace(config_dict=raw), loop=model, embedding=embedding, embed_norm=norm)
    blocks = [bridge.InklingBlock(owner, i) for i in range(4)]
    for block in blocks: block.device = torch.device('cpu')
    class FakeCache:
        def __init__(self):
            self.layers, self.recurrent_layers = {}, {}
            for visit in model.schedule:
                key = (visit.layer, visit.pass_index)
                kv = SimpleNamespace(k=torch.zeros(1, 256, 2, 8), v=torch.zeros(1, 256, 2, 8))
                kv.get_kv = lambda lengths, table, kv=kv: (kv.k, kv.v)
                def update(lengths, table, k, v, length, dest=kv):
                    start = int(lengths[0])
                    dest.k[0, start:start+length] = k[0]
                    dest.v[0, start:start+length] = v[0]
                kv.update_kv_direct = update
                self.layers[key] = kv
                conv = bridge.InklingConvLayerState(blocks[visit.layer], 1, 0, 0)
                conv.alloc('cpu')
                self.recurrent_layers[key] = conv
    fake_cache = FakeCache()
    request = SimpleNamespace(slot=0, position=0)
    outputs = []
    with torch.no_grad():
        for a, b in ((0, 3), (3, 6)):
            params = {'_inkling': {'blocks': None, 'positions': torch.arange(a, b)[None],
                                   'lengths': torch.tensor([a]), 'table': torch.tensor([[0]])}, 'cache': fake_cache,
                      'recurrent_states': [request], 'cache_seqlens': torch.tensor([a]),
                      'block_table': torch.tensor([[0]])}
            h = x[:, a:b]
            for visit in model.schedule:
                params['layer_instance'] = visit.pass_index
                h = blocks[visit.layer].forward(h, params)
            output = bridge.InklingOutput(owner).forward(h, params)
            outputs.append(output)
            request.position = b
    torch.testing.assert_close(torch.cat(outputs, 1), whole, rtol=1e-4, atol=1e-5)
    state = fake_cache.recurrent_layers[(1, 1)]
    saved = state.stash(0)
    state.clear(0)
    state.unstash(0, saved)
    for tensor, value in zip(state.states, saved):
        torch.testing.assert_close(tensor[0], value)

    # A separate interpreter can exercise the registered classes without polluting
    # the rest of pytest with extension/Triton stubs. These stubs are CPU test only.
    deps = ROOT / '.cpu-deps'
    if deps.exists() or importlib.util.find_spec('marisa_trie') is not None:
        import os
        import subprocess
        from inkling_fake.architecture.inkling2_loop_mtp import Inkling2MTPConfig
        from inkling_fake.modules.arch_specific.inkling2_dspark import Inkling2DSpark
        draft = Inkling2DSpark(Inkling2MTPConfig.from_target(raw), 29, unpadded_vocab_size=27)
        save_file(draft.state_dict(), tmp_path / 'model-mtp.safetensors')
        dense_path = tmp_path.parent / 'dense-mock.safetensors'
        save_file(dense, dense_path)
        expected_path = tmp_path.parent / 'expected.safetensors'
        save_file({'logits': whole.detach()}, expected_path)
        paths = [str(ROOT), str(deps), *os.environ.get('PYTHONPATH', '').split(os.pathsep)]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, paths)))
        subprocess.run([sys.executable, str(ROOT / 'tests/parity/inkling2_cpu_runner.py'),
                        str(tmp_path), str(dense_path), str(expected_path)],
                       check=True, env=env, timeout=30)
