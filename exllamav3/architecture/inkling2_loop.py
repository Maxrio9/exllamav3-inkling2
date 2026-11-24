"""Inkling 2 text model registration and layer-mapped execution."""
from contextlib import contextmanager, nullcontext
import math
import os
from pathlib import Path
import warnings
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from ..constants import PAGE_SIZE
from ..util.tensor import get_for_device
from ..model.config import Config
from ..model.model import Model
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence
from ..modules.arch_specific.inkling2_dspark import Inkling2DSpark, VisitTaps, confidence_width, proposal_token_ids
from .inkling2_exl import InklingBlock, InklingEmbed, InklingModule, InklingOutput, InklingState
from .inkling2_graph import DecodeGraph, DraftGraph
from .inkling2_loop_mtp import Inkling2MTPConfig
from .inkling2_weights import Inkling2ReleaseWeights, build_text_components


class _NoHostTensor(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        tensors = [x for x in tree_leaves((args, kwargs)) if isinstance(x, torch.Tensor)]
        target = kwargs.get('device')
        cuda = (target is not None and torch.device(target).type == 'cuda') or any(t.is_cuda for t in tensors)
        if cuda and any(t.device.type == 'cpu' for t in tensors):
            raise AssertionError(f'Inkling CPU tensor in {func}')
        return func(*args, **kwargs)


@contextmanager
def _no_sync():
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode('error')
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)


def visit_plan(raw):
    text = raw['text_config']
    n = text['num_hidden_layers']
    loop = text['loop_config']
    start, end, count = (loop[k] for k in ('core_start', 'core_end', 'num_passes'))
    if any(type(x) is not int for x in (n, start, end, count)) or not 0 <= start < end <= n or count < 1 or loop.get('is_varloop', False):
        raise ValueError('Invalid Inkling loop configuration')
    return list(range(start)) + list(range(start, end)) * count + list(range(end, n))


class Inkling2LoopConfig(Config):
    arch_string = 'Inkling2LoopForConditionalGeneration'

    def __init__(self, directory, **kwargs):
        super().__init__(directory, {'text': Inkling2LoopModel}, config_key='text_config', **kwargs)
        # The checkpoint has no max_position_embeddings; the base default (8192) would cap context.
        if 'max_position_embeddings' not in self.config_dict.get('text_config', {}) and 'max_position_embeddings' not in self.config_dict:
            text = self.config_dict.get('text_config', {})
            self.max_position_embeddings = int(text.get('model_max_length') or self.config_dict.get('model_max_length') or 1048576)
        self.layer_map = visit_plan(self.config_dict)
        self.num_hidden_layers = self.config_dict['text_config']['num_hidden_layers']
        self.hidden_size = self.config_dict['text_config']['hidden_size']
        self.num_experts = self.config_dict['text_config']['n_routed_experts']
        self.head_dim = self.config_dict['text_config']['head_dim']
        self.num_q_heads = self.config_dict['text_config']['num_attention_heads']
        self.num_kv_heads = self.config_dict['text_config']['num_key_value_heads']
        self.unpadded_vocab_size = self.config_dict['text_config'].get('unpadded_vocab_size') or self.vocab_size
        self.infer_params.no_reconstruct = True
        if self.config_dict.get('mtp_config'):
            for name in ('model-mtp.safetensors', 'mtp.safetensors'):
                path = Path(directory) / name
                if path.is_file():
                    self.dspark_path = path
                    self.model_classes['mtp'] = Inkling2DSparkModel
                    break


class Inkling2LoopModel(Model):
    config_class = Inkling2LoopConfig
    verify_rows_default = False

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.caps.update(supports_tp=False, recurrent_states=True,
                         default_recurrent_checkpoint_interval=256)
        self.recurrent_state_cls = InklingState
        self.embedding = self.embed_norm = self.loop = self.reader = None
        self.modules = ([InklingEmbed(self, 'model.llm.embed')] +
                        [InklingBlock(self, i) for i in range(config.num_hidden_layers)] +
                        [InklingOutput(self)])
        self.first_block_idx = 1
        self.last_kv_module_idx = config.num_hidden_layers
        self.logit_layer_idx = len(self.modules) - 1

    def load_components(self, device):
        if self.loop is not None:
            if self.embedding.weight.device != device:
                raise ValueError('Inkling requires all modules on one device')
            return
        self.reader = Inkling2ReleaseWeights(self.config.directory, self.config, device,
                                             getattr(self.config, 'inkling_linear_factory', None))
        dtype = torch.float32 if device.type == 'cpu' else torch.float16
        with self.reader.mapped():
            self.embedding, self.embed_norm, self.loop = build_text_components(self.config.config_dict, self.reader, dtype)

        # Bit-exact row-wise verification is opt-in (INKLING_VERIFY_ROWS=1): it matches width-1 decode
        # exactly but reads the weights once per row, so DSpark loses its speed-up (REPORT-p4.md).
        self.verify_rows_default = False
        if device.type == 'cuda' and getattr(self.config, 'inkling_int8_head', False):
            self.prepare_int8_head()

    def prepare_int8_head(self):
        from ..modules.arch_specific.inkling2_head import pack_int8
        if not hasattr(self.loop, 'head_codes'):
            codes, scales = pack_int8(self.loop.lm_head.weight[:self.config.unpadded_vocab_size])
            self.loop.register_buffer('head_codes', codes, persistent=False)
            self.loop.register_buffer('head_scales', scales, persistent=False)

    def load(self, *args, **kwargs):
        if not args and 'device' not in kwargs:
            kwargs['device'] = 'cuda:0'
        if kwargs.get('tensor_p') or kwargs.get('reserve_per_device') or kwargs.get('use_per_device'):
            raise ValueError('Inkling currently requires a single explicit device')
        return super().load(*args, **kwargs)

    def unload(self):
        super().unload()
        self.embedding = self.embed_norm = self.loop = self.reader = None

    def forward(self, input_ids, params=None):
        checked = self.embedding.weight.is_cuda and os.environ.get('INKLING_ASSERT_NO_SYNC') == '1'
        with _no_sync() if checked else nullcontext():
            return super().forward(input_ids, params)

    def run_section(self, x, params, a, b, prefill):
        if os.environ.get('INKLING_ASSERT_NO_HTOD') == '1':
            device = x.device
            for module in (self.embedding, self.embed_norm, self.loop):
                for name, t in (*module.named_parameters(), *module.named_buffers()):
                    if t.device != device:
                        raise AssertionError(f'Inkling {name} on {t.device}, expected {device}')
            for key, handle in self.reader._handles.items():
                for name, t in vars(handle).items():
                    if isinstance(t, torch.Tensor) and t.device != device:
                        raise AssertionError(f'Inkling {key}.{name} on {t.device}, expected {device}')
            cache = params.get('cache')
            if cache is not None:
                tensors = cache.get_all_tensors()
                for layer in cache.recurrent_layers.values():
                    tensors.extend(layer.get_state_tensors())
                if any(t.device != device for t in tensors):
                    raise AssertionError(f'Inkling cache tensor outside {device}')
            tap_cache = params.get('inkling_tap_cache')
            if tap_cache is not None and tap_cache.rows.device != device:
                raise AssertionError(f'Inkling tap cache outside {device}')
            if x.is_cuda:
                with _NoHostTensor():
                    return self._decode_section(x, params, a, b, prefill)
        return self._decode_section(x, params, a, b, prefill)

    def _decode_section(self, x, params, a, b, prefill):
        run = super().run_section
        states = params.get('recurrent_states')
        taps = params.get('inkling_tap_collector')
        # Both the verification and the retained prefix use a stable width.
        width = x.shape[1]
        draft_width = (self.config.config_dict.get('mtp_config') or {}).get('block_size')
        if (not x.is_cuda or prefill or a != 0 or b != len(self.fwd_modules)
                or not states or not states[0].position or params.get('cache') is not states[0].cache
                or (width != 1 and (taps is None or not draft_width or width > draft_width))
                or (taps is not None and type(taps) is not VisitTaps)
                or os.environ.get('EXL3_GRAPHS') == '0'):
            return run(x, params, a, b, prefill)
        state = states[0]
        ctx = params['_inkling']
        tap_cache = params.get('inkling_tap_cache')
        signature = (weakref.ref(self.loop), x.dtype, ctx['table'].shape, state.slot,
                     params.get('last_tokens_only') if width > 1 else None, taps.indices if taps else None,
                     bool(params.get('_inkling_verify_rows')), bool(params.get('_inkling_record_conv')),
                     bool(getattr(self.config, 'inkling_int8_head', False)),
                     tap_cache.rows.data_ptr() if tap_cache is not None else None,
                     torch.cuda.current_stream(x.device).cuda_stream)
        if state.decode_graphs is None:
            state.decode_graphs = {}
        key = width, params.get('last_tokens_only') if width > 1 else None
        previous, entry = state.decode_graphs.get(key, (None, None))
        if previous != signature:
            try:
                entry = DecodeGraph(x, ctx)
            except RuntimeError as exc:
                entry = None
                warnings.warn(f'Inkling CUDA graph setup failed, using eager decode: {exc}', stacklevel=2)
            state.decode_graphs[key] = signature, entry
        if entry is None:
            return run(x, params, a, b, prefill)

        def forward(ids, local):
            return run(ids, local, a, b, prefill)
        return entry.run(forward, x, params)

    def prepare_inputs(self, input_ids, params):
        if input_ids.shape[0] != 1:
            raise ValueError('Inkling currently supports batch size 1 only')
        device = self.embedding.weight.device
        input_ids = prepare_for_attn(input_ids, params, metadata_device=device)
        prepare_for_recurrence(input_ids, params, self)
        # Generator-owned token IDs and page assignments enter once at the request
        # boundary. The generator may already have staged a device view of its table.
        input_ids = input_ids.to(device, non_blocking=True)
        ctx = params['_inkling'] = {}
        past = 0
        if params.get('cache') is not None:
            states = params.get('recurrent_states')
            if not states or len(states) != 1:
                raise ValueError('Inkling requires one request-owned recurrent state')
            past = states[0].position
            if device.type == 'cuda' and past and input_ids.shape[1] <= 8:
                if states[0].window_workspace is None:
                    states[0].window_workspace = {}
                ctx['window_workspace'] = states[0].window_workspace
            lengths, table = params.get('cache_seqlens'), params.get('block_table')
            if lengths is None or table is None or lengths.shape != (1,) or table.ndim != 2 or table.shape[0] != 1:
                raise ValueError('Inkling requires one paged cache length and block table row')
            if lengths.device.type == 'cpu':
                if int(lengths[0]) != past:
                    raise ValueError('Inkling recurrent and KV positions differ')
            elif params.get('batch_shape') is None or params.get('past_len') is None:
                torch._assert_async((lengths == past).all(), 'Inkling recurrent and KV positions differ')
            # Staged views are scoped to this request. Direct callers may reuse
            # and mutate a host table between calls, so do not memoize their uploads.
            staged = params.get('dev_cache', {})
            table = staged.get((id(table), device), (None, table))[1]
            lengths = staged.get((id(lengths), device), (None, lengths))[1]
            ctx['lengths'] = (lengths.to(device=device, dtype=torch.int32).contiguous() if lengths.device == device
                              else torch.full((1,), past, dtype=torch.int32, device=device))
            ctx['table'] = table.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
            if past + input_ids.shape[1] > table.shape[1] * PAGE_SIZE:
                raise ValueError('Inkling input exceeds supplied block table')
        elif device.type == 'cuda':
            ctx['lengths'] = torch.zeros(1, device=device, dtype=torch.int32)
            ctx['table'] = ctx['lengths'][None]
        if '_inkling_positions' in params:
            ctx['positions'] = params['_inkling_positions']
        elif input_ids.shape[1] == 1 and 'lengths' in ctx:
            ctx['positions'] = ctx['lengths'][None]
        else:
            ctx['positions'] = torch.arange(past, past + input_ids.shape[1], device=device)[None]
        tap_cache = params.get('inkling_tap_cache')
        if tap_cache is not None:
            params['inkling_tap_collector'] = VisitTaps(len(self.config.layer_map), tap_cache.indices)
        # Only a short continuation with draft taps is a verifier. Initial and
        # chunked prefill must keep the plain prompt arithmetic.
        params['_inkling_verify_rows'] = (
            1 < input_ids.shape[1] <= 8 and past > 0 and input_ids.is_cuda
            and params.get('inkling_tap_collector') is not None
            and not params.get('_inkling_prompt_chunk') and not params.get('prefill')
            and os.environ.get('INKLING_VERIFY_ROWS', '1' if self.verify_rows_default else '0') == '1'
        )
        if params.get('recurrent_history'):
            state = params['recurrent_states'][0]
            length = input_ids.shape[1]
            if length - 1 > state.cache.max_history:
                raise ValueError('Inkling verification exceeds the configured history')
            params['_inkling_record_conv'] = length > 1 and state.can_retain(length)
            state.verification = (past, input_ids, ctx['table'], tap_cache, state.checkpoint(length))
        return input_ids


class InklingTapCache:
    """BF16 feature rows follow the target's global page assignments in Generator."""
    def __init__(self, config, module, cache_id, max_num_tokens):
        cfg = module.owner.draft_config
        self.attention, self.cache_id = module, cache_id
        self.indices = cfg.target_layer_ids
        self.shape = (max_num_tokens // PAGE_SIZE, PAGE_SIZE, len(self.indices), cfg.hidden_size)
        self.rows = None

    def alloc(self, device):
        self.rows = torch.empty(self.shape, dtype=torch.bfloat16, device=device)

    def free(self):
        self.rows = None

    def copy_page(self, source, from_page, to_page, num_tokens):
        self.rows[to_page, :num_tokens].copy_(source.rows[from_page, :num_tokens])

    def get_tensors(self):
        return [self.rows]

    def storage_size(self):
        return math.prod(self.shape) * torch.bfloat16.itemsize

    def overhead_size(self):
        return 0

    def addresses(self, table, positions):
        return table[0, positions // PAGE_SIZE], positions % PAGE_SIZE


class InklingDraftModule(InklingModule):
    def __init__(self, owner, key):
        super().__init__(owner, key)
        self.layer_idx, self.cache_layers = 0, []
        self.caps['kv_cache'] = True

    def cache_layer_type(self, layer_type, kwargs):
        return InklingTapCache, {}

    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        for cache in self.cache_layers:
            cache.alloc(self.device)

    def unload(self):
        for cache in self.cache_layers:
            cache.free()
        super().unload()

    def forward(self, x, params, out_dtype=None):
        return self.owner.propose(x, **params)


class Inkling2DSparkModel(Model):
    """Load a separate draft component that shares the loaded Inkling target."""
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.caps.update(supports_tp=False, attach_target=True, dspark_draft=True)
        self.draft_config = Inkling2MTPConfig.from_target(config.config_dict)
        self.caps['default_draft_size'] = self.draft_config.block_size - 1
        self.modules = [InklingDraftModule(self, 'inkling2_dspark')]
        self.target = self.draft = self.folded_head = None

    def get_layer_instances(self, layer_idx):
        # The draft has one feature cache, independent of the target layer map.
        return [(layer_idx, 0)]

    @torch.inference_mode()
    def propose_for_job(self, job, cache, window, distribution=None):
        seq, = job.sequences
        start = seq.kv_position
        cfg = self.draft_config
        if not start or start + cfg.block_size > cfg.max_position_embeddings:
            return None
        count = min(window, cfg.block_size - 1, job.max_new_tokens - job.new_tokens - 1,
                    len(seq.allocated_pages) * PAGE_SIZE - start - 1,
                    self.target.config.max_position_embeddings - start - 1)
        if count <= 0:
            return None
        taps = cache.layers[(0, 0)]
        device = taps.rows.device
        positions = torch.arange(start + cfg.block_size, device=device)[None]
        # Generator replaces this tensor whenever page assignments change.
        seq.block_index_tensor._static_dev_cache = True
        table = get_for_device({'table': seq.block_index_tensor}, 'table', device)
        features = taps.rows[taps.addresses(table, positions[:, :start])]
        anchor = seq.sequence_ids.torch_slice(start, start + 1).reshape(1).to(device)
        use_vocab = self.draft.vocab_size <= job.generator.tokenizer.actual_vocab_size
        gain = job.dspark_gain
        if gain is not None:
            from ..generator.sampler.custom import CustomSampler, SS_Fused
            sampler = job.sampler
            gain.confidence_width = (use_vocab and hasattr(self.draft, 'proposal_ids')
                                     and isinstance(sampler, CustomSampler) and type(sampler).forward is CustomSampler.forward
                                     and sampler.fused_only
                                     and sampler.steps[0].mode == SS_Fused.MODE_GREEDY)
        choose_width = gain is not None and gain.choose_width(count)
        ids, _, probs = self.propose(
            anchor, features, positions, num_proposals=count, state=job.recurrent_state.target,
            tokens_only=distribution is None and not choose_width, sample=distribution is not None,
            generator=job.dspark_rng(device) if distribution is not None else None,
            distribution=distribution,
            use_draft_vocab=use_vocab,
        )
        # Generator owns tokenization, target sampling and its host page records.
        if choose_width:
            staged = torch.cat((ids[0], confidence_width(probs).view(1))).cpu()
            return staged[None, :int(staged[-1])], None
        return ids.cpu(), (ids, probs) if distribution is not None else None

    def attach_to(self, target):
        if not isinstance(target, Inkling2LoopModel) or target.config is not self.config:
            raise ValueError('DSpark requires the Inkling target from the same config')
        if self.draft is not None and self.target is not target:
            raise ValueError('Unload DSpark before attaching a different target')
        self.target = target

    def load(self, device=None, **kwargs):
        if self.target is None or self.target.loop is None:
            raise ValueError('Load and attach the Inkling target before loading DSpark')
        return super().load(device=device or self.target.embedding.weight.device, **kwargs)

    def load_components(self, device):
        if self.target is None or self.target.loop is None or self.target.embedding.weight.device != device:
            raise ValueError('DSpark and its loaded Inkling target must be on the same device')
        head = self.target.loop.lm_head.weight
        if self.target.embedding.num_embeddings != head.shape[0]:
            raise ValueError('Inkling embedding and head have different padded vocabularies')
        draft = Inkling2DSpark(self.draft_config, head.shape[0], softcap=self.target.loop.softcap,
                              unpadded_vocab_size=self.config.unpadded_vocab_size)
        draft.load_mtp(self.config.dspark_path)
        draft = draft.to(device).eval()
        draft.prepare_inference()
        self.folded_head = draft.markov_head.fold(head[:draft.vocab_size])
        ids = proposal_token_ids(self.config.directory, draft.vocab_size)
        if ids is not None:
            draft.prepare_proposal_vocab(ids, head, self.folded_head)
        self.draft = draft

    def unload(self):
        super().unload()
        self.draft = self.folded_head = None

    def propose(self, anchor, features, positions, *, state=None, **kwargs):
        start = features.shape[1]
        guard = os.environ.get('INKLING_ASSERT_NO_HTOD') == '1'
        if guard:
            device = self.target.embedding.weight.device
            for t in (*self.draft.parameters(), *self.draft.buffers(), *self.folded_head,
                      anchor, features, positions):
                if t.device != device:
                    raise AssertionError(f'Inkling DSpark tensor on {t.device}, expected {device}')
        checked = anchor.is_cuda and os.environ.get('INKLING_ASSERT_NO_SYNC') == '1'
        with (_NoHostTensor() if guard and anchor.is_cuda else nullcontext(),
              _no_sync() if checked else nullcontext()):
            if state is not None:
                return self._request_proposals(state, anchor, features, positions, **kwargs)
            return self.draft.propose(
                anchor, positions[:, start:start + self.draft_config.block_size], features, positions[:, :start],
                self.target.embedding, self.target.embed_norm, self.target.loop.lm_head.weight,
                folded_head=self.folded_head, **kwargs,
            )

    def _request_proposals(self, state, anchor, features, positions, *, num_proposals,
                           sample=False, generator=None, transform_logits=None, tokens_only=False, distribution=None,
                           use_draft_vocab=False):
        if not 0 < num_proposals < self.draft_config.block_size:
            raise ValueError('invalid number of draft proposals')
        # Grow geometrically. Keep just the current graph and cap padding at the
        # round's bucket size, independent of the maximum configured context.
        length = features.shape[1]
        capacity = max(32, 1 << (length - 1).bit_length())
        greedy = not sample and transform_logits is None and distribution is None
        stream = torch.cuda.current_stream(anchor.device).cuda_stream if anchor.is_cuda else None
        key = (id(self.draft), capacity, min(length, self.draft_config.block_size), greedy, tokens_only,
               use_draft_vocab, stream)
        entry = state.draft_graph
        if entry is None or state.draft_signature != key:
            entry = state.draft_graph = DraftGraph(self.draft, anchor, features, capacity)
            state.draft_signature = key
        entry.stage(self.draft, anchor, features, positions)
        head = self.target.loop.lm_head.weight

        def body():
            entry.update_recent(self.draft)
            hidden = self.draft.forward_block(entry.anchor, entry.positions, entry.kv, entry.context_positions,
                                              self.target.embedding, self.target.embed_norm)
            if greedy:
                return self.draft.proposal_head(hidden, entry.anchor, entry.last, head, self.folded_head,
                                                self.draft_config.block_size - 1, tokens_only=tokens_only,
                                                use_draft_vocab=use_draft_vocab)
            return (hidden,)

        before = entry.replays
        outputs = entry.run(body, anchor.is_cuda and os.environ.get('EXL3_GRAPHS') != '0')
        state.draft_replays += entry.replays - before
        if tokens_only and greedy:
            return outputs[0][:, :num_proposals], None, None
        if greedy:
            return tuple(t[:, :num_proposals] for t in outputs)
        return self.draft.proposal_head(outputs[0], anchor, entry.last, head, self.folded_head, num_proposals,
                                        sample, generator, transform_logits, distribution=distribution,
                                        use_draft_vocab=use_draft_vocab)


from .inkling2_schedule import InklingVisit, inkling2_visit_schedule, inkling2_schedule_from_config, build_inkling2_loop  # noqa: F401
