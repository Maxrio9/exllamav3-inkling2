"""ExLlama module and cache adapters for the Inkling text components."""
from functools import partial
import math
import torch

from ..modules.module import Module
from ..modules.short_conv import ShortConvState
from ..cache.recurrent import host_copy
from ..constants import PAGE_SIZE
from ..modules.arch_specific.inkling2_attention import Inkling2AttentionState
from ..modules.arch_specific.inkling2_conv import Inkling2ConvState
from ..modules.arch_specific.inkling2_loop import decode_rows, rms
from .inkling2_weights import Inkling2ReleaseWeights, build_text_components


class InklingWindowCache:
    """SWA pages belong to a recurrent request slot, including on prefix restore."""
    def __init__(self, config, attention, cache_id, max_num_tokens, storage_type, **kwargs):
        self.attention, self.cache_id = attention, cache_id
        self.pages = min(max_num_tokens // PAGE_SIZE, math.ceil(attention.window / PAGE_SIZE))
        self.make_storage = partial(storage_type, config, attention, cache_id, **kwargs)
        self.storage = None
        self.table = None

    @torch.inference_mode(False)
    def alloc_slots(self, device, count):
        self.storage = self.make_storage(count * self.pages * PAGE_SIZE)
        self.storage.alloc(device)
        pages = torch.arange(count * self.pages, dtype=torch.int32, device=device).view(count, self.pages)
        self.table = pages.repeat(1, 2)

    def layout(self, ctx, past, seq_len, slot):
        # Every local visit with this geometry uses the same logical positions.
        key = (self.pages, self.attention.window, slot)
        layouts = ctx.setdefault('windows', {})
        if key not in layouts:
            window = self.attention.window
            first_page = max(0, past - window + 1) // PAGE_SIZE
            base = first_page * PAGE_SIZE
            write_table = self.table[slot:slot+1]
            if ctx.get('window_workspace') is not None:
                # The first and last logical pages can alias one physical page.
                # The same buffers serve every local visit of this geometry.
                pos = ctx['lengths']
                buffers = ctx['window_workspace']
                if key not in buffers:
                    buffers[key] = (torch.empty_like(write_table[:, :self.pages + 1]),
                                    torch.empty_like(pos), torch.empty_like(pos), torch.empty_like(pos))
                read_table, first, read_length, write_length = buffers[key]
                torch.sub(pos, window - 1, out=first)
                first.clamp_min_(0).floor_divide_(PAGE_SIZE)
                torch.add(write_table[:, :self.pages + 1], first, out=read_table)
                read_table.remainder_(self.pages).add_(slot * self.pages)
                torch.mul(first, PAGE_SIZE, out=read_length)
                torch.sub(pos, read_length, out=read_length)
                skip = max(0, seq_len - self.pages * PAGE_SIZE)
                torch.add(pos, skip, out=write_length)
                write_length.remainder_(self.pages * PAGE_SIZE)
                layouts[key] = base, read_length, read_table, skip, write_length, write_table
                return layouts[key]
            read_table = write_table[:, first_page % self.pages:]
            read_length = ctx['lengths'] - base if base else ctx['lengths']
            # Attend to the chunk before overwriting the ring. A long prefill
            # only needs to commit its last physical ring, even across multiple wraps.
            skip = max(0, seq_len - self.pages * PAGE_SIZE)
            start = (past + skip) % (self.pages * PAGE_SIZE)
            write_length = read_length if start == past - base else read_length.new_full((1,), start)
            layouts[key] = base, read_length, read_table, skip, write_length, write_table
        return layouts[key]

    def free(self):
        if self.storage is not None:
            self.storage.free()
        self.storage = self.table = None

    def copy_page(self, source, from_page, to_page, num_tokens):
        # Logical prompt pages contain only global KV. The recurrent checkpoint owns SWA.
        pass

    def get_tensors(self):
        return []

    def storage_size(self):
        return self.storage.storage_size() if self.storage is not None else 0

    def overhead_size(self):
        return self.make_storage(self.pages * PAGE_SIZE).overhead_size()


class InklingState(ShortConvState):
    verification = None
    graph_replays = draft_replays = 0
    draft_graph = draft_signature = verify_inputs = None
    decode_graphs = moe_workspace = window_workspace = None

    def free(self):
        self.draft_graph = self.verify_inputs = None
        self.decode_graphs = self.moe_workspace = self.window_workspace = None
        super().free()

    @staticmethod
    def copy_rows(pairs):
        # FP32 conv histories and FP16 KV need separate foreach lists for the
        # CUDA multi-tensor copy kernel. All destinations are disjoint.
        groups = {}
        for dst, src in pairs:
            outputs, inputs = groups.setdefault(dst.dtype, ([], []))
            outputs.append(dst)
            inputs.append(src)
        for outputs, inputs in groups.values():
            torch._foreach_copy_(outputs, inputs)

    def unstash(self, stashed):
        self.draft_graph = None
        super().unstash(stashed)

    def checkpoint(self, length):
        return {key: layer.stash(self.slot, self.position, length=length)
                for key, layer in self.cache.recurrent_layers.items()}

    def restore(self, position, length, checkpoint):
        if self.draft_graph is not None:
            self.draft_graph.length = min(self.draft_graph.length, position)
        self.position, self.last_history = position, 0
        for key, saved in checkpoint.items():
            self.cache.recurrent_layers[key].unstash(self.slot, saved, position, length=length)

    def can_retain(self, length):
        return all(layer.window_cache is None or layer.window_cache.attention.window >= length
                   for layer in self.cache.recurrent_layers.values())

    def retained_copies(self, key, position, length, keep, originals):
        """Repair conv histories and only the rejected suffix of each local KV plane."""
        layer = self.cache.recurrent_layers[key]
        segments = layer.checkpoint_segments(self.slot, position, length)
        for i, ((view, first), saved) in enumerate(zip(segments, originals, strict=True)):
            if i < 4:
                history = len(view)
                fresh = min(keep, history)
                if fresh < history:
                    yield view[:history - fresh], saved[fresh:]
                if fresh:
                    yield view[history - fresh:], self.verify_inputs[(key, length)][i][keep - fresh:keep]
            else:
                offset = max(0, keep - first)
                if offset < len(view):
                    yield view[offset:], saved[offset:]

    def rewind(self, num_tokens):
        if self.verification is None:
            return super().rewind(num_tokens)
        past, ids, table, tap_cache, checkpoint = self.verification
        self.verification = None
        if num_tokens:
            length = ids.shape[1]
            if not 0 < num_tokens < length:
                raise ValueError('Invalid Inkling verification rewind')
            if self.can_retain(length):
                keep = length - num_tokens
                copies = [pair for key, saved in checkpoint.items()
                          for pair in self.retained_copies(key, past, length, keep, saved)]
                self.copy_rows(copies)
                self.position, self.last_history = past + keep, 0
                return
            self.restore(past, length, checkpoint)
            # Rebuild the retained rows, including tap features for the next
            # draft. Global rows beyond the new position are no longer visible.
            params = dict(attn_mode='flash_attn', cache=self.cache, block_table=table,
                          cache_seqlens=table.new_full((1,), past), recurrent_states=[self])
            prefix = ids[:, :length - num_tokens]
            if tap_cache is None:
                self.cache.model.prefill(prefix, params)
            else:
                params.update(inkling_tap_cache=tap_cache, last_tokens_only=1)
                self.cache.model.forward(prefix, params)
        self.last_history = 0

    def can_stash_at(self, position):
        return self.position == position and not self.last_history

    def reset(self):
        self.draft_graph = None
        self.verification = None
        self.last_history = 0
        super().reset()
        for layer in self.cache.get_all_recurrent_layers().values():
            layer.clear(self.slot)


class InklingConvLayerState:
    """Four convolution histories for a physical layer instance and cache slot."""
    def __init__(self, module, max_batch_size, max_history, cache_id):
        self.module, self.cache_id = module, cache_id
        self.max_batch_size, self.device = max_batch_size, None
        self.window_cache = self.kv_rotation = None
        self.states = None
        self.shapes = [(max_batch_size, module.kernel_size - 1, width)
                       for width in (module.kv_width, module.kv_width, module.hidden_size, module.hidden_size)]

    def get_checkpoint_size(self):
        if self.states is not None:
            return sum(s[0].numel() * s.element_size() for s in self.states)
        return sum(math.prod(s[1:]) * 4 for s in self.shapes)

    def storage_size(self):
        return sum(math.prod(s) * 4 for s in self.shapes)

    def alloc(self, device):
        self.device = torch.device(device)
        gb10 = self.device.type == 'cuda' and torch.cuda.get_device_capability(self.device) == (12, 1)
        self.copy_to_host = partial(host_copy, populate=True) if gb10 else host_copy
        # Cache slots are cleared and restored outside model.forward's inference
        # context, including when a new job is enqueued.
        with torch.inference_mode(False):
            self.states = [torch.zeros(s, dtype=torch.float32, device=self.device) for s in self.shapes]
            if self.window_cache:
                # Slot-major views keep all mutable request state visible to
                # checkpoint and transaction callers, including the quantization scales.
                for plane in self.window_cache.storage.get_tensors():
                    self.states.append(plane.unflatten(0, (self.max_batch_size, self.window_cache.pages)))

    def free(self):
        self.states, self.device = None, None
        self.window_cache = self.kv_rotation = None

    def clear(self, slot):
        if self.states is not None:
            for state in self.states:
                state[slot].zero_()

    def get_state_tensors(self):
        return tuple(self.states or ())

    def checkpoint_segments(self, slot, position, length):
        for i, state in enumerate(self.states):
            if length is None or i < 4:
                yield state[slot], 0
            else:
                rows = state[slot].flatten(0, 1)
                count = min(length, len(rows))
                start = (position + length - count) % len(rows)
                first = min(count, len(rows) - start)
                yield rows[start:start + first], length - count
                if first < count:
                    yield rows[:count - first], length - count + first

    def checkpoint_views(self, slot, position, length):
        return (view for view, _ in self.checkpoint_segments(slot, position, length))

    def stash(self, slot, position=0, *, length=None):
        # Persistent prefix checkpoints live on the host. Verification only
        # protects the rows a single forward can overwrite, on their device.
        copy = self.copy_to_host if length is None else torch.clone
        return tuple(copy(view) for view in self.checkpoint_views(slot, position, length))

    def unstash(self, slot, stashed, position=0, *, length=None):
        for view, source in zip(self.checkpoint_views(slot, position, length), stashed, strict=True):
            view.copy_(source)

    def rewind(self, slot, last_history, num_tokens):
        if num_tokens:
            raise ValueError('Inkling convolution state cannot rewind; restore a checkpoint')


class InklingModule(Module):
    def __init__(self, owner, key):
        super().__init__(owner.config, key, None)
        self.owner = owner

    def optimizer_targets(self):
        return []

    def can_defer_load(self):
        return False

    def load(self, device, **kwargs):
        device = torch.device(device)
        self.owner.load_components(device)
        self.device = device


class InklingEmbed(InklingModule):
    def forward(self, x, params, out_dtype=None):
        if x.shape[0] != 1:
            raise ValueError('Inkling currently supports batch size 1 only')
        params['_inkling'].update(blocks=None, visit=0)
        return decode_rows(self.owner.embed_norm, self.owner.embedding(x), params.get('_inkling_verify_rows'))


class InklingBlock(InklingModule):
    layer_state_cls = InklingConvLayerState

    def __init__(self, owner, index):
        super().__init__(owner, f'model.llm.layers.{index}')
        self.layer_idx = index
        text = self.config.config_dict['text_config']
        local = index in text.get('local_layer_ids', [i for i in range(text['num_hidden_layers']) if (i + 1) % 6])
        self.num_kv_heads = text['swa_num_key_value_heads'] if local else text['num_key_value_heads']
        self.head_dim = text['swa_head_dim'] if local else text['head_dim']
        self.window = text.get('sliding_window_size', 1024) if local else 0
        self.kv_width = self.num_kv_heads * self.head_dim
        self.hidden_size = text['hidden_size']
        self.kernel_size = text.get('sconv_kernel_size', 4)
        self.cache_layers, self.recurrent_layers = [], []
        self.caps.update(kv_cache=True, recurrent_cache=True)

    def cache_layer_type(self, layer_type, kwargs):
        if self.window:
            return InklingWindowCache, dict(kwargs, storage_type=layer_type)
        return layer_type, kwargs

    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        for kv, conv in zip(self.cache_layers, self.recurrent_layers, strict=True):
            if self.window:
                kv.alloc_slots(self.device, conv.max_batch_size)
                conv.window_cache = kv
            else:
                kv.alloc(self.device)
            storage = kv.storage if self.window else kv
            if (self.device.type == 'cuda' and hasattr(storage, 'get_qkv')
                    and not storage.compand_a and self.head_dim % 32 == 0):
                from ..modules.attention_fn.triton_paged import _get_h32
                conv.kv_rotation = _get_h32(self.device)
            conv.alloc(self.device)

    def unload(self):
        for layer in self.cache_layers: layer.free()
        for layer in self.recurrent_layers: layer.free()
        super().unload()

    def _state(self, params, instance, seq_len):
        cache = params.get('cache')
        if cache is None:
            ctx = params['_inkling']
            attention = Inkling2AttentionState()
            if self.device.type == 'cuda':
                attention.paged = None, None, ctx['table'], ctx['lengths'], 0, 0
            return {'attention': {'positions': ctx['positions'], 'cache': attention}}, None
        states = params.get('recurrent_states')
        if states is None or len(states) != 1:
            raise ValueError('Inkling requires one request-owned recurrent state')
        ctx = params['_inkling']
        lengths, table, past = ctx['lengths'], ctx['table'], states[0].position
        kv = cache.layers[(self.layer_idx, instance)]
        conv = cache.recurrent_layers[(self.layer_idx, instance)]
        slot = states[0].slot
        attention = Inkling2AttentionState()
        base, skip = 0, 0
        write_length, write_table = lengths, table
        if conv.window_cache:
            base, lengths, table, skip, write_length, write_table = kv.layout(ctx, past, seq_len, slot)
            kv = kv.storage
        if past and conv.kv_rotation is not None:
            keys, sk, values, sv, kb, vb = kv.get_qkv()
            attention.quant = sk, sv, kb, vb, conv.kv_rotation
        else:
            keys, values = kv.get_kv(lengths, table) if past else (None, None)
        attention.workspace = ctx.get('window_workspace')
        attention.paged = keys, values, table, lengths, past - base, base
        attention.conv.slots = {i: conv.states[i][slot:slot+1] for i in (0, 1)}
        ac, mc = Inkling2ConvState(), Inkling2ConvState()
        ac.slots[2], mc.slots[3] = conv.states[2][slot:slot+1], conv.states[3][slot:slot+1]
        if params.get('_inkling_record_conv'):
            key = (self.layer_idx, instance), seq_len
            if states[0].verify_inputs is None:
                states[0].verify_inputs = {}
            saved = states[0].verify_inputs
            if key not in saved:
                saved[key] = tuple(t.new_empty(seq_len, t.shape[-1]) for t in conv.states[:4])
            attention.conv.inputs = ac.inputs = mc.inputs = saved[key]
        state = {'attention': {'cache': attention, 'positions': ctx['positions']},
                 'attn_conv': {'cache': ac}, 'mlp_conv': {'cache': mc}}
        if self.device.type == 'cuda' and seq_len <= 8:
            if states[0].moe_workspace is None:
                states[0].moe_workspace = {}
            state['moe'] = {'workspace': states[0].moe_workspace}
        elif self.device.type == 'cuda':
            # Release the large prompt scratch at the end of this forward.
            state['moe'] = {'workspace': ctx.setdefault('moe_workspace', {})}
        if params.get('_inkling_verify_rows') and attention.quant is None:
            state['verify_rows'] = attention.verify_rows = state['moe']['verify_rows'] = True
        return state, (conv, slot, attention, ac, mc, kv, skip, write_length, write_table)

    def _store(self, state_info):
        if state_info is None:
            return
        conv, slot, attention, ac, mc, kv, skip, lengths, table = state_info
        for i, value in ((0, attention.conv.slots[0]), (1, attention.conv.slots[1]),
                         (2, ac.slots[2]), (3, mc.slots[3])):
            conv.states[i][slot:slot+1].copy_(value)
        k = attention.key[:, :, skip:].transpose(1, 2).contiguous()
        v = attention.value[:, :, skip:].transpose(1, 2).contiguous()
        kv.update_kv_direct(lengths, table, k, v, k.shape[1])

    def forward(self, x, params, out_dtype=None):
        ctx = params['_inkling']
        blocks = ctx['blocks']
        if blocks is None:
            blocks = x.new_empty(*x.shape[:-1], 0, x.shape[-1])
        loop = self.owner.loop
        instance = params.get('layer_instance', 0)
        if self.layer_idx == loop.core_start:
            if instance == 0:
                ctx['pre_hidden'], ctx['pre_blocks'] = x, blocks
            else:
                previous_blocks = blocks[..., ctx['pre_blocks'].shape[-2] + 1:, :]
                values = torch.cat((previous_blocks, x.unsqueeze(-2)), -2)
                ctx['message_values'] = decode_rows(rms, values, params.get('_inkling_verify_rows'))
                x, blocks = ctx['pre_hidden'], ctx['pre_blocks']
        taps = params.get('inkling_tap_collector')
        if taps is not None:
            taps.before_visit(ctx['visit'], x)
            ctx['visit'] += 1
        messages = None
        if loop.core_start <= self.layer_idx < loop.core_end and instance:
            values = ctx['message_values']
            query = loop.message_query[self.layer_idx - loop.core_start].float()
            if x.is_cuda and values.shape[-2] <= 16 and values.shape[-1] <= 4096:
                from ..modules.arch_specific.inkling2_fused import residual_query
                messages = residual_query(values, None, query)
            else:
                scores = torch.matmul(query, values.transpose(-1, -2)) / math.sqrt(values.shape[-1])
                messages = torch.matmul(scores.softmax(-1), values)
        layer = loop.layers[self.layer_idx]
        state, state_info = self._state(params, instance, x.shape[1])
        old = x
        x = layer(x, blocks, messages, state)
        self._store(state_info)
        if layer.boundary: blocks = torch.cat((blocks, old.unsqueeze(-2)), -2)
        ctx['blocks'] = blocks
        if taps is not None and ctx['visit'] == taps.visit_count and params.get('last_tokens_only'):
            ctx['last_hidden'] = x
        return x


class InklingOutput(InklingModule):
    def __init__(self, owner):
        super().__init__(owner, 'model.llm.output')
        self.caps['logits_output'] = True

    def project(self, x):
        loop = self.owner.loop
        if x.is_cuda:
            from ..modules.arch_specific.inkling2_head import capped_logits, int8_logits
            weight = loop.lm_head.weight
            real = self.config.unpadded_vocab_size
            if getattr(self.config, 'inkling_int8_head', False) and x.shape[1] <= 8:
                return int8_logits(x, loop.head_codes, loop.head_scales, len(weight), loop.softcap)
            # Keep the aligned body on the same dense GEMV as the padded head.
            aligned = real // 256 * 256
            linear = torch.nn.functional.linear
            if x.shape[1] == 1 and aligned and aligned < real:
                logits = torch.cat((linear(x, weight[:aligned]), linear(x, weight[aligned:real])), dim=-1)
            else:
                logits = linear(x, weight[:real])
            return capped_logits(logits, len(weight), loop.softcap)
        logits = loop.lm_head(x).float()
        return loop.softcap * torch.tanh(logits / loop.softcap)

    def forward(self, x, params, out_dtype=None):
        loop = self.owner.loop
        ctx = params.pop('_inkling')
        blocks = ctx['blocks']
        taps = params.get('inkling_tap_collector')
        count = params.get('last_tokens_only')
        if count and taps is None:
            blocks = blocks[:, -count:]
        x = loop.final_input(blocks, ctx.get('last_hidden', x)).squeeze(-2)
        x = decode_rows(loop.final_norm, x, params.get('_inkling_verify_rows')).to(loop.lm_head.weight.dtype)
        if taps is not None:
            taps.final(x)
            tap_cache = params.get('inkling_tap_cache')
            if tap_cache is not None:
                address = tap_cache.addresses(ctx['table'], ctx['positions'])
                tap_cache.rows[address] = taps.result().to(tap_cache.rows.dtype)
            if count:
                x = x[:, -count:]
        if params.get('_inkling_verify_rows') and x.shape[1] > 1:
            return decode_rows(self.project, x)
        return self.project(x)
