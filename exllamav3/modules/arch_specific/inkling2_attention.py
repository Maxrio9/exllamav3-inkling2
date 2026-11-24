"""Inkling attention with a paged CUDA path and a dense reference path."""
import torch
import torch.nn.functional as F

from .inkling2_loop import decode_rows
from .inkling2_conv import Inkling2ConvState, short_conv


class Inkling2AttentionState:
    verify_rows = False

    def __init__(self):
        self.workspace = None
        self.conv = Inkling2ConvState()
        self.paged = self.quant = None
        self.key = self.value = self.positions = None

    def reset(self):
        self.conv.reset()
        self.paged = self.quant = None
        self.key = self.value = self.positions = None


def _clip(x, threshold):
    y = x.float()
    return (y * (threshold / (torch.linalg.vector_norm(y, dim=-1, keepdim=True) + 1e-6)).clamp(max=1)).to(x.dtype)


def _project(handle, x):
    if not handle.config.infer_params.no_reconstruct:
        raise ValueError('compressed execution requires no_reconstruct')
    return handle.forward(x.contiguous(), {})


class Inkling2Attention(torch.nn.Module):
    """Parameters are explicit so loader and visit scheduling remain independent.

    q, k, v, o, gate are LinearEXL3-style handles. rel_linear, sink and the
    dynamic convolution projections are ordinary callable unquantized linears.
    """
    def __init__(self, *, q, k, v, o, gate, rel_linear, sink, q_scale,
                 rel_scale, rel_proj, k_conv, v_conv, heads, kv_heads, head_dim,
                 qk_threshold, rel_threshold, global_attention=False,
                 window=1024, rel_extent=1024, eps=1e-6, log_floor=8192,
                 log_alpha=0.1, k_dynamic=None, v_dynamic=None):
        super().__init__()
        if heads % kv_heads:
            raise ValueError('query heads must be divisible by KV heads')
        self.q, self.k, self.v, self.o, self.gate = q, k, v, o, gate
        self.rel_linear, self.sink = rel_linear, sink
        for name, tensor in (('q_scale', q_scale), ('rel_scale', rel_scale), ('rel_proj', rel_proj),
                             ('k_conv', k_conv), ('v_conv', v_conv)):
            self.register_buffer(name, tensor)
        self.k_dynamic, self.v_dynamic = k_dynamic, v_dynamic
        if global_attention != (k_dynamic is not None and v_dynamic is not None):
            raise ValueError('global attention requires both dynamic convolutions')
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.qk_threshold, self.rel_threshold = qk_threshold, rel_threshold
        self.global_attention, self.window, self.rel_extent = global_attention, window, rel_extent
        self.eps, self.log_floor, self.log_alpha = eps, log_floor, log_alpha

        self.projection_pairs = []
        handles = (q, k, v, gate)
        self.bundle_sq = (q_scale.is_cuda and not torch.version.hip and
                          all(getattr(h, 'K', None) == 6 and h.mul1 and not h.mcg and
                              h.bias is None and h.default_out_dtype == torch.float16 and h.in_features % 128 == 0 and
                              h.out_features % 256 == 0 for h in handles))
        if q_scale.is_cuda and all(getattr(h, 'quant_type', None) == 'exl3' and
                                   h.default_out_dtype == torch.float16 for h in handles):
            # Uniform-width mgemm accepts distinct source rows, including K/V
            # after their separate message gates. Each output slab is contiguous.
            for pair in ((q, gate), (k, v)):
                a, b = pair
                if ((a.K, a.mcg, a.mul1, a.out_features) != (b.K, b.mcg, b.mul1, b.out_features)
                        or a.bias is not None or b.bias is not None):
                    self.projection_pairs.clear()
                    break
                tables = [torch.tensor([getattr(h, name).data_ptr() for h in pair],
                                       device=q_scale.device, dtype=torch.long)
                          for name in ('trellis', 'suh', 'svh')]
                self.projection_pairs.append((a, *tables))

    def _projections(self, x, key, value, state):
        n = x.shape[1]
        if (self.bundle_sq or self.projection_pairs) and not self.q.config.infer_params.no_reconstruct:
            raise ValueError('compressed execution requires no_reconstruct')
        if self.bundle_sq and x.shape[0] == 1 and state is not None and state.workspace is not None:
            from ...ext import exllamav3_ext as ext
            launch = ext.inkling_qkvg_sq if n == 1 else getattr(ext, 'inkling_verify_qkvg_sq', None)
            if launch is not None and (n == 1 or state.verify_rows):
                handles = (self.q, self.k, self.v, self.gate)
                words = n * 4 * max(h.out_features for h in handles) // 256
                workspace_key = ('projection counters', words)
                if workspace_key not in state.workspace:
                    state.workspace[workspace_key] = (x.new_zeros(words, dtype=torch.int32),)
                counters, = state.workspace[workspace_key]
                outputs = launch((x.contiguous(), key.contiguous(), value.contiguous(), x.contiguous()),
                                 [h.trellis for h in handles], [h.suh for h in handles],
                                 [h.svh for h in handles], counters)
                if outputs:
                    return outputs
        if state is not None and state.verify_rows and n > 1:
            # A prebuilt scalar extension or an unsupported quant uses the
            # same per-row dispatch as the reference without re-running layers.
            projections = [self._projections(a.contiguous(), b.contiguous(), c.contiguous(), state)
                           for a, b, c in zip(x.split(1, 1), key.split(1, 1), value.split(1, 1))]
            return tuple(torch.cat(parts, dim=1) if parts[0] is not None else None
                         for parts in zip(*projections))
        if x.shape[0] == 1 and 2 < n <= 8 and self.projection_pairs:
            from ...ext import exllamav3_ext as ext
            outputs = []
            for (h, weights, suh, svh), (a, b) in zip(self.projection_pairs, ((x, x), (key, value))):
                inputs = a.contiguous() if a is b else torch.cat((a, b), 0)
                result = x.new_empty((2, n, h.out_features))
                scratch = x.new_empty((2, n, h.in_features))
                ext.exl3_mgemm(inputs, weights, result, suh, scratch, svh, None, None,
                               h.K, -1, h.mcg, h.mul1, -1, -1, 0)
                outputs.extend(result.split(1))
            q, gate, k, v = outputs
            return q, k, v, gate
        return _project(self.q, x), _project(self.k, key), _project(self.v, value), None

    def _scaled(self, x, scale, threshold, positions):
        dtype = x.dtype
        x = _clip(x.float() * scale.float(), threshold).to(dtype)
        if self.global_attention:
            factor = 1 + self.log_alpha * torch.log(((positions + 1).float() / self.log_floor).clamp(min=1))
            x = (x.float() * factor[..., None, None]).to(dtype)
        return x

    def forward(self, x, positions, state=None, key_input=None, value_input=None, conv_mask=None, attention_mask=None):
        b, n, _ = x.shape
        if positions.shape != (b, n):
            raise ValueError('positions must have shape [batch, sequence]')
        key_input = x if key_input is None else key_input
        value_input = x if value_input is None else value_input
        cs = None if state is None else state.conv
        verify = state is not None and state.verify_rows
        q, k, v, gate = self._projections(x, key_input, value_input, state)
        q = q.reshape(b, n, self.heads, self.head_dim)
        k = short_conv(k, self.k_conv, cs, 0, conv_mask,
                       key_input, self.k_dynamic, self.kv_heads, verify_rows=verify)
        v = short_conv(v, self.v_conv, cs, 1, conv_mask,
                       value_input, self.v_dynamic, self.kv_heads, verify_rows=verify)
        k = k.reshape(b, n, self.kv_heads, self.head_dim)
        v = v.reshape_as(k)
        rel = (decode_rows(self.rel_linear, x) if verify else self.rel_linear(x)).reshape(b, n, self.heads, -1)
        if (x.is_cuda and b == 1 and n <= 8 and self.q_scale.shape == q.shape[-2:]
                and self.rel_scale.shape == rel.shape[-2:]
                and all(t.is_contiguous() for t in (q, k, v, rel, self.q_scale, self.rel_scale, positions))):
            from .inkling2_fused import attention_norms
            q, k, v, rel = attention_norms(q, k, v, rel, self.q_scale, self.rel_scale, positions,
                                           self.qk_threshold, self.rel_threshold, self.eps,
                                           self.log_floor if self.global_attention else 0, self.log_alpha)
        else:
            q = self._scaled(q, self.q_scale, self.qk_threshold, positions)
            k = _clip(k, self.qk_threshold)
            v = (v.float() * torch.rsqrt(v.float().square().mean(-1, keepdim=True) + self.eps)).to(v.dtype)
            rel = self._scaled(rel, self.rel_scale, self.rel_threshold, positions)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        relative = (decode_rows(lambda row: row @ self.rel_proj, rel) if verify else rel @ self.rel_proj)
        sinks = decode_rows(self.sink, x) if verify else self.sink(x)
        if state is not None and state.paged is not None:
            kc, vc, table, lengths, past, base = state.paged
            state.key, state.value = k, v  # only the fresh rows are committed after attention
            if x.is_cuda:
                from ..attention_fn.triton_paged import inkling_paged_attention
                if attention_mask is not None:
                    raise ValueError('paged Inkling attention takes a causal/window mask')
                out = inkling_paged_attention(
                    q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
                    v.transpose(1, 2).contiguous(), kc, vc, table, lengths,
                    relative.contiguous(), sinks.contiguous(), 0 if self.global_attention else self.window, state.quant, verify_rows=verify)
                return self._output(x, gate, out, verify_rows=verify)
            # CPU oracle for the cache layout. Read through the actual page table.
            if past:
                offsets = torch.arange(past, device=x.device)
                pages = table[0, offsets // kc.shape[1]]
                def append(cache, new):
                    rows = cache[pages, offsets % kc.shape[1]].transpose(0, 1).unsqueeze(0)
                    return torch.cat((rows, new), dim=2)
                k, v = append(kc, k), append(vc, v)
            key_pos = torch.arange(base, base + past + n, device=x.device)[None]
        elif state is not None:
            if state.key is not None:
                k = torch.cat((state.key, k), dim=2)
                v = torch.cat((state.value, v), dim=2)
                key_pos = torch.cat((state.positions, positions), dim=1)
            else:
                key_pos = positions
            if self.global_attention or k.shape[2] <= self.window:
                state.key, state.value, state.positions = k, v, key_pos
            else:
                state.key = k[:, :, -self.window:].clone()
                state.value = v[:, :, -self.window:].clone()
                state.positions = key_pos[:, -self.window:].clone()
        else:
            key_pos = positions
        distance = positions[:, :, None] - key_pos[:, None, :]
        repeats = self.heads // self.kv_heads
        logits = q @ k.repeat_interleave(repeats, dim=1).transpose(-1, -2) * (self.head_dim ** -0.5)
        rel_logits = relative.transpose(1, 2)
        index = distance.clamp(0, self.rel_extent - 1)[:, None].expand(-1, self.heads, -1, -1)
        bias = rel_logits.gather(-1, index).masked_fill(((distance < 0) | (distance >= self.rel_extent))[:, None], 0)
        logits = logits + bias
        valid = distance >= 0
        if not self.global_attention:
            valid = valid & (distance < self.window)
        logits = logits.masked_fill(~valid[:, None], torch.finfo(logits.dtype).min)
        if attention_mask is not None:
            logits = logits + attention_mask
        sinks = sinks.transpose(1, 2).unsqueeze(-1)
        probs = F.softmax(torch.cat((logits, sinks), dim=-1).float(), dim=-1)[..., :-1].to(q.dtype)
        out = (probs @ v.repeat_interleave(repeats, dim=1)).transpose(1, 2)
        return self._output(x, gate, out)

    def _output(self, x, gate, out, *, verify_rows=False):
        if gate is None:
            gate = decode_rows(lambda row: _project(self.gate, row), x) if verify_rows else _project(self.gate, x)
        out = out.reshape_as(gate)
        if out.is_cuda and out.shape[0] == 1 and out.shape[1] <= 8 and out.is_contiguous():
            from .inkling2_fused import apply_output_gate
            out = apply_output_gate(out, gate)
        else:
            out = (out.float() * torch.sigmoid(gate.float())).to(out.dtype)
        return decode_rows(lambda row: _project(self.o, row), out) if verify_rows else _project(self.o, out)
