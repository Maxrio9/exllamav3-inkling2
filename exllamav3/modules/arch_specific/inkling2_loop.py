"""Inkling loop plumbing, with injectable attention, MoE and convolutions.

Callables receive their visit-specific state dictionary; they own its contents.
The implementation intentionally keeps physical layer objects shared across visits.
"""
import math
from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F


def rms(x, eps=1e-6):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def decode_rows(fn, x, enabled=True):
    """Evaluate a shape-sensitive projection with its single-token reduction tree."""
    if not enabled:
        return fn(x)
    return torch.cat([fn(row.contiguous()) for row in x.split(1, dim=1)], dim=1)


class ResidualQuery(nn.Module):
    def __init__(self, hidden_size, count=1):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(count, hidden_size, dtype=torch.float32))

    def forward(self, blocks, partial):
        if partial.is_cuda and partial.shape[0] == 1 and blocks.shape[-2] < 16 and partial.shape[-1] <= 4096:
            from .inkling2_fused import residual_query
            return residual_query(blocks, partial, self.query)
        values = rms(torch.cat((blocks, partial.unsqueeze(-2)), dim=-2))
        scores = torch.matmul(self.query.float(), values.transpose(-1, -2)) / math.sqrt(values.shape[-1])
        return torch.matmul(scores.softmax(dim=-1), values)


class MessageGate(nn.Module):
    def __init__(self, hidden_size, rank=64, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.down_proj = nn.Linear(2 * hidden_size, rank, bias=False)
        self.forget_gate_proj = nn.Linear(rank, hidden_size, bias=False, dtype=torch.float32)
        self.message_gate_proj = nn.Linear(rank, hidden_size, bias=False, dtype=torch.float32)

    def pack_outputs(self):
        # Both FP32 projections consume the same rank-r row. Keep the original
        # parameter names and back them by views of one inference weight.
        weights = torch.cat((self.forget_gate_proj.weight, self.message_gate_proj.weight)).detach()
        self.register_buffer('output_weight', weights, persistent=False)
        for module, weight in zip((self.forget_gate_proj, self.message_gate_proj), weights.chunk(2)):
            module.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, hidden, message, normalized=None, *, verify_rows=False):
        fused = hidden.is_cuda and hidden.shape[0] == 1
        if fused:
            from .inkling2_fused import gate_input, gate_mix
            x = hidden if normalized is None else normalized
            inputs = gate_input(x, message, self.eps, self.down_proj.weight.dtype, normalized is not None)
        else:
            hidden, message = hidden.float(), message.float()
            x = rms(hidden, self.eps) if normalized is None else normalized
            inputs = torch.cat((x, rms(message, self.eps)), dim=-1).to(self.down_proj.weight.dtype)
        gate = decode_rows(self.down_proj, inputs) if verify_rows else self.down_proj(inputs)
        if fused and hidden.shape[1] <= 8 and gate.shape[-1] <= 128 and hasattr(self, 'output_weight'):
            from .inkling2_fused import project_gate_mix
            return project_gate_mix(hidden, message, gate, self.output_weight)
        gate = gate.float()
        if hasattr(self, 'output_weight'):
            forget, gain = F.linear(gate, self.output_weight).chunk(2, dim=-1)
        else:
            forget, gain = self.forget_gate_proj(gate), self.message_gate_proj(gate)
        if fused:
            return gate_mix(hidden, message, forget, gain)
        return (1 - forget.tanh()) * hidden + gain.tanh() * message


class AttentionAdapter(nn.Module):
    """Bridge the attention slice to the loop's request-owned visit state."""
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, q, k, v, state):
        from .inkling2_attention import Inkling2AttentionState
        if q.shape[0] != 1:
            raise ValueError('Inkling attention currently supports batch size 1 only')
        cache = state.get('cache')
        if cache is None:
            cache = state['cache'] = Inkling2AttentionState()
        positions = state.get('positions')
        if positions is None:
            start = state.get('past', 0)
            positions = torch.arange(start, start + q.shape[1], device=q.device)[None]
            state['past'] = start + q.shape[1]
        return self.attention.forward(q, positions, cache, k, v)


class MoEAdapter(nn.Module):
    def __init__(self, moe):
        super().__init__()
        self.moe = moe

    def forward(self, x, state):
        if 'workspace' in state:
            if state.get('verify_rows'):
                return self.moe.verify(x, state['workspace'])
            return self.moe.forward(x, workspace=state['workspace'])
        return self.moe.forward(x)


class ConvAdapter(nn.Module):
    def __init__(self, weight, slot):
        super().__init__()
        self.weight = weight
        self.slot = slot

    def forward(self, x, state):
        from .inkling2_conv import Inkling2ConvState, short_conv
        cache = state.get('cache')
        if cache is None:
            cache = state['cache'] = Inkling2ConvState()
        return short_conv(x, self.weight, cache, self.slot)


class LoopLayer(nn.Module):
    """One physical layer. Full attention uses three queries and norms."""
    def __init__(self, index, hidden_size, attention, moe, attn_conv, mlp_conv,
                 full=False, in_loop=False, block_size=6, rank=64, eps=1e-6):
        super().__init__()
        self.index, self.full, self.in_loop = index, full, in_loop
        self.boundary = index % block_size == 0
        self.attention_input = ResidualQuery(hidden_size, 3 if full else 1) if index else None
        self.mlp_input = ResidualQuery(hidden_size)
        self.attention, self.moe = attention, moe
        self.attn_conv, self.mlp_conv = attn_conv, mlp_conv
        self.attn_norms = nn.ModuleList([nn.RMSNorm(hidden_size, eps=eps) for _ in range(3 if full else 1)])
        self.mlp_norm = nn.RMSNorm(hidden_size, eps=eps)
        self.gates = nn.ModuleList([MessageGate(hidden_size, rank, eps) for _ in range(4)]) if in_loop else None

    def forward(self, hidden, blocks, messages, state):
        dtype = hidden.dtype
        block_residual = hidden
        verify_rows = state.get('verify_rows', False)
        if self.attention_input is None:
            inputs = [hidden] * 3
        else:
            result = self.attention_input(blocks, hidden).unbind(-2)
            inputs = list(result) if self.full else [result[0]] * 3
        if messages is not None:
            # Each CUDA gate already has a normalization kernel for its message.
            # Normalize x in that same launch on the short-row path.
            shared = None
            if not self.full and not (hidden.is_cuda and hidden.shape[0] == 1 and hidden.shape[1] <= 8):
                shared = rms(inputs[0], self.gates[0].eps)
            inputs = [self.gates[i](x, messages[..., i, :], shared, verify_rows=verify_rows) for i, x in enumerate(inputs)]
        if not self.full and messages is None:
            # Preserve the target RMS reduction before casting. A different
            # reduction tree can flip a close greedy choice during verification.
            inputs = [decode_rows(self.attn_norms[0], inputs[0], verify_rows).to(dtype)] * 3
        else:
            inputs = [decode_rows(self.attn_norms[i if self.full else 0], x, verify_rows).to(dtype) for i, x in enumerate(inputs)]
        x = self.attention(*inputs, state=state.setdefault('attention', {}))
        x = self.attn_conv(x, state=state.setdefault('attn_conv', {}))
        fused_add = (hidden.is_cuda and hidden.shape[0] == 1 and hidden.shape[1] <= 8 and hidden.is_contiguous()
                     and x.is_contiguous())
        if fused_add:
            from .inkling2_fused import add_residual
        if self.boundary:
            local_blocks = torch.cat((blocks, block_residual.unsqueeze(-2)), dim=-2)
            residual = x = x.float()
        else:
            local_blocks = blocks
            residual = x = (add_residual(block_residual, x, torch.float32) if fused_add
                            else block_residual.float() + x.float())
        x = self.mlp_input(local_blocks, x).squeeze(-2)
        if messages is not None:
            x = self.gates[3](x, messages[..., 3, :], verify_rows=verify_rows)
        x = self.moe(decode_rows(self.mlp_norm, x, verify_rows).to(dtype), state=state.setdefault('moe', {}))
        x = self.mlp_conv(x, state=state.setdefault('mlp_conv', {}))
        if fused_add and x.is_contiguous():
            return add_residual(residual, x, dtype)
        return (residual + x).to(dtype)


@dataclass
class RequestState:
    visits: dict = field(default_factory=dict)

    def reset(self):
        self.visits.clear()


class InklingLoop(nn.Module):
    def __init__(self, layers, schedule, core_start, core_end, passes, hidden_size, lm_head, eps=1e-6, softcap=30):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.schedule = tuple(schedule)
        self.core_start, self.core_end, self.passes = core_start, core_end, passes
        self.message_query = nn.Parameter(torch.zeros(core_end - core_start, 4, hidden_size, dtype=torch.float32))
        self.final_input = ResidualQuery(hidden_size)
        self.final_norm = nn.RMSNorm(hidden_size, eps=eps)
        self.lm_head, self.softcap = lm_head, softcap

    def forward(self, embeddings, state=None, trace=None, tap_collector=None):
        if state is None:
            state = RequestState()
        hidden = embeddings
        blocks = embeddings.new_zeros(*embeddings.shape[:-1], 0, embeddings.shape[-1])
        pre_hidden = pre_blocks = previous_hidden = previous_blocks = None
        for visit in self.schedule:
            layer = self.layers[visit.layer]
            if visit.layer == self.core_start:
                if visit.pass_index == 0:
                    pre_hidden, pre_blocks = hidden, blocks
                else:
                    previous_hidden = hidden
                    previous_blocks = blocks[..., pre_blocks.shape[-2] + 1:, :]
                    hidden, blocks = pre_hidden, pre_blocks
            if tap_collector is not None:
                tap_collector.before_visit(visit.index, hidden)
            messages = None
            if self.core_start <= visit.layer < self.core_end and visit.pass_index > 0:
                values = rms(torch.cat((previous_blocks, previous_hidden.unsqueeze(-2)), dim=-2))
                query = self.message_query[visit.layer - self.core_start].float()
                scores = torch.matmul(query, values.transpose(-1, -2)) / math.sqrt(values.shape[-1])
                messages = torch.matmul(scores.softmax(-1), values)
            old_hidden = hidden
            hidden = layer(hidden, blocks, messages, state.visits.setdefault(visit.cache_slot, {}))
            if layer.boundary:
                blocks = torch.cat((blocks, old_hidden.unsqueeze(-2)), dim=-2)
            if trace is not None:
                trace(visit, hidden, blocks)
        hidden = self.final_input(blocks, hidden).squeeze(-2)
        hidden = self.final_norm(hidden).to(embeddings.dtype)
        if tap_collector is not None:
            tap_collector.final(hidden)
        logits = self.lm_head(hidden).float()
        return self.softcap * torch.tanh(logits / self.softcap)
