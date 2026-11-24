"""Inkling 2 routed experts. Handles wrap individual compressed EXL3 linears.

Each expert is a (gate_up, down) pair exposing forward(x, params) and
config.infer_params.no_reconstruct. No decoded weights are kept here.
"""
import os

import torch
import torch.nn.functional as F


def route(hidden_states, weight, bias, top_k=8, route_scale=8.0):
    """Return PR logits, unbiased normalized weights, and selected indices."""
    flat = hidden_states.float().reshape(-1, weight.shape[-1])
    logits = F.linear(flat, weight)
    scores = torch.sigmoid(logits)
    indices = torch.topk(scores + bias, top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * route_scale, indices


def softcapped_swiglu(gate, up, gate_cap=4.0, up_cap=25.0):
    dtype = gate.dtype
    gate, up = gate.float(), up.float()
    return (torch.tanh(gate / gate_cap) * gate_cap * torch.sigmoid(gate)
            * (torch.tanh(up / up_cap) * up_cap)).to(dtype)


class _ExpertProjection(torch.nn.Module):
    """Pointer tables for the indexed multi-matrix GEMM used by BlockSparseMLP."""
    def __init__(self, handles, device):
        super().__init__()
        from exllamav3.modules.moe_batch_recon import _addr_table
        self.width = handles[0].out_features
        for name in ('trellis', 'suh', 'svh'):
            self.register_buffer(name, _addr_table([getattr(h, name) for h in handles], device))
        formats = [(h.K, h.mcg, h.mul1) for h in handles]
        self.formats = tuple(dict.fromkeys(formats))
        # Each launch skips negative indices and writes its own format's slots.
        # The final entry stands for an unused tile in the padded prefill batch.
        lookups = [[i if fmt == active else -1 for i, fmt in enumerate(formats)] + [-1]
                   for active in self.formats]
        self.register_buffer('lookups', torch.tensor(lookups, device=device))

    def forward(self, x, experts, workspace=None):
        from exllamav3.ext import exllamav3_ext as ext
        slots, rows = experts.numel(), x.shape[1]
        if workspace is None:
            out = x.new_empty(slots, rows, self.width)
            scratch = x.new_empty(slots, rows, x.shape[-1])
            index_buffer = None
        else:
            # Visits are sequential on the request stream. The two projections
            # can reuse storage once the intervening activation has consumed it.
            key = (slots, x.shape[-1], self.width, x.dtype)
            if key not in workspace:
                workspace[key] = (x.new_empty(slots, 1, self.width), x.new_empty(slots, 1, x.shape[-1]),
                                  experts.new_empty(slots))
            out, scratch, index_buffer = workspace[key]
        for (bits, mcg, mul1), lookup in zip(self.formats, self.lookups.unbind()):
            indices = torch.index_select(lookup, 0, experts, out=index_buffer).view(1, slots)
            ext.exl3_mgemm(x, self.trellis, out, self.suh, scratch, self.svh, indices,
                           None, bits, -1, mcg, mul1, -1, -1, 0)
        return out


class Inkling2MoE(torch.nn.Module):
    def __init__(self, router_weight, router_bias, experts, top_k=8,
                 route_scale=8.0, gate_cap=4.0, up_cap=25.0):
        super().__init__()
        if len(experts) != router_weight.shape[0] or not 1 <= top_k <= len(experts):
            raise ValueError('expert count or top_k does not match router')
        self.register_buffer('router_weight', router_weight)
        self.register_buffer('router_bias', router_bias)
        self.experts = tuple(experts)
        self.top_k, self.route_scale = top_k, route_scale
        self.gate_cap, self.up_cap = gate_cap, up_cap
        for pair in self.experts:
            for linear in pair:
                if not linear.config.infer_params.no_reconstruct:
                    raise ValueError('compressed execution requires no_reconstruct')

        self.register_buffer('coop_tables', None, persistent=False)
        self.batched = router_weight.is_cuda
        if self.batched:
            self.gate_up, self.down = [_ExpertProjection(handles, router_weight.device)
                                       for handles in zip(*self.experts)]
            self.register_buffer('expert_bounds', torch.arange(len(experts) + 1, device=router_weight.device))

            if torch.version.hip is None and os.environ.get('INKLING_COOP_MOE') != '0':
                self.coop_tables = self._coop_tables()

        # GB10's specialized single- and multi-row kernels share their 16
        # logical K partitions and FP32 fold. The generic schedules differ.
        self.grouped_verify = (self.coop_tables is not None and self.down.width == 2304
                               and self.gate_up.width == 2304
                               and torch.cuda.get_device_capability(router_weight.device) == (12, 1))

    def verify(self, hidden_states, workspace):
        if not self.grouped_verify:
            from .inkling2_loop import decode_rows
            # Clone each result before the next row reuses its expert scratch.
            return decode_rows(lambda row: self.forward(row, workspace).clone(), hidden_states)

        rows = hidden_states.reshape(-1, hidden_states.shape[-1])
        routes = [route(row, self.router_weight, self.router_bias, self.top_k, self.route_scale)
                  for row in rows.split(1)]
        weights = torch.cat([weights for _, weights, _ in routes])
        indices = torch.cat([indices for _, _, indices in routes])
        result = self._batched(rows, weights, indices, workspace)
        return result.to(hidden_states.dtype).reshape_as(hidden_states)

    def _coop_tables(self):
        # Keep the original packed 2I-wide trellis. The cooperative kernel
        # selects its half by column and keeps the full stride between K tiles.
        if self.top_k != 8 or self.down.width % 128 or self.gate_up.width % 256:
            return None
        tables = []
        for projection, handles in zip((self.gate_up, self.down), zip(*self.experts)):
            if any(h.K not in (3, 3.5, 4, 8) or not h.mul1 or h.mcg or h.bias is not None for h in handles):
                return None
            rates = projection.trellis.new_tensor([int(2 * h.K) for h in handles])
            tables.append(torch.stack((projection.trellis, projection.suh, projection.svh, rates)))
        return torch.stack(tables)

    def _slot_tiles(self, ids):
        # Sort once, then pack up to sixteen token/expert slots for each MMA.
        # The upper bound avoids a device-to-host read of the actual tile count.
        ordered, permutation = ids.flatten().sort()
        bounds = torch.searchsorted(ordered, self.expert_bounds)
        counts = (bounds.diff() + 15) // 16
        ends = counts.cumsum(0)
        starts = ends - counts
        n = min(ids.numel(), len(self.experts) + (ids.numel() - 1) // 16)
        tile = torch.arange(n, device=ids.device)
        expert = torch.searchsorted(ends, tile, right=True).clamp_max(len(self.experts) - 1)
        rank = (tile - starts[expert])[:, None] * 16 + torch.arange(16, device=ids.device)
        index = bounds[expert, None] + rank
        slots = permutation[index.clamp_max(ids.numel() - 1)]
        return torch.where(index < bounds[expert + 1, None], slots, -1).int()

    def _cooperative(self, x, weights, ids, workspace):
        from exllamav3.ext import exllamav3_ext as ext
        rows, hidden = x.shape
        inter = self.gate_up.width // 2
        slots = ids.numel()
        key = ('coop', rows, hidden, inter)
        if key not in workspace:
            counters = slots * (inter // 128) + rows * (hidden // 128)
            # Short verification blocks build their expert run table on device.
            run_words = 3 + 2 * slots if 1 < rows <= 8 else 0
            workspace[key] = (
                x.new_empty(2, slots, inter, dtype=torch.float),
                x.new_empty(slots, inter),
                x.new_empty(slots, hidden, dtype=torch.float),
                x.new_zeros(counters + run_words, dtype=torch.int32),
                x.new_empty(rows, hidden, dtype=torch.float),
            )
        scratch = workspace[key]
        tiles = self._slot_tiles(ids) if rows > 8 else None
        ext.inkling_moe_coop(x, ids, weights, self.coop_tables, tiles, self.gate_cap, self.up_cap, *scratch)
        return scratch[-1]

    def _batched(self, x, weights, ids, workspace=None):
        tokens, picks = ids.shape
        assignments = tokens * picks
        x = x.half()
        if self.coop_tables is not None:
            return self._cooperative(x.contiguous(), weights, ids, workspace if workspace is not None else {})
        if tokens <= 8:
            # One row per selection. At decode all selected experts share one input.
            experts = ids.flatten()
            inputs = x if tokens == 1 else x.repeat_interleave(picks, dim=0)
            inputs = inputs.unsqueeze(1)
        else:
            # Pack assignments into 16-row tiles per expert. The host only needs
            # an upper bound on tile count. Routing and all row maps stay on GPU.
            rows = 16
            ordered, perm = ids.flatten().sort()
            bounds = torch.searchsorted(ordered, self.expert_bounds)
            num_tiles = (bounds.diff() + rows - 1) // rows
            tile_ends = num_tiles.cumsum(0)
            tile_starts = tile_ends - num_tiles
            count = min(assignments, len(self.experts) + (assignments - 1) // rows)
            tiles = torch.arange(count, device=x.device)
            experts = torch.searchsorted(tile_ends, tiles, right=True)
            active = experts.clamp_max(len(self.experts) - 1)
            tile_row = (tiles - tile_starts[active]) * rows + bounds[active]
            rows_in_tile = torch.arange(rows, device=x.device)
            src = (tile_row[:, None] + rows_in_tile).clamp(0, assignments - 1)
            inputs = x[perm[src] // picks]
        buffers = (workspace,) if workspace is not None else ()
        fused = self.gate_up(inputs, experts, *buffers)
        gate, up = fused.chunk(2, dim=-1)
        activated = softcapped_swiglu(gate, up, self.gate_cap, self.up_cap)
        projected = self.down(activated.contiguous(), experts, *buffers)
        if tokens > 8:
            rank = torch.arange(assignments, device=x.device) - bounds[ordered]
            source = tile_starts[ordered] * rows + rank
            # Restore router order so weights and duplicate expert picks across tokens align.
            projected = projected.flatten(0, 1)[source[perm.argsort()]]
        return (projected.reshape(tokens, picks, -1).float() * weights[..., None]).sum(dim=1)

    def forward(self, hidden_states, workspace=None):
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        _, weights, indices = route(flat, self.router_weight, self.router_bias,
                                    self.top_k, self.route_scale)
        if self.batched:
            chunk_size = 2048 if self.coop_tables is not None else 256
            if flat.shape[0] <= chunk_size:
                result = self._batched(flat, weights, indices, workspace if self.coop_tables is not None or flat.shape[0] <= 8 else None)
            else:
                chunks = zip(flat.split(chunk_size), weights.split(chunk_size), indices.split(chunk_size))
                result = torch.cat([self._batched(x, w, idx) for x, w, idx in chunks])
            return result.to(hidden_states.dtype).reshape(shape)
        result = torch.zeros_like(flat, dtype=torch.float32)
        # Match PR expert order, position order, output multiplication and fp32 accumulation.
        mask = F.one_hot(indices, num_classes=len(self.experts) + 1).permute(2, 1, 0)
        for expert_idx in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten():
            index = int(expert_idx)
            if index == len(self.experts):
                continue
            positions, tokens = torch.where(mask[index])
            gate_up, down = self.experts[index]
            fused = gate_up.forward(flat[tokens].contiguous(), {})
            gate, up = fused.chunk(2, dim=-1)
            activated = softcapped_swiglu(gate, up, self.gate_cap, self.up_cap)
            out = down.forward(activated.contiguous(), {})
            out = out * weights[tokens, positions, None]
            result.index_add_(0, tokens, out.to(result.dtype))
        return result.to(hidden_states.dtype).reshape(shape)
