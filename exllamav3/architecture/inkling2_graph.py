"""Request-owned CUDA graph for a fixed-width Inkling target forward."""
import warnings

import torch

from ..modules.arch_specific.inkling2_dspark import VisitTaps


class DecodeGraph:
    def __init__(self, ids, ctx):
        self.ids = torch.empty_like(ids)
        self.lengths = torch.empty_like(ctx['lengths'])
        self.table = torch.empty_like(ctx['table'])
        self.offsets = torch.arange(ids.shape[1], device=ids.device)[None] if ids.shape[1] > 1 else None
        self.windows = {}
        self.stream = torch.cuda.Stream(device=ids.device) if ids.is_cuda else None
        self.graph = self.output = self.taps = None
        self.replays = 0
        self.warmed = False
        self.failed = False

    def stage(self, ids, ctx):
        self.ids.copy_(ids)
        self.table.copy_(ctx['table'])
        self.lengths.copy_(ctx['lengths'])

    def body(self, forward, params):
        # Only the kernels' effects are replayed. Never let capture consume the
        # caller's Python tap collector or advance its host-side recurrent state.
        local = params.copy()
        positions = self.lengths[None] if self.offsets is None else self.lengths + self.offsets
        local['_inkling'] = dict(lengths=self.lengths, table=self.table, positions=positions,
                                 window_workspace=self.windows)
        collector = params.get('inkling_tap_collector')
        taps = None
        if collector is not None:
            taps = VisitTaps(collector.visit_count, collector.indices)
            local['inkling_tap_collector'] = taps
        return forward(self.ids, local), taps

    def capture(self, forward, params):
        with torch.cuda.stream(self.stream):
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin()
            try:
                output, taps = self.body(forward, params)
            finally:
                graph.capture_end()
        self.graph, self.output, self.taps = graph, output, taps

    def run(self, forward, ids, params):
        if self.failed:
            return forward(ids, params)
        self.stage(ids, params['_inkling'])
        collector = params.get('inkling_tap_collector')

        if not self.warmed:
            # Warm libraries and autotuners on the capture stream with one real
            # decode. It advances the slot exactly once, like any eager step.
            caller = torch.cuda.current_stream(ids.device)
            self.stream.wait_stream(caller)
            try:
                with torch.cuda.stream(self.stream):
                    output, taps = self.body(forward, params)
            finally:
                caller.wait_stream(self.stream)
            output.record_stream(caller)
            state = params['recurrent_states'][0]
            for buffers in (self.windows, state.moe_workspace or {}, state.verify_inputs or {}):
                for tensors in buffers.values():
                    for tensor in tensors:
                        tensor.record_stream(caller)
            if taps is not None:
                collector.values.update(taps.values)
                for value in taps.values.values():
                    value.record_stream(caller)
            self.warmed = True
            params.pop('_inkling')
            return output

        if self.graph is None:
            caller = torch.cuda.current_stream(ids.device)
            self.stream.wait_stream(caller)
            try:
                self.capture(forward, params)
            except RuntimeError as exc:
                # Capture records mutations without executing them. Retry this
                # token eagerly using the untouched cache and collector.
                self.failed = True
                warnings.warn(f'Inkling CUDA graph capture failed, using eager decode: {exc}', stacklevel=2)
            if self.failed:
                return forward(ids, params)

        # A replay failure can have partially written state. Never retry that
        # step. Public forward results must survive subsequent replays.
        self.graph.replay()
        self.replays += 1
        params['recurrent_states'][0].graph_replays += 1
        params.pop('_inkling')
        if self.taps is not None:
            collector.values.update({i: t.clone() for i, t in self.taps.values.items()})
        return self.output if params.get('_inkling_borrow_logits') else self.output.clone()


class DraftGraph:
    """Projected KV for committed taps and a fixed-shape draft block.

    Padding rows have finite, zero-initialized values and their absolute
    positions are masked by DraftAttention. The Markov head uses the last
    committed row of the staged block.
    """
    def __init__(self, draft, anchor, features, capacity):
        cfg = draft.config
        self.anchor = torch.empty_like(anchor)
        self.positions = anchor.new_empty(1, cfg.block_size, dtype=torch.long)
        self.context_positions = torch.arange(capacity, device=anchor.device)[None]
        width = min(cfg.block_size, features.shape[1])
        self.recent = torch.empty_like(features[:, -width:])
        self.recent_positions = anchor.new_empty(1, width, dtype=torch.long)
        self.last = self.recent[:, -1]
        shape = (1, capacity, cfg.num_key_value_heads, cfg.head_dim)
        self.kv = [(features.new_zeros(shape), features.new_zeros(shape)) for _ in draft.layers]
        self.length = 0
        self.stream = torch.cuda.Stream(device=anchor.device) if anchor.is_cuda else None
        self.graph = self.output = None
        self.warmed = self.failed = False
        self.replays = 0

    def stage(self, draft, anchor, features, positions):
        start = features.shape[1]
        self.anchor.copy_(anchor)
        self.positions.copy_(positions[:, start:start + draft.config.block_size])
        tail = start - self.recent.shape[1]
        self.recent.copy_(features[:, tail:])
        self.recent_positions.copy_(positions[:, tail:start])
        # First use (or a restored prefix) may span many pages. Ordinary
        # rounds update at most a block, entirely inside the captured body.
        for lo in range(self.length, tail, 256):
            hi = min(lo + 256, tail)
            for i, planes in enumerate(self.kv):
                for dst, src in zip(planes, draft.context_kv(i, features[:, lo:hi], positions[:, lo:hi])):
                    dst[:, lo:hi].copy_(src)
        self.length = start

    def update_recent(self, draft):
        offsets = self.recent_positions[0]
        for i, planes in enumerate(self.kv):
            for dst, src in zip(planes, draft.context_kv(i, self.recent, self.recent_positions)):
                dst.index_copy_(1, offsets, src)

    def run(self, body, capture):
        if not capture or self.failed:
            return body()
        if not self.warmed:
            caller = torch.cuda.current_stream(self.anchor.device)
            self.stream.wait_stream(caller)
            try:
                with torch.cuda.stream(self.stream):
                    outputs = body()
            finally:
                caller.wait_stream(self.stream)
            for t in outputs:
                t.record_stream(caller)
            self.warmed = True
            return outputs
        if self.graph is None:
            self.stream.wait_stream(torch.cuda.current_stream(self.anchor.device))
            try:
                with torch.cuda.stream(self.stream):
                    graph = torch.cuda.CUDAGraph()
                    graph.capture_begin()
                    try:
                        self.output = body()
                    finally:
                        graph.capture_end()
                self.graph = graph
            except RuntimeError as exc:
                self.failed = True
                warnings.warn(f'Inkling draft capture failed, using eager draft: {exc}', stacklevel=2)
                return body()
        self.graph.replay()
        self.replays += 1
        return self.output
