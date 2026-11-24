"""Single-request Inkling residual, message and causal convolution kernels."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _query(B, X, Q, Y, BS: tl.constexpr, XS: tl.constexpr, H: tl.constexpr,
           N: tl.constexpr, QUERIES: tl.constexpr, PARTIAL: tl.constexpr,
           BN: tl.constexpr, BH: tl.constexpr):
    row, query = tl.program_id(0), tl.program_id(1)
    i, d = tl.arange(0, BN), tl.arange(0, BH)
    v = tl.load(B + row * BS + i[:, None] * H + d[None, :],
                (i[:, None] < N) & (d[None, :] < H), 0).to(tl.float32)
    if PARTIAL:
        x = tl.load(X + row * XS + d, d < H, 0).to(tl.float32)
        v = tl.where(i[:, None] == N, x[None, :], v)
        v *= tl.rsqrt(tl.sum(v * v, 1) / H + 1.e-6)[:, None]
    q = tl.load(Q + query * H + d, d < H, 0).to(tl.float32)
    logits = tl.sum(v * q[None, :], 1) * (H ** -0.5)
    logits = tl.where(i < N + PARTIAL, logits, -float('inf'))
    probs = tl.softmax(logits, 0)
    result = tl.sum(v * probs[:, None], 0)
    tl.store(Y + (row * QUERIES + query) * H + d, result, d < H)


def residual_query(blocks, partial, query):
    _, rows, n, h = blocks.shape
    queries = query.shape[0]
    count = n + int(partial is not None)
    out = blocks.new_empty(1, rows, queries, h, dtype=torch.float32)
    # A zero-sized tensor can have a null data pointer. All its loads are masked.
    base = blocks if n else partial
    x = partial if partial is not None else blocks
    _query[(rows, queries)](base, x, query, out, blocks.stride(1), x.stride(1), h, n, queries,
                            partial is not None, triton.next_power_of_2(count), triton.next_power_of_2(h))
    return out


@triton.jit
def _gate_input(X, M, Y, XS: tl.constexpr, MS: tl.constexpr, H: tl.constexpr,
                EPS: tl.constexpr, NORMALIZED: tl.constexpr, BH: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BH)
    x = tl.load(X + row * XS + d, d < H, 0).to(tl.float32)
    m = tl.load(M + row * MS + d, d < H, 0).to(tl.float32)
    if not NORMALIZED:
        x *= tl.rsqrt(tl.sum(x * x, 0) / H + EPS)
    m *= tl.rsqrt(tl.sum(m * m, 0) / H + EPS)
    tl.store(Y + row * 2 * H + d, x, d < H)
    tl.store(Y + row * 2 * H + H + d, m, d < H)


def gate_input(x, message, eps, dtype, normalized=False):
    _, rows, h = x.shape
    out = x.new_empty(1, rows, 2 * h, dtype=dtype)
    _gate_input[(rows,)](x, message, out, x.stride(1), message.stride(1), h, eps, normalized,
                         triton.next_power_of_2(h))
    return out


@triton.jit
def _gate_mix(X, M, F, G, Y, XS: tl.constexpr, MS: tl.constexpr, FS: tl.constexpr,
              GS: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * XS + d, d < H, 0).to(tl.float32)
    m = tl.load(M + row * MS + d, d < H, 0).to(tl.float32)
    f = tl.load(F + row * FS + d, d < H, 0)
    g = tl.load(G + row * GS + d, d < H, 0)
    forget = 2.0 * tl.sigmoid(2.0 * f) - 1.0
    gain = 2.0 * tl.sigmoid(2.0 * g) - 1.0
    tl.store(Y + row * H + d, (1.0 - forget) * x + gain * m, d < H)


def gate_mix(x, message, forget, gain):
    _, rows, h = x.shape
    out = x.new_empty(x.shape, dtype=torch.float32)
    _gate_mix[(rows, triton.cdiv(h, 256))](x, message, forget, gain, out, x.stride(1), message.stride(1),
                                           forget.stride(1), gain.stride(1), h, 256, enable_fp_fusion=False)
    return out


@triton.jit
def _conv(X, W, DYN, MASK, PREV, TAIL, Y, INPUTS, RECORD: tl.constexpr, N: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
          HEADS: tl.constexpr, PRIOR: tl.constexpr, SAVE: tl.constexpr, MASKED: tl.constexpr,
          SCALE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    residual = tl.load(X + row * H + d, d < H, 0).to(tl.float32)
    if RECORD:
        fresh = residual
        if MASKED:
            fresh *= tl.load(MASK + row)
        tl.store(INPUTS + row * H + d, fresh, d < H)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for j in tl.static_range(K):
        src = row + j - K + 1
        if src >= 0:
            value = tl.load(X + src * H + d, d < H, 0).to(tl.float32)
            if MASKED:
                value *= tl.load(MASK + src)
        elif PRIOR:
            value = tl.load(PREV + (row + j) * H + d, d < H, 0).to(tl.float32)
        else:
            value = tl.full((BLOCK,), 0, tl.float32)
        if HEADS:
            k = K - 1 - j
            w = tl.load(W + k * H + d, d < H, 0).to(tl.float32)
            w += tl.load(DYN + (row * HEADS + d // (H // HEADS)) * K + k, d < H, 0).to(tl.float32)
        else:
            w = tl.load(W + d * K + j, d < H, 0).to(tl.float32)
            value = value.to(W.dtype.element_ty).to(tl.float32)
        acc += value * w
    if not HEADS:
        acc = acc.to(W.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * H + d, (acc + residual) / SCALE, d < H)
    if SAVE:
        # Row zero owns the tail per channel tile. For N=1 PREV and TAIL
        # can alias: the shift proceeds left after the convolution reads.
        if row == 0:
            for j in tl.static_range(K - 1):
                src = N - (K - 1) + j
                if src >= 0:
                    tail = tl.load(X + src * H + d, d < H, 0).to(tl.float32)
                    if MASKED:
                        tail *= tl.load(MASK + src)
                elif PRIOR:
                    tail = tl.load(PREV + (N + j) * H + d, d < H, 0).to(tl.float32)
                else:
                    tail = tl.full((BLOCK,), 0, tl.float32)
                tl.store(TAIL + j * H + d, tail, d < H)


@triton.jit
def _chunk_conv(X, W, DYN, MASK, PREV, TAIL, Y, INPUTS, n, H: tl.constexpr, K: tl.constexpr,
                HEADS: tl.constexpr, PRIOR: tl.constexpr, SAVE: tl.constexpr, MASKED: tl.constexpr,
                RECORD: tl.constexpr, SCALE: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(1) * BR + tl.arange(0, BR)
    d = tl.program_id(0) * BD + tl.arange(0, BD)
    valid = (row[:, None] < n) & (d[None, :] < H)
    off = row[:, None] * H + d[None, :]
    residual = tl.load(X + off, valid, 0).to(tl.float32)
    if RECORD or (SAVE and K > 1):
        fresh = residual
        if MASKED:
            fresh *= tl.load(MASK + row, row < n, 0)[:, None]
        if RECORD:
            tl.store(INPUTS + off, fresh, valid)
        if SAVE and K > 1:
            # This path takes at least K-1 rows. Every saved row thus comes
            # from X, and TAIL is disjoint from the previous chunk's history.
            tail_row = row - (n - (K - 1))
            tl.store(TAIL + tail_row[:, None] * H + d[None, :], fresh,
                     valid & (tail_row[:, None] >= 0))
    acc = tl.full((BR, BD), 0, tl.float32)
    for j in tl.static_range(K):
        src = row - (K - 1 - j)
        value = tl.load(X + src[:, None] * H + d[None, :], valid & (src[:, None] >= 0), 0).to(tl.float32)
        if MASKED:
            value *= tl.load(MASK + src, (src >= 0) & (src < n), 0)[:, None]
        if PRIOR and j < K - 1:
            history = tl.load(PREV + (row + j)[:, None] * H + d[None, :],
                              (src[:, None] < 0) & (d[None, :] < H), 0)
            value = tl.where(src[:, None] < 0, history, value)
        if HEADS:
            k = K - 1 - j
            base = tl.load(W + k * H + d, d < H, 0).to(tl.float32)
            dw = tl.load(DYN + (row[:, None] * HEADS + d[None, :] // (H // HEADS)) * K + k,
                         valid, 0).to(tl.float32)
            acc += value * (base[None, :] + dw)
        else:
            w = tl.load(W + d * K + j, d < H, 0).to(tl.float32)
            acc += value.to(W.dtype.element_ty).to(tl.float32) * w[None, :]
    if not HEADS:
        acc = acc.to(W.dtype.element_ty).to(tl.float32)
    tl.store(Y + off, (acc + residual) / SCALE, valid)


def causal_conv(x, weight, dynamic, heads, previous, mask, save, record=None):
    _, n, h = x.shape
    kernel = weight.shape[0] if dynamic is not None else weight.shape[-1]
    tail = previous if n == 1 and previous is not None else None
    if save and tail is None:
        tail = x.new_empty(1, kernel - 1, h, dtype=torch.float32)
    out = torch.empty_like(x)
    scale = (2 * kernel + 1 if dynamic is not None else kernel + 1) ** .5
    if n >= max(16, kernel - 1):
        _chunk_conv[(triton.cdiv(h, 128), triton.cdiv(n, 16))](
            x, weight, x if dynamic is None else dynamic, x if mask is None else mask,
            x if previous is None else previous, x if tail is None else tail, out,
            x if record is None else record, n, h, kernel, heads if dynamic is not None else 0,
            previous is not None, save, mask is not None, record is not None, scale, 16, 128,
            enable_fp_fusion=False)
        return out, tail
    _conv[(n, triton.cdiv(h, 128))](
        x, weight, x if dynamic is None else dynamic, x if mask is None else mask,
        x if previous is None else previous, x if tail is None else tail, out,
        x if record is None else record, record is not None, n, h, kernel, heads if dynamic is not None else 0, previous is not None, save, mask is not None,
        scale, 128, enable_fp_fusion=False)
    return out, tail


@triton.jit
def _head_norm(X, S, P, Y, head, row, D: tl.constexpr, HEADS: tl.constexpr, BLOCK: tl.constexpr,
                    CLIP: tl.constexpr, SCALE: tl.constexpr, THRESHOLD: tl.constexpr,
                    EPS: tl.constexpr, LOG_FLOOR: tl.constexpr, LOG_ALPHA: tl.constexpr):
    d = tl.arange(0, BLOCK)
    off = (row * HEADS + head) * D + d
    v = tl.load(X + off, d < D, 0).to(tl.float32)
    if SCALE:
        v *= tl.load(S + head * D + d, d < D, 0).to(tl.float32)
    if D == 128:
        # ATen Reduce.cuh uses four contiguous values per CUDA lane. Match
        # the sequential sums within each lane before reducing the warp.
        lane = tl.arange(0, 32) * 4
        squares = v * v
        partial = tl.gather(squares, lane, 0) + tl.gather(squares, lane + 1, 0)
        partial += tl.gather(squares, lane + 2, 0)
        partial += tl.gather(squares, lane + 3, 0)
        ss = tl.sum(partial, 0)
    else:
        ss = tl.sum(v * v, 0)
    if CLIP:
        v *= tl.minimum(1., THRESHOLD * tl.div_rn(1., tl.sqrt_rn(ss) + 1.e-6))
    else:
        v *= tl.rsqrt(ss / D + EPS)
    if LOG_FLOOR:
        pos = tl.load(P + row)
        factor = 1. + LOG_ALPHA * libdevice.log(tl.maximum(1., (pos + 1).to(tl.float32) / LOG_FLOOR))
        v = v.to(X.dtype.element_ty).to(tl.float32) * factor
    tl.store(Y + off, v, d < D)


@triton.jit
def _qkvr_norm(Q, K, V, R, QS, RS, P, OQ, OK, OV, OR, H: tl.constexpr, KV: tl.constexpr,
               D: tl.constexpr, DR: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr,
               QC: tl.constexpr, RC: tl.constexpr, EPS: tl.constexpr, FLOOR: tl.constexpr, ALPHA: tl.constexpr):
    # One program per real head. The four groups have distinct scale/clip
    # semantics and all consume unmodified inputs from this visit.
    h, row = tl.program_id(0), tl.program_id(1)
    if h < H:
        _head_norm(Q, QS, P, OQ, h, row, D, H, BD, True, True, QC, EPS, FLOOR, ALPHA)
    elif h < H + KV:
        _head_norm(K, QS, P, OK, h - H, row, D, KV, BD, True, False, QC, EPS, 0, ALPHA)
    elif h < H + 2 * KV:
        _head_norm(V, QS, P, OV, h - H - KV, row, D, KV, BD, False, False, 0., EPS, 0, ALPHA)
    else:
        _head_norm(R, RS, P, OR, h - H - 2 * KV, row, DR, H, BR, True, True, RC, EPS, FLOOR, ALPHA)


def attention_norms(q, k, v, rel, q_scale, rel_scale, positions, q_clip, rel_clip, eps, floor, alpha):
    _, n, heads, dim = q.shape
    kv, dr = k.shape[-2], rel.shape[-1]
    outputs = tuple(torch.empty_like(t) for t in (q, k, v, rel))
    _qkvr_norm[(2 * (heads + kv), n)](
        q, k, v, rel, q_scale, rel_scale, positions, *outputs, heads, kv, dim, dr,
        triton.next_power_of_2(dim), triton.next_power_of_2(dr), q_clip, rel_clip, eps, floor, alpha,
        num_warps=1, enable_fp_fusion=False)
    return outputs


@triton.jit
def _apply_output_gate(X, G, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    g = tl.load(G + i, i < N, 0).to(tl.float32)
    tl.store(Y + i, x * tl.sigmoid(g), i < N)


def apply_output_gate(x, gate):
    y = torch.empty_like(x)
    _apply_output_gate[(triton.cdiv(x.numel(), 256),)](x, gate, y, x.numel(), 256)
    return y


@triton.jit
def _project_gate_mix(X, M, A, W, Y, H: tl.constexpr, R: tl.constexpr,
                      XS: tl.constexpr, MS: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr):
    row = tl.program_id(1)
    d = tl.program_id(0) * BD + tl.arange(0, BD)
    r = tl.arange(0, BR)
    a = tl.load(A + row * R + r, r < R, 0).to(tl.float32)
    off = d[:, None] * R + r[None, :]
    mask = (d[:, None] < H) & (r[None, :] < R)
    f = tl.sum(tl.load(W + off, mask, 0) * a[None, :], 1)
    g = tl.sum(tl.load(W + H * R + off, mask, 0) * a[None, :], 1)
    x = tl.load(X + row * XS + d, d < H, 0).to(tl.float32)
    m = tl.load(M + row * MS + d, d < H, 0).to(tl.float32)
    f = 2. * tl.sigmoid(2. * f) - 1.
    g = 2. * tl.sigmoid(2. * g) - 1.
    tl.store(Y + row * H + d, (1. - f) * x + g * m, d < H)


def project_gate_mix(x, message, gate, weights):
    _, n, h = x.shape
    rank = gate.shape[-1]
    y = x.new_empty(x.shape, dtype=torch.float32)
    _project_gate_mix[(triton.cdiv(h, 16), n)](x, message, gate, weights, y, h, rank, x.stride(1),
                                               message.stride(1), 16, triton.next_power_of_2(rank),
                                               enable_fp_fusion=False)
    return y


@triton.jit
def _add_residual(A, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + i, i < N, 0).to(tl.float32)
    b = tl.load(B + i, i < N, 0).to(tl.float32)
    tl.store(Y + i, a + b, i < N)


def add_residual(a, b, dtype):
    out = a.new_empty(a.shape, dtype=dtype)
    _add_residual[(triton.cdiv(a.numel(), 256),)](a, b, out, a.numel(), 256)
    return out
