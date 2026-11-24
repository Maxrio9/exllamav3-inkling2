"""Decode logits over real token rows, with an optional row-scaled INT8 head."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _softcap(X, Y, REAL: tl.constexpr, PAD: tl.constexpr, CAP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * REAL + col, col < REAL, 0).to(tl.float32)
    y = CAP * libdevice.tanh(x * (1. / CAP))
    tl.store(Y + row * PAD + col, tl.where(col < REAL, y, -float('inf')), col < PAD)


def capped_logits(x, padded_rows, cap):
    """Keep the sampler's padded stride and mask tokens absent from the vocabulary."""
    real = x.shape[-1]
    y = x.new_empty((*x.shape[:-1], padded_rows), dtype=torch.float32)
    _softcap[(triton.cdiv(padded_rows, 256), x.numel() // real)](x, y, real, padded_rows, cap, 256)
    return y


@torch.no_grad()
def pack_int8(weight):
    rows, hidden = weight.shape
    codes = weight.new_empty((rows, hidden), dtype=torch.int8)
    scales = weight.new_empty(rows, dtype=torch.float32)
    for start in range(0, rows, 4096):
        block = weight[start:start + 4096].float()
        scale = block.abs().amax(1) / 127.
        # A zero row gets zero codes and a unit scale.
        scale = torch.where(scale > 0, scale, 1.)
        scales[start:start + len(block)] = scale
        codes[start:start + len(block)] = (block / scale[:, None]).round().clamp_(-127, 127).to(torch.int8)
    return codes, scales


@triton.jit
def _int8_head(X, W, S, Y, H: tl.constexpr, REAL: tl.constexpr, PAD: tl.constexpr,
               CAP: tl.constexpr, ROWS: tl.constexpr, BH: tl.constexpr):
    token = tl.program_id(1)
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    h = tl.arange(0, BH)
    x = tl.load(X + token * H + h, h < H, 0).to(tl.float32)
    w = tl.load(W + r[:, None] * H + h[None, :], (r[:, None] < REAL) & (h[None, :] < H), 0)
    scale = tl.load(S + r, r < REAL, 0)
    dot = (tl.sum(w.to(tl.float32) * x[None, :], 1) * scale).to(tl.float16).to(tl.float32)
    y = CAP * libdevice.tanh(dot * (1. / CAP))
    tl.store(Y + token * PAD + r, tl.where(r < REAL, y, -float('inf')), r < PAD)


def int8_logits(x, codes, scales, padded_rows, cap):
    real, hidden = codes.shape
    y = x.new_empty((*x.shape[:-1], padded_rows), dtype=torch.float32)
    _int8_head[(triton.cdiv(padded_rows, 8), x.numel() // hidden)](
        x.contiguous(), codes, scales, y, hidden, real, padded_rows, cap, 8, triton.next_power_of_2(hidden),
        num_warps=8)
    return y


@triton.jit
def _weight_read(W, S, N: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    i = block * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(W + i, i < N, 0).to(tl.float32)
    tl.store(S + block, tl.sum(values, 0))
