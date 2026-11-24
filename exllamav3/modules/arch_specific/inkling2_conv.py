"""Inkling causal convolutions. State belongs to one request and one visit."""

import math

import torch
import torch.nn.functional as F


class Inkling2ConvState:
    inputs = None

    def __init__(self):
        self.slots = {}

    def reset(self):
        self.slots.clear()


def short_conv(x, weight, state=None, slot=0, mask=None, dynamic_input=None,
               dynamic_projection=None, num_heads=None, *, verify_rows=False):
    """Run a static (weight [channels,1,k]) or dynamic ([k,channels]) convolution.

    Slots 0 and 1 are key/value, 2 and 3 post-attention/post-MLP. Pass
    a separate state for each request and physical-layer visit.
    """
    dynamic = dynamic_projection is not None
    kernel = weight.shape[0] if dynamic else weight.shape[-1]
    record = state.inputs[slot] if state is not None and state.inputs is not None else None
    if x.is_cuda and x.shape[0] == 1 and x.is_contiguous() and weight.is_contiguous():
        from .inkling2_fused import causal_conv
        if dynamic and (not num_heads or x.shape[-1] % num_heads):
            raise ValueError('invalid dynamic convolution head count')
        previous = state.slots.get(slot) if state is not None else None
        if previous is not None and previous.shape != (1, kernel - 1, x.shape[-1]):
            raise ValueError('convolution state shape mismatch')
        coefficients = None
        if dynamic:
            if verify_rows:
                from .inkling2_loop import decode_rows
                coefficients = decode_rows(dynamic_projection, dynamic_input)
            else:
                coefficients = dynamic_projection(dynamic_input)
        out, tail = causal_conv(x, weight, coefficients,
                                num_heads, previous, mask.contiguous() if mask is not None else None, state is not None, record)
        if state is not None:
            state.slots[slot] = tail
        return out
    residual = x.float()
    context = residual if mask is None else residual * mask[..., None]
    if record is not None:
        record.copy_(context[0])
    previous = None if state is None else state.slots.get(slot)
    if previous is None:
        previous = context.new_zeros(context.shape[0], kernel - 1, context.shape[-1])
    if previous.shape != (context.shape[0], kernel - 1, context.shape[-1]):
        raise ValueError('convolution state shape mismatch')
    context = torch.cat((previous, context), dim=1)
    if state is not None:
        tail = context[:, -(kernel - 1):] if kernel > 1 else context[:, :0]
        state.slots[slot] = tail if x.shape[1] == 1 else tail.clone()
    if dynamic:
        batch, length, channels = x.shape
        if num_heads is None or channels % num_heads:
            raise ValueError('invalid dynamic convolution head count')
        windows = context.unfold(1, kernel, 1).flip(-1)
        windows = windows.reshape(batch, length, num_heads, channels // num_heads, kernel)
        weights = weight.float().T.reshape(num_heads, channels // num_heads, kernel)
        weights = weights + dynamic_projection(dynamic_input).float().reshape(batch, length, num_heads, 1, kernel)
        out = (windows * weights).sum(-1).reshape(batch, length, channels)
        scale = math.sqrt(2 * kernel + 1)
    else:
        out = F.conv1d(context.transpose(1, 2).to(weight.dtype), weight, groups=x.shape[-1]).transpose(1, 2)
        scale = math.sqrt(kernel + 1)
    return ((out + residual) / scale).to(x.dtype)
