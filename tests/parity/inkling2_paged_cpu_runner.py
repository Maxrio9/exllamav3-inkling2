"""Exercise the production paged attention kernel on CPU with TRITON_INTERPRET=1."""
import os
from pathlib import Path
import sys
import types

import torch


def load_kernel():
    # Import the kernel without running the engine's CUDA extension initializer.
    root = Path(__file__).resolve().parents[2] / 'exllamav3'
    for name, path in [('exllamav3', root), ('exllamav3.modules', root / 'modules'),
                       ('exllamav3.modules.attention_fn', root / 'modules/attention_fn'),
                       ('exllamav3.util', root / 'util')]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    common = types.ModuleType('exllamav3.modules.attention_fn.common')
    common.AttnArgs = common.get_non_causal_span_arglist = None
    sys.modules[common.__name__] = common
    from exllamav3.modules.attention_fn import triton_paged
    return triton_paged


def check(kernel, past, n, window, q8):
    dim, heads = 32, 4
    k, v = [torch.randn(1, past + n, 2, dim).half() for _ in range(2)]
    q = torch.randn(1, n, heads, dim).half()
    relative = torch.randn(1, n, heads, 11).half()
    sink = torch.randn(1, n, heads).half()
    table = torch.tensor([[2, 0, 1, 3]], dtype=torch.int32)
    rows = torch.arange(past)
    physical = table[0, rows // 16], rows % 16
    kc, vc = [torch.zeros(4, 16, 2, dim).half() for _ in range(2)]
    quant = None
    if q8:
        h = kernel._get_h32(torch.device('cpu'))
        sk, sv = [torch.rand(4, 16, 2).half() + 1 for _ in range(2)]
        def pack(cache, scale, logical):
            codes = torch.randint(0, 256, cache.shape, dtype=torch.uint8)
            decoded = ((codes.float() - 127.5) * scale.unsqueeze(-1) / 128).half() @ h
            logical[0, :past] = decoded[physical]
            return codes.view(torch.int32).flatten(2)
        kc, vc = pack(kc, sk, k), pack(vc, sv, v)
        quant = sk, sv, 8, 8, h
    else:
        kc[physical], vc[physical] = k[0, :past], v[0, :past]
    result = kernel.inkling_paged_attention(
        q, k[:, past:].contiguous(), v[:, past:].contiguous(), kc, vc, table,
        torch.tensor([past], dtype=torch.int32), relative, sink, window, quant)
    distance = (torch.arange(n) + past)[:, None] - torch.arange(past + n)[None, :]
    scores = q.transpose(1, 2) @ k.repeat_interleave(2, 2).permute(0, 2, 3, 1) * dim**-.5
    bias = relative.transpose(1, 2).gather(-1, distance.clamp(0, 10).expand(1, heads, -1, -1))
    scores += bias.masked_fill((distance < 0) | (distance >= 11), 0)
    scores.masked_fill_((distance < 0) | ((window > 0) & (distance >= window)), -torch.inf)
    probs = torch.cat((scores, sink.transpose(1, 2).unsqueeze(-1)), -1).float().softmax(-1)[..., :-1].half()
    expected = (probs @ v.repeat_interleave(2, 2).transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(result, expected, atol=.003, rtol=.003)


def check_controllers():
    from exllamav3.modules.arch_specific import inkling2_fused as fused
    from exllamav3.modules.arch_specific.inkling2_loop import rms
    rows, h = 5, 70
    x = torch.randn(1, rows, h).half()
    for n in (0, 3, 8):
        blocks = torch.randn(1, rows, n, h).half()
        query = torch.randn(3, h)
        values = rms(torch.cat((blocks, x.unsqueeze(-2)), -2))
        expected = ((query @ values.transpose(-1, -2)) * h ** -.5).softmax(-1) @ values
        for base, partial in ((blocks, x), (values, None)):
            torch.testing.assert_close(fused.residual_query(base, partial, query), expected, rtol=2e-5, atol=2e-5)
    message = torch.randn(1, rows, 4, h)[:, :, 2]
    for normalized in (False, True):
        result = fused.gate_input(rms(x) if normalized else x, message, 1.e-6, torch.half, normalized)
        expected = torch.cat((rms(x), rms(message)), -1).half()
        torch.testing.assert_close(result, expected)
    forget, gain = torch.randn(1, rows, h * 2).chunk(2, -1)
    expected = (1 - forget.tanh()) * x + gain.tanh() * message
    torch.testing.assert_close(fused.gate_mix(x, message, forget, gain), expected, atol=1e-6, rtol=2e-5)


def check_convolutions():
    from exllamav3.modules.arch_specific import inkling2_fused as fused
    from exllamav3.modules.arch_specific.inkling2_conv import short_conv, Inkling2ConvState
    heads, h = 3, 150
    for dynamic in (False, True):
        for k in (1, 4):
            weight = torch.randn(k, h) if dynamic else torch.randn(h, 1, k).half()
            state = Inkling2ConvState()
            prev = None
            for n in (19, 8, 1, 2, 1, 37):
                x = torch.randn(1, n, h).half()
                projected = torch.randn(1, n, heads * k).half() if dynamic else None
                mask = None if n == 37 else torch.randint(0, 2, (1, n)).float()
                expected = short_conv(x, weight, state, 0, mask, x,
                                      (lambda _: projected) if dynamic else None, heads)
                fresh = torch.empty(n, h)
                out, prev = fused.causal_conv(x, weight, projected, heads, prev, mask, True, fresh)
                torch.testing.assert_close(out, expected, atol=.002, rtol=.002)
                torch.testing.assert_close(prev, state.slots[0], atol=0, rtol=0)
                torch.testing.assert_close(fresh, x[0].float() if mask is None else x[0] * mask[0, :, None], atol=0, rtol=0)


if __name__ == '__main__':
    if os.environ.get('TRITON_INTERPRET') != '1':
        raise SystemExit('Set TRITON_INTERPRET=1 to execute this test on CPU')
    torch.manual_seed(20)
    kernel = load_kernel()
    for q8 in (False, True):
        for past, n, window in [(0, 9, 8), (39, 5, 8), (32, 1, 0), (32, 43, 0), (39, 97, 51)]:
            check(kernel, past, n, window, q8)
    check_controllers()
    check_convolutions()
    print('10 paged attention, 9 controller and 24 streaming convolution cases passed on the CPU interpreter')
