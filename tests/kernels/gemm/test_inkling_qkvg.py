import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


@torch.inference_mode()
def test_inkling_qkvg_matches_single_projections(device):
    if not hasattr(ext, 'inkling_qkvg_sq'):
        pytest.skip('CUDA per-slice int8 GEMV required')
    hidden = 2304
    widths = (3072, 1024, 1024, 3072)
    weights = [torch.randint(-32768, 32767, (hidden // 16, n // 16, 96),
                             dtype=torch.int16, device=device) for n in widths]
    suh = [torch.randn(hidden, device=device).sign().half() for _ in widths]
    svh = [torch.randn(n, device=device).half() for n in widths]
    counters = torch.zeros(4, max(widths) // 256, dtype=torch.int32, device=device)
    for _ in range(2):
        q, k, v = torch.randn(3, 1, 1, hidden, dtype=torch.float16, device=device)
        inputs = [q, k, v, q]
        outputs = ext.inkling_qkvg_sq(inputs, weights, suh, svh, counters)
        if not outputs:
            pytest.skip('6-bit per-slice int8 GEMV disabled on this device')
        for x, weight, su, sv, y in zip(inputs, weights, suh, svh, outputs, strict=True):
            ref = torch.empty_like(y)
            ext.exl3_gemm(x.view(1, hidden), weight, ref.view(1, -1), su, torch.empty_like(x).view(1, hidden),
                          sv, -1, False, True, 0)
            torch.testing.assert_close(y, ref, rtol=0, atol=0)
        assert counters.count_nonzero().item() == 0
