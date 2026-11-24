"""Compare short Inkling expert runs with the generic tiled kernel at release dimensions."""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_scale, rand_trellis
from testlib.moe import ptr_table


@pytest.mark.cuda_only
@pytest.mark.parametrize("rows", [1, 3, 8])
def test_mixed_rate_decode(device, rows):
    gen = torch.Generator().manual_seed(958)
    hidden, inter, experts = 2304, 1152, 12
    rates = (3, 3.5, 4, 8)
    fields, backing = [], []
    for proj, (inner, out) in enumerate(((hidden, 2 * inter), (inter, hidden))):
        bits = [rates[(e + proj) % len(rates)] for e in range(experts)]
        packed = [rand_trellis(inner, out, rate, gen, device) for rate in bits]
        suh = [rand_scale(inner, gen, device) * (inner ** -0.5) for _ in bits]
        svh = [rand_scale(out, gen, device) for _ in bits]
        backing.extend((packed, suh, svh))
        fields.append(torch.stack([ptr_table(group, device) for group in (packed, suh, svh)] +
                                  [torch.tensor([int(2 * rate) for rate in bits], device=device)]))
    tables = torch.stack(fields)
    x = torch.randn(rows, hidden, generator=gen).to(device, torch.float16)
    ids = torch.stack([torch.randperm(experts, generator=gen)[:8] for _ in range(rows)]).to(device)
    weights = torch.rand(rows, 8, generator=gen).to(device)
    slots = ids.numel()
    gu = x.new_empty(2, slots, inter, dtype=torch.float32)
    activation = x.new_empty(slots, inter)
    down = x.new_empty(slots, hidden, dtype=torch.float32)
    counters = ids.new_zeros(slots * (inter // 128) + rows * (hidden // 128) + 3 + 2 * slots, dtype=torch.int32)
    output = x.new_empty(x.shape, dtype=torch.float32)
    # An explicit one-slot tile per selection forces the generic path, while
    # the implicit run table groups overlapping selections in the new kernel.
    tiles = ids.new_full((slots, 8), -1, dtype=torch.int32)
    tiles[:, 0] = torch.arange(slots, device=device)
    for selected in (ids, ids.flip(0)):
        args = (x, selected.contiguous(), weights, tables)
        ext.inkling_moe_coop(*args, tiles, 4., 25., gu, activation, down, counters, output)
        buffers = (gu, activation, down, output)
        reference = [tensor.clone() for tensor in buffers]
        ext.inkling_moe_coop(*args, None, 4., 25., gu, activation, down, counters, output)
        for tensor, expected in zip(buffers, reference):
            torch.testing.assert_close(tensor, expected, atol=0, rtol=0)
