from types import SimpleNamespace

import pytest
import torch

from exllamav3.architecture.inkling2_exl import InklingOutput


@pytest.mark.parametrize('width', [1, 8])
@torch.inference_mode()
def test_inkling_head_masks_padding_and_uses_dense_weights_by_default(device, width):
    from exllamav3.modules.arch_specific.inkling2_head import pack_int8

    real, padded, hidden = 269, 288, 128
    linear = torch.nn.Linear(hidden, padded, bias=False, device=device, dtype=torch.float16)
    linear.weight[0].zero_()
    linear.weight[real:].fill_(100)
    x = torch.randn(1, width, hidden, device=device, dtype=torch.float16)
    config = SimpleNamespace(unpadded_vocab_size=real)
    loop = SimpleNamespace(lm_head=linear, softcap=30.)
    head = SimpleNamespace(config=config, owner=SimpleNamespace(loop=loop))

    def check(logits, expected):
        assert logits.shape == (1, width, padded)
        assert logits.dtype == torch.float32
        assert logits[..., real:].isneginf().all()
        assert logits[..., 0].count_nonzero() == 0
        expected = loop.softcap * (expected.float() / loop.softcap).tanh()
        torch.testing.assert_close(logits[..., :real], expected, atol=1e-3, rtol=1e-3)

    check(InklingOutput.project(head, x), torch.nn.functional.linear(x, linear.weight[:real]))
    config.inkling_int8_head = True
    loop.head_codes, loop.head_scales = pack_int8(linear.weight[:real])
    reconstructed = loop.head_codes.float() * loop.head_scales[:, None]
    check(InklingOutput.project(head, x), torch.nn.functional.linear(x.float(), reconstructed).half())
