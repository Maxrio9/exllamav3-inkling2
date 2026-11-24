import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT.parent / 'ref/modeling_inkling2_loop.py'
spec = importlib.util.spec_from_file_location('moe', ROOT / 'exllamav3/modules/arch_specific/inkling2_moe.py')
moe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moe)


def oracle():
    tree = ast.parse(REF.read_text())
    names = {'Inkling2LoopSoftcap', 'Inkling2LoopSoftcappedSwiGLU',
             'Inkling2LoopExperts', 'Inkling2LoopTopKRouter', 'Inkling2LoopSparseMoeBlock'}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    ns = {'torch': torch, 'nn': nn, 'F': F, 'ACT2FN': {'sigmoid': torch.sigmoid},
          'Inkling2LoopTextConfig': object, 'Inkling2LoopConfig': object}
    exec(compile(module, str(REF), 'exec'), ns)
    return ns


class MockLinear:
    def __init__(self, weight, rate):
        self.weight, self.rate = weight, rate
        self.config = SimpleNamespace(infer_params=SimpleNamespace(no_reconstruct=True))
        self.calls = 0

    def forward(self, x, params):
        assert params == {} and x.is_contiguous()
        self.calls += 1
        return F.linear(x, self.weight)


def setup(k):
    torch.manual_seed(51050)
    config = SimpleNamespace(n_routed_experts=5, num_local_experts=5, hidden_size=6,
        moe_intermediate_size=4, num_experts_per_tok=k, scoring_func='sigmoid',
        routed_scaling_factor=8.0, mlp_gate_softcap=4.0, mlp_up_softcap=25.0)
    ref = oracle()['Inkling2LoopSparseMoeBlock'](config)
    with torch.no_grad():
        ref.gate.weight.copy_(torch.randn_like(ref.gate.weight))
        ref.gate.e_score_correction_bias.copy_(torch.tensor([0.2, -0.3, 0., 0.4, -0.1]))
        ref.experts.gate_up_proj.copy_(torch.randn_like(ref.experts.gate_up_proj))
        ref.experts.down_proj.copy_(torch.randn_like(ref.experts.down_proj))
    pairs = [(MockLinear(ref.experts.gate_up_proj[i], 3.0 + (i % 2) * 0.5),
              MockLinear(ref.experts.down_proj[i], 3.0 + (i % 2) * 0.5)) for i in range(5)]
    actual = moe.Inkling2MoE(ref.gate.weight, ref.gate.e_score_correction_bias, pairs, k)
    return ref, actual, pairs


@pytest.mark.parametrize('k', range(1, 6))
@pytest.mark.parametrize('ties', [False, True])
def test_router_and_fused_oracle(k, ties):
    ref, actual, pairs = setup(k)
    x = torch.randn(2, 7, 6)
    if ties:
        with torch.no_grad():
            ref.gate.weight.zero_()
            ref.gate.e_score_correction_bias.zero_()
    logits, weights, ids = ref.gate(x)
    got_logits, got_weights, got_ids = moe.route(x, ref.gate.weight, ref.gate.e_score_correction_bias, k)
    torch.testing.assert_close(got_logits, logits, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got_weights, weights, rtol=1e-4, atol=1e-5)
    assert torch.equal(got_ids, ids)
    out, expected = actual.forward(x), ref(x)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)
    diff = (out - expected).abs()
    print(f'k={k} ties={ties} max={diff.max().item():.8g} mean={diff.mean().item():.8g} cosine={F.cosine_similarity(out.flatten(), expected.flatten(), dim=0).item():.8g} top1={(out.argmax(-1)==expected.argmax(-1)).float().mean().item():.8g}')
    for i in ids.unique().tolist():
        assert pairs[i][0].calls and pairs[i][1].calls


def test_softcaps_and_dtypes():
    ref = oracle()['Inkling2LoopSoftcappedSwiGLU'](SimpleNamespace(mlp_gate_softcap=4., mlp_up_softcap=25.))
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        gate = torch.tensor([-100., -4.001, -4., 0., 4., 4.001, 100.], dtype=dtype)
        up = torch.tensor([-100., -25.001, -25., 0., 25., 25.001, 100.], dtype=dtype)
        torch.testing.assert_close(moe.softcapped_swiglu(gate, up), ref(gate, up), rtol=1e-4, atol=1e-5)


def test_reconstruction_rejected():
    ref, _, pairs = setup(1)
    pairs[0][0].config.infer_params.no_reconstruct = False
    with pytest.raises(ValueError, match='no_reconstruct'):
        moe.Inkling2MoE(ref.gate.weight, ref.gate.e_score_correction_bias, pairs, 1)


def test_bias_changes_selection_not_weight_scores():
    x = torch.tensor([[1.]])
    weight = torch.tensor([[0.], [1.], [2.]])
    bias = torch.tensor([2., 0., 0.])
    _, weights, indices = moe.route(x, weight, bias, 2)
    assert set(indices[0].tolist()) == {0, 2}
    scores = torch.sigmoid(weight.flatten())[indices[0]]
    torch.testing.assert_close(weights[0], scores / scores.sum() * 8.)


def test_mini_router_top_eight_of_128():
    torch.manual_seed(27)
    ns = oracle()
    config = SimpleNamespace(num_experts_per_tok=8, num_local_experts=128, hidden_size=9,
                             scoring_func='sigmoid', routed_scaling_factor=8.)
    ref = ns['Inkling2LoopTopKRouter'](config)
    with torch.no_grad():
        ref.weight.copy_(torch.randn_like(ref.weight))
        ref.e_score_correction_bias.copy_(torch.randn_like(ref.e_score_correction_bias))
    x = torch.randn(11, 9)
    expected = ref(x)
    actual = moe.route(x, ref.weight, ref.e_score_correction_bias)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('rows', [1, 8, 9, 32, 256])
def test_batched_expert_row_maps(rows):
    """Exercise the same tile and assignment maps as CUDA with indexed dense matmuls."""
    _, model, pairs = setup(3)
    x = torch.randn(rows, 6).half()
    _, weights, ids = moe.route(x, model.router_weight, model.router_bias, model.top_k)
    # Leave some experts empty and concentrate enough rows on one expert to span tiles.
    ids[:, 0], ids[:, 1], ids[:, 2] = 0, 3, 4
    model.expert_bounds = torch.arange(len(pairs) + 1)

    def projection(handles):
        bank = torch.stack([h.weight.half() for h in handles])
        def run(inputs, selected):
            return inputs @ bank[selected.clamp_max(len(bank) - 1)].transpose(1, 2)
        return run

    model.gate_up, model.down = [projection(handles) for handles in zip(*pairs)]
    actual = model._batched(x, weights, ids)
    expected = torch.zeros_like(actual)
    for i, (gate_up, down) in enumerate(pairs):
        tokens, slots = torch.where(ids == i)
        gate, up = F.linear(x[tokens], gate_up.weight.half()).chunk(2, -1)
        output = F.linear(moe.softcapped_swiglu(gate, up), down.weight.half())
        expected[tokens] += output.float() * weights[tokens, slots, None]
    torch.testing.assert_close(actual, expected, rtol=.002, atol=.002)


def test_mixed_projection_pointer_tables(monkeypatch):
    import sys
    device = torch.device('cpu')
    handles = [SimpleNamespace(K=bits, mcg=False, mul1=True, out_features=2,
                               trellis=torch.randn(2, 3).half(), suh=torch.ones(3), svh=torch.ones(2))
               for bits in (3, 3.5, 3, 3.5)]
    by_address = {h.trellis.data_ptr(): h for h in handles}
    def pointers(tensors, device):
        return torch.tensor([t.data_ptr() for t in tensors], device=device)
    def mgemm(x, trellis, out, suh, scratch, svh, indices, weights, bits, *args):
        for slot, expert in enumerate(indices[0].tolist()):
            if expert < 0:
                continue
            handle = by_address[int(trellis[expert])]
            assert handle.K == bits
            assert int(suh[expert]) == handle.suh.data_ptr()
            assert int(svh[expert]) == handle.svh.data_ptr()
            out[slot] = F.linear(x[0 if x.shape[0] == 1 else slot], handle.trellis)

    monkeypatch.setitem(sys.modules, 'exllamav3.modules.moe_batch_recon', SimpleNamespace(_addr_table=pointers))
    monkeypatch.setitem(sys.modules, 'exllamav3.ext', SimpleNamespace(exllamav3_ext=SimpleNamespace(exl3_mgemm=mgemm)))
    grouped = moe._ExpertProjection(handles, device)
    experts = torch.tensor([3, 0, 2, 1, 0])
    for inputs in (torch.randn(1, 1, 3).half(), torch.randn(5, 16, 3).half()):
        expected = torch.stack([F.linear(row, handles[idx].trellis) for row, idx in
                                zip(inputs.expand(5, -1, -1), experts.tolist())])
        torch.testing.assert_close(grouped(inputs, experts), expected, rtol=0, atol=0)
    workspace = {}
    inputs = torch.randn(1, 1, 3).half()
    for ids in (experts, experts.flip(0)):
        expected = torch.stack([F.linear(inputs[0], handles[i].trellis) for i in ids.tolist()])
        result = grouped(inputs, ids, workspace)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert len(workspace) == 1 and result is next(iter(workspace.values()))[0]


@pytest.mark.parametrize('rates', [(3, 3.5), (4, 8)])
def test_cooperative_packed_tables_and_reduction(monkeypatch, rates):
    """Run the native binding contract with CPU stand-ins for the two GEMVs."""
    import sys
    torch.manual_seed(13)
    hidden = inter = 128
    experts = 12
    device = torch.device('cpu')
    router = torch.randn(experts, hidden)
    handles = {}
    def handle(inp, out, rate):
        h = SimpleNamespace(trellis=torch.randn(out, inp).half() / 8, suh=torch.ones(inp).half(),
                            svh=torch.ones(out).half(), K=rate, in_features=inp, out_features=out,
                            config=SimpleNamespace(infer_params=SimpleNamespace(no_reconstruct=True)),
                            bias=None, mul1=True, mcg=False)
        handles[h.trellis.data_ptr()] = h
        return h

    pairs = [(handle(hidden, inter * 2, rates[i % 2]), handle(inter, hidden, rates[1 - i % 2]))
             for i in range(experts)]
    model = moe.Inkling2MoE(router, torch.randn(experts), pairs)
    monkeypatch.setitem(sys.modules, 'exllamav3.modules.moe_batch_recon', SimpleNamespace(
        _addr_table=lambda tensors, device: torch.tensor([t.data_ptr() for t in tensors], device=device)))
    model.gate_up, model.down = [moe._ExpertProjection(h, device) for h in zip(*pairs)]
    model.coop_tables = model._coop_tables()
    model.batched = True
    model.expert_bounds = torch.arange(experts + 1)
    calls = []

    def cooperative(x, ids, rw, tables, tiles, gc, uc, gu, activation, down, ctr, out):
        assert rw.dtype == out.dtype == torch.float32
        assert gu.shape == (2, ids.numel(), inter) and torch.count_nonzero(ctr) == 0
        calls.append(ctr.data_ptr())
        assert (tiles is None) == (len(x) <= 8)
        if tiles is not None:
            assert tiles.shape[1] == 16
            valid = tiles[tiles >= 0].long()
            assert torch.equal(valid.sort().values, torch.arange(ids.numel()))
            selected = ids.flatten()[tiles.long().clamp_min(0)]
            assert torch.all((selected == selected[:, :1]) | (tiles < 0))
            assert torch.all(tiles[:, 1:].lt(0) | tiles[:, :-1].ge(0))
        for row in range(len(x)):
            out[row].zero_()
            for slot, e in enumerate(ids[row]):
                matrices = []
                for fields in tables:
                    addr, suh, svh, rate = fields[:, e].tolist()
                    h = handles[addr]
                    assert (suh, svh, rate) == (h.suh.data_ptr(), h.svh.data_ptr(), h.K * 2)
                    matrices.append(h.trellis)
                g, u = F.linear(x[row], matrices[0]).chunk(2)
                projected = F.linear(moe.softcapped_swiglu(g, u, gc, uc), matrices[1])
                out[row] += rw[row, slot] * projected.float()

    monkeypatch.setitem(sys.modules, 'exllamav3.ext', SimpleNamespace(
        exllamav3_ext=SimpleNamespace(inkling_moe_coop=cooperative)))
    workspace = {}
    for rows in (1, 8, 1, 33):
        x = torch.randn(rows, hidden).half()
        _, weights, ids = moe.route(x, router, model.router_bias)
        expected = torch.zeros_like(x, dtype=torch.float)
        for i, (gu, d) in enumerate(pairs):
            r, k = torch.where(ids == i)
            g, u = F.linear(x[r], gu.trellis).chunk(2, -1)
            expected[r] += weights[r, k, None] * F.linear(moe.softcapped_swiglu(g, u), d.trellis).float()
        torch.testing.assert_close(model(x, workspace), expected.half(), atol=.008, rtol=.003)
    assert len(calls) == 4 and calls[0] == calls[2] and len(workspace) == 3

    # Check the release chunk boundary without thousands of dense CPU GEMVs.
    widths = []
    def copy_input(x, ids, rw, tables, tiles, gc, uc, gu, activation, down, ctr, out):
        widths.append(len(x))
        if tiles is not None:
            valid = tiles[tiles >= 0].sort().values
            torch.testing.assert_close(valid, torch.arange(ids.numel(), dtype=valid.dtype))
        out.copy_(x)
    monkeypatch.setattr(sys.modules['exllamav3.ext'].exllamav3_ext, 'inkling_moe_coop', copy_input)
    x = torch.randn(2049, hidden).half()
    torch.testing.assert_close(model(x[:2048]), x[:2048])
    torch.testing.assert_close(model(x), x)
    assert widths == [2048, 2048, 1]
