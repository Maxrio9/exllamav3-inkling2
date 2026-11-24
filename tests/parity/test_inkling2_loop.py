"""Integration coverage for the Inkling attention, MoE, convolution and loop slices.

This deliberately does not assert ExLlama generator or checkpoint-loader support.
"""
import importlib.util
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


attn_test = load('inkling_integration_attention_test', ROOT / 'tests/inkling2/test_attention.py')
moe_test = load('inkling_integration_moe_test', ROOT / 'tests/inkling2/test_moe.py')
loop_test = load('inkling_integration_loop_test', ROOT / 'tests/inkling2/test_loop.py')
from inkling_test_modules.inkling2_loop import LoopLayer, InklingLoop, RequestState, AttentionAdapter, ConvAdapter, MoEAdapter
schedule = loop_test.arch.inkling2_visit_schedule


def build():
    torch.manual_seed(31415)
    layers, refs = [], []
    for i in range(4):
        _, attention_ref, attention = attn_test.make(i % 2 == 1)
        cfg = SimpleNamespace(n_routed_experts=3, num_local_experts=3, hidden_size=8,
            moe_intermediate_size=5, num_experts_per_tok=2, scoring_func='sigmoid',
            routed_scaling_factor=8., mlp_gate_softcap=4., mlp_up_softcap=25.)
        moe_ref = moe_test.oracle()['Inkling2LoopSparseMoeBlock'](cfg)
        with torch.no_grad():
            for p in moe_ref.parameters(): p.normal_(0, .15)
        pairs = [(moe_test.MockLinear(moe_ref.experts.gate_up_proj[j], 3),
                  moe_test.MockLinear(moe_ref.experts.down_proj[j], 3)) for j in range(3)]
        moe = moe_test.moe.Inkling2MoE(moe_ref.gate.weight, moe_ref.gate.e_score_correction_bias, pairs, 2)
        convs = [nn.Parameter(torch.randn(8, 1, 3) * .1) for _ in range(2)]
        layer = LoopLayer(i, 8, AttentionAdapter(attention), MoEAdapter(moe),
                          ConvAdapter(convs[0], 2), ConvAdapter(convs[1], 3),
                          full=i % 2 == 1, in_loop=1 <= i < 3, block_size=2, rank=4)
        layers.append(layer)
        refs.append((attention_ref, moe_ref, convs))
    model = InklingLoop(layers, schedule(4, 1, 3, 3), 1, 3, 3, 8, nn.Linear(8, 13, bias=False))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'weight' not in name or 'conv' not in name:
                p.normal_(0, .1)
        for layer in layers:
            for norm in [*layer.attn_norms, layer.mlp_norm]: norm.weight.fill_(1)
        model.final_norm.weight.fill_(1)
    return model.eval(), refs


def reference(model, refs, x):
    classes = loop_test.oracle_classes()
    cfg = SimpleNamespace(hidden_size=x.shape[-1], loop_gate_rank=4, rms_norm_eps=1e-6,
                          loop_layer_index_start=1, loop_layer_index_end=3, _attn_implementation='eager')
    query = classes['Inkling2LoopMessageAttentionResidual'](cfg)
    query.query = model.message_query
    def residual(src):
        if src is None: return None
        dst = classes['Inkling2LoopBlockAttentionResidual'](cfg, src.query.shape[0])
        dst.query = nn.Parameter(src.query.squeeze(0) if src.query.shape[0] == 1 else src.query)
        return dst
    conv_class = attn_test.load_reference()['Inkling2LoopShortConvolution']
    def ref_conv(weight, index, slot):
        conv = conv_class(x.shape[-1], weight.shape[-1], index, slot)
        conv.conv1d.weight = weight
        return conv
    n = x.shape[1]
    positions = torch.arange(n)[None]
    distance = positions[:, :, None] - positions[:, None, :]
    hidden, blocks = x, x.new_zeros(*x.shape[:-1], 0, x.shape[-1])
    trace = []
    for visit in model.schedule:
        i, p = visit.layer, visit.pass_index
        if i == 1:
            if p == 0: pre_hidden, pre_blocks = hidden, blocks
            else:
                previous_hidden = hidden
                previous_blocks = blocks[..., pre_blocks.shape[-2] + 1:, :]
                hidden, blocks = pre_hidden, pre_blocks
        source = model.layers[i]
        kind = ('Full' if source.full else 'Sliding') + ('Loop' if source.in_loop else '')
        cls = classes['Inkling2Loop' + kind + 'DecoderLayer']
        layer = cls.__new__(cls); nn.Module.__init__(layer)
        layer.is_block_boundary = source.boundary
        layer.attention_input, layer.mlp_input = residual(source.attention_input), residual(source.mlp_input)
        layer.self_attn, layer.mlp = refs[i][:2]
        layer.post_attention_convolution = ref_conv(refs[i][2][0], i, 2)
        layer.post_mlp_convolution = ref_conv(refs[i][2][1], i, 3)
        if source.full:
            layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm = source.attn_norms
        else: layer.input_layernorm = source.attn_norms[0]
        layer.post_attention_layernorm = source.mlp_norm
        if source.in_loop:
            layer.q_message_gate, layer.k_message_gate, layer.v_message_gate, layer.mlp_message_gate = source.gates
        valid = distance >= 0
        if not source.full: valid = valid & (distance < 1024)
        mask = x.new_zeros(1, 1, n, n).masked_fill(~valid[:, None], torch.finfo(x.dtype).min)
        kwargs = dict(attention_mask=mask, relative_distance=distance[:, None], position_ids=positions)
        if source.in_loop: kwargs['messages'] = query(previous_blocks, previous_hidden, i - 1) if p else None
        old = hidden
        hidden = layer(hidden, blocks, **kwargs)
        if source.boundary: blocks = torch.cat((blocks, old.unsqueeze(-2)), -2)
        trace.append(hidden)
    hidden = residual(model.final_input)(blocks, hidden)[0]
    logits = model.lm_head(model.final_norm(hidden)).float()
    return 30 * torch.tanh(logits / 30), trace


def check(a, b):
    d = (a - b).abs()
    print(f'max={d.max().item():.9g} mean={d.mean().item():.9g} cosine={F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item():.9g} top1={(a.argmax(-1)==b.argmax(-1)).float().mean().item():.6f}')
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_combined_slices_reference_and_resume():
    torch.set_num_threads(1)
    model, refs = build()
    torch.manual_seed(77)
    x, y = torch.randn(1, 6, 8), torch.randn(1, 6, 8)
    with torch.no_grad():
        trace = []
        whole = model(x, RequestState(), lambda v, h, b: trace.append(h))
        expected, ref_trace = reference(model, refs, x)
        check(whole, expected)
        for a, b in zip(trace, ref_trace): check(a, b)
        state, other = RequestState(), RequestState()
        pieces, other_pieces = [], []
        for i in range(6):
            pieces.append(model(x[:, i:i+1], state))
            other_pieces.append(model(y[:, i:i+1], other))
        check(torch.cat(pieces, 1), whole)
        check(torch.cat(other_pieces, 1), model(y, RequestState()))
        assert len(state.visits) == len(model.schedule)
        assert torch.equal(torch.cat(pieces, 1).argmax(-1), whole.argmax(-1))
        prefix = RequestState()
        model(x[:, :3], prefix)
        resumed = copy.deepcopy(prefix)
        check(model(x[:, 3:], resumed), whole[:, 3:])


def test_combined_slices_cross_window():
    torch.set_num_threads(1)
    model, _ = build()
    torch.manual_seed(9)
    x = torch.randn(1, 1026, 8)
    with torch.no_grad():
        whole = model(x, RequestState())
        state = RequestState()
        parts = [model(x[:, :1023], state), model(x[:, 1023:1025], state),
                 model(x[:, 1025:], state)]
        check(torch.cat(parts, 1), whole)
