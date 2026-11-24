"""Reader for the single-file Inkling EXL3 release layout.

The reader owns one handle per physical packed group. It deliberately opens
only model.safetensors; the optional vision and MTP files are not text weights.
"""
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open


class Inkling2ReleaseWeights:
    def __init__(self, directory, config, device, linear_factory=None):
        self.directory = Path(directory)
        self.config = config
        self.device = torch.device(device)
        self.linear_factory = linear_factory
        self.config.infer_params.no_reconstruct = True
        with (self.directory / 'quantization_config.json').open(encoding='utf8') as f:
            metadata = json.load(f)
        self.bits = metadata['bits']
        if not isinstance(self.bits, dict):
            raise ValueError('quantization_config.json bits must be a group-to-rate mapping')
        self.path = self.directory / 'model.safetensors'
        with safe_open(self.path, framework='pt', device='cpu') as f:
            self.keys = frozenset(f.keys())
        self._handles = {}
        self._mapping = None

    @contextmanager
    def mapped(self):
        """Parse the large release header once for the whole component load."""
        if self._mapping is not None:
            yield
            return
        with safe_open(self.path, framework='pt', device='cpu') as mapping:
            self._mapping = mapping
            try:
                yield
            finally:
                self._mapping = None

    def tensor(self, key):
        if key not in self.keys:
            raise KeyError(f'missing release tensor {key} in {self.path}')
        if self._mapping is not None:
            return self._mapping.get_tensor(key).to(self.device)
        with safe_open(self.path, framework='pt', device='cpu') as f:
            return f.get_tensor(key).to(self.device)

    def linear(self, group, in_features, out_features):
        if group in self._handles:
            handle = self._handles[group]
            if (handle.in_features, handle.out_features) != (in_features, out_features):
                raise ValueError(f'{group}: inconsistent requested dimensions')
            return handle
        names = ('trellis', 'suh', 'svh', 'mul1')
        tensors = {name: self.tensor(f'{group}.{name}') for name in names}
        trellis, suh, svh, mul1 = (tensors[name] for name in names)
        if (in_features <= 0 or out_features <= 0 or in_features % 16 or out_features % 16
                or trellis.ndim != 3 or tuple(trellis.shape[:2]) != (in_features // 16, out_features // 16)
                or trellis.dtype != torch.int16 or suh.shape != (in_features,) or svh.shape != (out_features,)
                or suh.dtype != torch.float16 or svh.dtype != torch.float16 or mul1.ndim != 0
                or mul1.dtype != torch.int32):
            raise ValueError(f'{group}: invalid EXL3 field shapes or dtypes')
        rate = trellis.shape[-1] / 16
        if rate not in (2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8):
            raise ValueError(f'{group}: unsupported EXL3 bit rate {rate}')
        if self.bits.get(group) != rate:
            raise ValueError(f'{group}: quantization_config bit rate does not match trellis ({rate})')
        factory = self.linear_factory
        if factory is None:
            from ..modules.quant.exl3 import LinearEXL3
            factory = LinearEXL3
        handle = factory(config=self.config, in_features=in_features, out_features=out_features,
                         key=group, **tensors)
        self._handles[group] = handle
        return handle


def build_text_components(raw, reader, dtype=torch.float16):
    """Construct the standalone text components from the release's official names.

    This does not attach ExLlama's paged cache or generator. The caller supplies
    the reader's device and may inject a linear factory for CPU-only tests.
    """
    from torch import nn
    from ..modules.arch_specific.inkling2_attention import Inkling2Attention
    from ..modules.arch_specific.inkling2_moe import Inkling2MoE
    from ..modules.arch_specific.inkling2_loop import (LoopLayer, InklingLoop,
                                                       AttentionAdapter, MoEAdapter, ConvAdapter)
    from .inkling2_schedule import inkling2_schedule_from_config

    text = raw['text_config']
    loop = text['loop_config']
    n, hidden, experts = text['num_hidden_layers'], text['hidden_size'], text['n_routed_experts']
    start, end = loop['core_start'], loop['core_end']
    eps = text.get('rms_norm_eps', 1e-6)
    root = 'model.llm.'

    def tensor(key, as_dtype=None):
        value = reader.tensor(root + key)
        return value.to(dtype=as_dtype) if as_dtype is not None else value

    def parameter(value):
        return nn.Parameter(value, requires_grad=False)

    def assign(dst, value, name):
        if dst.shape != value.shape:
            if dst.numel() != value.numel():
                raise ValueError(f'{name}: expected {tuple(dst.shape)}, got {tuple(value.shape)}')
            value = value.reshape(dst.shape)
        dst.data = value.to(device=reader.device, dtype=dst.dtype)

    class CastLinear(nn.Linear):
        def forward(self, x):
            return torch.nn.functional.linear(x.to(self.weight.dtype), self.weight)

    def linear_weight(key):
        weight = tensor(key, dtype)
        linear = CastLinear(weight.shape[1], weight.shape[0], bias=False, device='meta')
        linear.weight = parameter(weight)
        return linear

    layers = []
    for i in range(n):
        base = f'layers.{i}.'
        local = i in text.get('local_layer_ids', [j for j in range(n) if (j + 1) % 6])
        heads = text['swa_num_attention_heads'] if local else text['num_attention_heads']
        kv_heads = text['swa_num_key_value_heads'] if local else text['num_key_value_heads']
        head_dim = text['swa_head_dim'] if local else text['head_dim']
        qdim, kdim = heads * head_dim, kv_heads * head_dim
        inter = text['intermediate_size']
        def packed(suffix, d_in, d_out):
            return reader.linear(f'{i}.{suffix}', d_in, d_out)
        attention_args = dict(
            q=packed('self_attn.q_proj.linear', hidden, qdim),
            k=packed('self_attn.k_proj.linear', hidden, kdim),
            v=packed('self_attn.v_proj.linear', hidden, kdim),
            o=packed('self_attn.o_proj.linear', qdim, hidden),
            gate=packed('self_attn.o_proj.gate_proj', hidden, qdim),
            rel_linear=linear_weight(base + 'attn.wr_du.weight'),
            sink=linear_weight(base + 'attn.attn_sink_proj.weight'),
            q_scale=tensor(base + 'attn.q_l2_clip.scale'),
            rel_scale=tensor(base + 'attn.rel_l2_clip.scale'),
            rel_proj=tensor(base + 'attn.rel_logits_proj.proj', dtype),
            heads=heads, kv_heads=kv_heads, head_dim=head_dim,
            qk_threshold=text['qk_l2_clip_threshold'], rel_threshold=text['rel_l2_clip_threshold'],
            global_attention=not local, window=text.get('sliding_window_size', 1024),
            rel_extent=text.get('rel_extent', 1024), eps=eps,
            log_floor=text.get('log_scaling_n_floor', 8192), log_alpha=text.get('log_scaling_alpha', .1))
        if local:
            attention_args.update(k_conv=tensor(base + 'attn.k_sconv.weight'),
                                  v_conv=tensor(base + 'attn.v_sconv.weight'))
        else:
            attention_args.update(k_conv=tensor(base + 'attn.k_sconv.static_w'),
                                  v_conv=tensor(base + 'attn.v_sconv.static_w'),
                                  k_dynamic=linear_weight(base + 'attn.k_sconv.weight_proj'),
                                  v_dynamic=linear_weight(base + 'attn.v_sconv.weight_proj'))
        attention = Inkling2Attention(**attention_args)
        pairs = [(packed(f'experts.gate_up_proj.{e}', hidden, 2 * inter),
                  packed(f'experts.down_proj.{e}', inter, hidden)) for e in range(experts)]
        moe = Inkling2MoE(tensor(base + 'mlp.gate.weight', torch.float32),
                          tensor(base + 'mlp.gate.bias', torch.float32), pairs,
                          text['num_experts_per_tok'], text.get('route_scale', 8.),
                          text.get('mlp_gate_softcap', 4.), text.get('mlp_up_softcap', 25.))
        layer = LoopLayer(i, hidden, AttentionAdapter(attention), MoEAdapter(moe),
                          ConvAdapter(parameter(tensor(base + 'attn_sconv.weight', torch.float32)), 2),
                          ConvAdapter(parameter(tensor(base + 'mlp_sconv.weight', torch.float32)), 3),
                          full=not local, in_loop=start <= i < end,
                          block_size=text.get('n_layers_per_attn_res_block', 6),
                          rank=loop.get('gate_rank', 64), eps=eps).to(reader.device)
        if i:
            assign(layer.attention_input.query, tensor(f'attn_res.attn_queries.{i-1}'), 'attention query')
        assign(layer.mlp_input.query, tensor(f'attn_res.mlp_queries.{i}'), 'mlp query')
        norm_names = ['attn_norm.weight'] if local else [f'{q}_input_norm.weight' for q in ('q', 'k', 'v')]
        for norm, name in zip(layer.attn_norms, norm_names):
            assign(norm.weight, tensor(base + name), name)
        assign(layer.mlp_norm.weight, tensor(base + 'mlp_norm.weight'), 'mlp norm')
        if layer.gates is not None:
            for gate, stream in zip(layer.gates, ('attn', 'k', 'v', 'mlp')):
                prefix = base + f'controllers.{stream}.'
                for projection, suffix, target_dtype in (
                    (gate.down_proj, 'down_weight', dtype),
                    (gate.forget_gate_proj, 'forget_weight', torch.float32),
                    (gate.message_gate_proj, 'gain_weight', torch.float32),
                ):
                    value = tensor(prefix + suffix, target_dtype)
                    if projection.weight.shape != value.shape:
                        raise ValueError(f'{prefix + suffix}: invalid shape {tuple(value.shape)}')
                    projection.weight = parameter(value)
                gate.pack_outputs()
        layers.append(layer)
    head_weight = tensor('unembed.weight', dtype)
    head = nn.Linear(hidden, head_weight.shape[0], bias=False, device=reader.device, dtype=dtype)
    head.weight = parameter(head_weight)
    model = InklingLoop(layers, inkling2_schedule_from_config(raw), start, end,
                        loop['num_passes'], hidden, head, eps=eps,
                        softcap=text.get('final_logit_softcapping', 30.)).to(reader.device)
    a = tensor('middle_cycle_cross_pass_queries', torch.float32).view(-1, 2, hidden)
    kv = tensor('middle_cycle_cross_pass_kv_queries', torch.float32).view(-1, 2, hidden)
    assign(model.message_query, torch.stack((a[:, 0], kv[:, 0], kv[:, 1], a[:, 1]), dim=1), 'message query')
    assign(model.final_input.query, tensor('attn_res.output_query'), 'output query')
    assign(model.final_norm.weight, tensor('norm.weight'), 'final norm')
    embedding = nn.Embedding.from_pretrained(tensor('embed.weight', dtype))
    embed_norm = nn.RMSNorm(hidden, eps=eps, device=reader.device)
    assign(embed_norm.weight, tensor('embed_norm.weight'), 'embedding norm')
    return embedding, embed_norm, model
