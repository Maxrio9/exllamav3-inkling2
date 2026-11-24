# SPDX-License-Identifier: Apache-2.0
# DSpark math adapted from vLLM, Copyright contributors to the vLLM project.
"""Pure-torch Inkling DSpark and request-local verification primitives.

The generator must provide external cache transactions and sampling transforms.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


def proposal_token_ids(directory, vocab_size, limit=32768):
    """Select a fixed support from the engine's bundled text."""
    path = Path(directory) / 'tokenizer.json'
    if vocab_size <= limit or not path.is_file():
        return None
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    corpus = Path(__file__).resolve().parents[2] / 'conversion' / 'standard_cal_data'
    counts = Counter()
    for source in sorted(corpus.glob('*.utf8')):
        text = source.read_text(encoding='utf-8')
        for start in range(0, len(text), 16384):
            counts.update(tokenizer.encode(text[start:start + 16384], add_special_tokens=False).ids)
    if not counts:
        return None
    required = {i for i in tokenizer.get_added_tokens_decoder() if i < vocab_size}
    if len(required) >= limit:
        return None
    eligible = (i for i in counts if i < vocab_size and i not in required)
    common = sorted(eligible, key=lambda i: (-counts[i], i))[:limit - len(required)]
    return sorted(required.union(common))


def _check(condition, message):
    if condition.is_cuda:
        torch._assert_async(condition, message)
    elif not condition:
        raise ValueError(message)


class VisitTaps:
    """Collect pre-visit partial residuals and the final normalized output."""
    def __init__(self, visit_count, indices=(5, 23, 41, 48)):
        if not indices or tuple(sorted(set(indices))) != tuple(indices) or indices[-1] != visit_count:
            raise ValueError('tap IDs must be sorted and end at the visit count')
        self.indices, self.visit_count = tuple(indices), visit_count
        self.values = {}

    def before_visit(self, visit_index, partial):
        if visit_index in self.indices[:-1]:
            if visit_index in self.values:
                raise ValueError('duplicate visit tap')
            self.values[visit_index] = partial.clone()

    def final(self, hidden):
        if self.visit_count in self.values:
            raise ValueError('duplicate final tap')
        self.values[self.visit_count] = hidden.clone()

    def result(self):
        if set(self.values) != set(self.indices):
            raise ValueError('missing target taps')
        return torch.stack([self.values[i] for i in self.indices], dim=-2)


def mix_features(features, fuse_weight, norm_weight, eps):
    """vLLM DFlare: softmax-weighted sum, then RMS norm in FP32."""
    alpha = fuse_weight.float().softmax(-1)
    mixed = (features.float() * alpha[..., None]).sum(-2)
    return (mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + eps)
            * norm_weight.float()).to(features.dtype)


def markov_step(state, previous_embedding, hidden, context, weight, bias):
    """One rnn_hidden recurrence; returns the state and latent correction."""
    rank = state.shape[-1]
    fixed = F.linear(torch.cat((hidden, context), -1).float(), weight[:, 2 * rank:].float(), bias.float())
    return _markov_update(state, previous_embedding, fixed, weight[:, :2 * rank].float().T)


def _markov_update(state, previous_embedding, fixed, dynamic_weight):
    dynamic = torch.cat((state, previous_embedding), -1).float()
    raw = torch.addmm(fixed, dynamic, dynamic_weight).to(state.dtype)
    gate, candidate, output = raw.chunk(3, dim=-1)
    gate = gate.sigmoid()
    return gate * state + (1 - gate) * candidate.tanh(), output.tanh()


@dataclass(frozen=True)
class Verification:
    accepted: int
    tokens: tuple


def restore_state(state, snapshot):
    """Restore every nested request-owned state, including per-visit caches."""
    saved = deepcopy(snapshot)
    if isinstance(state, dict):
        state.clear()
        state.update(saved)
    elif hasattr(state, '__dict__') and hasattr(saved, '__dict__'):
        state.__dict__.clear()
        state.__dict__.update(saved.__dict__)
    else:
        raise TypeError('state must support in-place restoration')


def verify_greedy(state, initial_logits, proposals, step):
    """Verify a block and replay its accepted prefix after any rejection.

    `step(state, token)` advances one token and returns next-token logits.
    On rejection the replacement is also advanced, so state represents all
    returned tokens. There is no bonus token on full acceptance.
    """
    snapshot = deepcopy(state)
    try:
        logits = initial_logits
        checks = []
        for proposal in proposals:
            checks.append(int(torch.argmax(logits, dim=-1).item()))
            logits = step(state, int(proposal))
        accepted = next((i for i, (p, expected) in enumerate(zip(proposals, checks))
                         if int(p) != expected), len(checks))
        if accepted == len(checks):
            return Verification(accepted, tuple(map(int, proposals))), logits
        tokens = tuple(map(int, proposals[:accepted])) + (checks[accepted],)
        restore_state(state, snapshot)
        for token in tokens:
            logits = step(state, token)
        return Verification(accepted, tokens), logits
    except BaseException:
        restore_state(state, snapshot)
        raise


class DraftRMSNorm(torch.nn.Module):
    def __init__(self, width, eps, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width, dtype=dtype))
        self.eps = eps

    def forward(self, x, residual=None):
        value = x.float() if residual is None else x.float() + residual.float()
        normalized = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
                      * self.weight.float()).to(x.dtype)
        if residual is None:
            return normalized
        return normalized, value.to(x.dtype)


def _rope(x, positions, theta):
    """NeoX (split-half) rotary embedding on [batch, time, heads, dim]."""
    dim = x.shape[-1]
    inv = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    angle = positions.float()[..., None] * inv
    cos, sin = angle.cos().unsqueeze(-2), angle.sin().unsqueeze(-2)
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), -1).to(x.dtype)


class DraftProjections:
    """Pack inference projections sharing an input while retaining checkpoint keys."""

    def pack(self, name, *layers):
        weight = torch.cat([layer.weight.detach() for layer in layers])
        self.register_buffer(name, weight, persistent=False)
        sizes = [layer.out_features for layer in layers]
        for layer, view in zip(layers, weight.split(sizes), strict=True):
            layer.weight = torch.nn.Parameter(view, requires_grad=False)

    def project(self, name, x, *layers):
        packed = getattr(self, name, None)
        if packed is None:
            return tuple(layer(x) for layer in layers)
        return F.linear(x, packed).split([layer.out_features for layer in layers], dim=-1)


class DraftAttention(DraftProjections, torch.nn.Module):
    def __init__(self, config, local, dtype):
        super().__init__()
        h, d = config.hidden_size, config.head_dim
        self.heads, self.kv_heads, self.dim = config.num_attention_heads, config.num_key_value_heads, d
        self.local, self.window, self.theta = local, config.sliding_window_size, config.rope_theta
        kw = {'bias': False, 'dtype': dtype}
        self.q_proj = torch.nn.Linear(h, self.heads * d, **kw)
        self.k_proj = torch.nn.Linear(h, self.kv_heads * d, **kw)
        self.v_proj = torch.nn.Linear(h, self.kv_heads * d, **kw)
        self.k_proj_ctx = torch.nn.Linear(h, self.kv_heads * d, **kw)
        self.v_proj_ctx = torch.nn.Linear(h, self.kv_heads * d, **kw)
        self.o_proj = torch.nn.Linear(self.heads * d, h, **kw)
        self.q_norm = DraftRMSNorm(d, config.rms_norm_eps, dtype)
        self.k_norm = DraftRMSNorm(d, config.rms_norm_eps, dtype)
        self.k_norm_ctx = DraftRMSNorm(d, config.rms_norm_eps, dtype)

    def context_kv(self, context, positions):
        b, c, _ = context.shape
        key, value = self.project('context_weight', context, self.k_proj_ctx, self.v_proj_ctx)
        key = self.k_norm_ctx(key.view(b, c, self.kv_heads, self.dim))
        return _rope(key, positions, self.theta), value.view_as(key)

    def forward(self, x, positions, context, context_positions):
        b, n, _ = x.shape
        ck, cv = self.context_kv(context, context_positions) if isinstance(context, torch.Tensor) else context
        c = ck.shape[1]
        q, k, v = self.project('qkv_weight', x, self.q_proj, self.k_proj, self.v_proj)
        q = _rope(self.q_norm(q.view(b, n, self.heads, self.dim)), positions, self.theta)
        k = _rope(self.k_norm(k.view(b, n, self.kv_heads, self.dim)), positions, self.theta)
        v = v.view(b, n, self.kv_heads, self.dim)
        keys, values = torch.cat((ck, k), 1), torch.cat((cv, v), 1)
        repeat = self.heads // self.kv_heads
        queries = q.transpose(1, 2).reshape(b, self.kv_heads, repeat * n, self.dim)
        scores = torch.matmul(queries.float(), keys.permute(0, 2, 3, 1).float()) * self.dim ** -0.5
        start = positions[:, :1]
        valid = context_positions < start
        if self.local:
            valid = valid & (context_positions >= start - self.window)
        scores[..., :c].masked_fill_(~valid[:, None, None, :], float('-inf'))
        out = torch.matmul(scores.softmax(-1), values.transpose(1, 2).float())
        out = out.reshape(b, self.heads, n, self.dim).transpose(1, 2).reshape(b, n, -1).to(x.dtype)
        return self.o_proj(out)


class DraftMLP(DraftProjections, torch.nn.Module):
    def __init__(self, config, dtype):
        super().__init__()
        h, i = config.hidden_size, config.intermediate_size
        self.gate_proj = torch.nn.Linear(h, i, bias=False, dtype=dtype)
        self.up_proj = torch.nn.Linear(h, i, bias=False, dtype=dtype)
        self.down_proj = torch.nn.Linear(i, h, bias=False, dtype=dtype)

    def forward(self, x):
        gate, up = self.project('gate_up_weight', x, self.gate_proj, self.up_proj)
        return self.down_proj((F.silu(gate.float()) * up.float()).to(x.dtype))


class DraftLayer(torch.nn.Module):
    def __init__(self, config, index, dtype):
        super().__init__()
        h, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = DraftRMSNorm(h, eps, dtype)
        self.post_attention_layernorm = DraftRMSNorm(h, eps, dtype)
        self.dflare_norm = DraftRMSNorm(h, eps, dtype)
        self.self_attn = DraftAttention(config, index in config.local_layer_ids, dtype)
        self.mlp = DraftMLP(config, dtype)

    def forward(self, x, residual, positions, context, context_positions):
        if residual is None:
            residual = x
            x = self.input_layernorm(x)
        else:
            x, residual = self.input_layernorm(x, residual)
        x = self.self_attn(x, positions, context, context_positions)
        x, residual = self.post_attention_layernorm(x, residual)
        return self.mlp(x), residual


class DraftMarkovHead(torch.nn.Module):
    def __init__(self, config, vocab_size, dtype):
        super().__init__()
        h, r = config.hidden_size, config.markov_rank
        self.markov_w1 = torch.nn.Embedding(vocab_size, r, dtype=dtype)
        self.rnn_proj = torch.nn.Linear(2 * r + 2 * h, 3 * r, dtype=dtype)
        self.hidden_out = torch.nn.Linear(r, h, dtype=dtype)
        self.dflare_norm = DraftRMSNorm(h, config.rms_norm_eps, dtype)

    def next_correction(self, state, prev, fixed, dynamic_weight, folded_head):
        embedding = self.markov_w1(prev)
        state, latent = _markov_update(state, embedding, fixed, dynamic_weight)
        return state, F.linear(latent, *folded_head)

    def fold(self, head_weight):
        # vLLM folds in fp64, then rounds to the drafter's dtype. Tile the
        # vocabulary to bound the fp64 temporary for the shared target head.
        projection, bias = self.hidden_out.weight.double(), self.hidden_out.bias.double()
        rows = head_weight.shape[0]
        folded = self.hidden_out.weight.new_empty((rows, self.markov_w1.embedding_dim))
        offset = folded.new_empty(rows)
        for start in range(0, rows, 4096):
            interval = slice(start, start + 4096)
            head = head_weight[interval].double()
            folded[interval].copy_(head @ projection)
            offset[interval].copy_(head @ bias)
        return folded, offset

    def forward(self, state, prev, hidden, context, folded_head):
        state, latent = markov_step(state, self.markov_w1(prev), hidden, context,
                                    self.rnn_proj.weight, self.rnn_proj.bias)
        correction = F.linear(latent, *folded_head)
        return state, correction


class Inkling2DSpark(torch.nn.Module):
    """Pure torch single-request draft model; target embedding/head stay shared.

    Context features are [1, context, taps, hidden], block positions [1, B].
    Registered requests append committed context KV through context_kv().
    """
    def __init__(self, config, vocab_size, *, dtype=torch.bfloat16, softcap=30.0, unpadded_vocab_size=None):
        super().__init__()
        self.config, self.softcap = config, softcap
        self.vocab_size = unpadded_vocab_size or vocab_size
        if not 0 < self.vocab_size <= vocab_size:
            raise ValueError('invalid unpadded vocabulary size')
        self.layers = torch.nn.ModuleList(DraftLayer(config, i, dtype) for i in range(config.num_hidden_layers))
        self.norm = DraftRMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.markov_head = DraftMarkovHead(config, vocab_size, dtype)
        self.fuse_weight = torch.nn.Parameter(torch.zeros(config.num_hidden_layers + 1, len(config.target_layer_ids), dtype=dtype))
        self.register_buffer('initial_state', torch.zeros((1, config.markov_rank), dtype=dtype), persistent=False)
        self.mask_embed = torch.nn.Parameter(torch.zeros(config.hidden_size, dtype=dtype))

    @torch.no_grad()
    def prepare_inference(self):
        for layer in self.layers:
            attn, mlp = layer.self_attn, layer.mlp
            attn.pack('qkv_weight', attn.q_proj, attn.k_proj, attn.v_proj)
            attn.pack('context_weight', attn.k_proj_ctx, attn.v_proj_ctx)
            mlp.pack('gate_up_weight', mlp.gate_proj, mlp.up_proj)
        # The fixed projection covers the complete proposal block. Only the
        # small state/previous-token projection depends on an earlier choice.
        proj = self.markov_head.rnn_proj
        split = 2 * self.config.markov_rank
        self.register_buffer('fixed_weight', proj.weight[:, split:].float().contiguous(), persistent=False)
        self.register_buffer('dynamic_weight', proj.weight[:, :split].float().T.contiguous(), persistent=False)
        self.register_buffer('fixed_bias', proj.bias.float(), persistent=False)

    @torch.no_grad()
    def prepare_proposal_vocab(self, ids, head_weight, folded_head):
        ids = torch.tensor(ids, device=head_weight.device, dtype=torch.long)
        self.register_buffer('proposal_ids', ids, persistent=False)
        self.register_buffer('proposal_weight', head_weight.index_select(0, ids), persistent=False)
        self.register_buffer('proposal_folded', folded_head[0].index_select(0, ids), persistent=False)
        self.register_buffer('proposal_bias', folded_head[1].index_select(0, ids), persistent=False)

    def load_mtp(self, path):
        """Load released BF16 mtp.safetensors, rejecting missing or extra keys."""
        from safetensors.torch import load_file
        tensors = load_file(str(path), device='cpu')
        if any(value.dtype != torch.bfloat16 for value in tensors.values()):
            raise ValueError('mtp checkpoint tensors must be BF16')
        if any(value.dtype != torch.bfloat16 for value in self.state_dict().values()):
            raise ValueError('mtp model must have BF16 parameters')
        self.load_state_dict(tensors, strict=True)
        if hasattr(self, 'fixed_weight'):
            self.prepare_inference()

    def forward(self, anchor_ids, positions, context_features, context_positions, embed_tokens, embed_norm):
        cfg = self.config
        if anchor_ids.shape != (1,) or positions.shape != (1, cfg.block_size):
            raise ValueError('expected one request and a complete draft block')
        if context_features.shape != (1, context_positions.shape[1], len(cfg.target_layer_ids), cfg.hidden_size):
            raise ValueError('context features/positions shape mismatch')
        if not context_positions.shape[1]:
            raise ValueError('the committed anchor feature is required')
        if context_positions.shape[0] != 1:
            raise ValueError('invalid context or block positions')
        valid = ((context_positions < positions[:, :1]).all()
                 & (context_positions[:, 1:] > context_positions[:, :-1]).all()
                 & (context_positions >= 0).all() & (positions >= 0).all()
                 & (positions < cfg.max_position_embeddings).all())
        _check(valid, 'invalid context or block positions')
        _check((positions[:, 1:] - positions[:, :-1] == 1).all(), 'block positions must be contiguous')
        context = [self.context_kv(i, context_features, context_positions) for i in range(len(self.layers))]
        return self.forward_block(anchor_ids, positions, context, context_positions, embed_tokens, embed_norm)

    def context_kv(self, i, features, positions):
        layer = self.layers[i]
        context = mix_features(features, self.fuse_weight[i], layer.dflare_norm.weight, layer.dflare_norm.eps)
        return layer.self_attn.context_kv(context, positions)

    def forward_block(self, anchor_ids, positions, context, context_positions, embed_tokens, embed_norm):
        x = self.mask_embed.view(1, 1, -1).expand(1, self.config.block_size, -1).clone()
        x[:, 0] = embed_norm(embed_tokens(anchor_ids))
        residual = None
        for i, layer in enumerate(self.layers):
            x, residual = layer(x, residual, positions, context[i], context_positions)
        x, _ = self.norm(x, residual)
        return x

    def propose(self, anchor_ids, positions, context_features, context_positions, embed_tokens, embed_norm, head_weight,
                *, generator=None, sample=False, transform_logits=None, folded_head=None, num_proposals=None,
                distribution=None, use_draft_vocab=False):
        """Return tokens, model logits and actual proposal distributions.

        For stochastic decoding, transform_logits(logits, generated_ids) must
        apply the same temperature, masks and penalties used for the draft.
        With use_draft_vocab, logits passed to the transform follow proposal_ids
        order. Generated IDs and returned tensors always use target vocabulary IDs.
        """
        if num_proposals is None:
            num_proposals = self.config.block_size - 1
        if not 0 < num_proposals < self.config.block_size:
            raise ValueError('invalid number of draft proposals')
        hidden = self(anchor_ids, positions, context_features, context_positions, embed_tokens, embed_norm)
        return self.proposal_head(hidden, anchor_ids, context_features[:, -1], head_weight, folded_head,
                                  num_proposals, sample, generator, transform_logits, distribution=distribution,
                                  use_draft_vocab=use_draft_vocab)

    def proposal_head(self, hidden, anchor_ids, last_features, head_weight, folded_head, num_proposals,
                      sample=False, generator=None, transform_logits=None, tokens_only=False, distribution=None,
                      use_draft_vocab=False):
        ids = getattr(self, 'proposal_ids', None) if use_draft_vocab else None
        if ids is not None:
            head_weight = self.proposal_weight
            folded_head = self.proposal_folded, self.proposal_bias
        else:
            head_weight = head_weight[:self.vocab_size]
            if folded_head is None:
                folded_head = self.markov_head.fold(head_weight)
        f_rnn = mix_features(last_features, self.fuse_weight[-1], self.markov_head.dflare_norm.weight, self.config.rms_norm_eps)
        state = self.initial_state
        prev, tokens, logits_out, probs_out = anchor_ids, [], [], []
        rows = hidden[:, 1:1 + num_proposals]
        base_rows = F.linear(rows.to(head_weight.dtype), head_weight).to(hidden.dtype)
        proj = self.markov_head.rnn_proj
        split = 2 * state.shape[-1]
        fixed_input = torch.cat((rows, f_rnn[:, None, :].expand_as(rows)), -1)
        if hasattr(self, 'fixed_weight'):
            fixed = F.linear(fixed_input.float(), self.fixed_weight, self.fixed_bias)
            dynamic_weight = self.dynamic_weight
        else:
            fixed = F.linear(fixed_input.float(), proj.weight[:, split:].float(), proj.bias.float())
            dynamic_weight = proj.weight[:, :split].float().T
        for t in range(num_proposals):
            state, correction = self.markov_head.next_correction(state, prev, fixed[:, t],
                                                                dynamic_weight, folded_head)
            base = base_rows[:, t]
            logits = (base.float() + correction.float()).to(base.dtype)
            logits = torch.tanh(logits / self.softcap) * self.softcap
            if tokens_only:
                prev = logits.argmax(-1)
                if ids is not None:
                    prev = ids.index_select(0, prev)
                tokens.append(prev)
                continue
            if distribution is not None:
                probs = distribution(logits)
                greedy = probs
            else:
                sampling_logits = logits.float()
                if transform_logits is not None:
                    sampling_logits = transform_logits(sampling_logits, tuple(tokens))
                    if sampling_logits.shape != logits.shape:
                        raise ValueError('draft sampling transform changed the logits shape')
                probs = sampling_logits.softmax(-1)
                greedy = sampling_logits
            _check(torch.isfinite(probs).all(), 'invalid transformed draft logits')
            if ids is not None and sample:
                # Draw in target-ID order so identical proposal weights consume
                # the same random numbers as the full vocabulary.
                probs = probs.new_zeros((probs.shape[0], self.vocab_size)).index_copy_(-1, ids, probs)
            prev = (torch.multinomial(probs, 1, generator=generator).squeeze(-1) if sample
                    else greedy.argmax(-1))
            if ids is not None and not sample:
                prev = ids.index_select(0, prev)
            tokens.append(prev)
            logits_out.append(logits)
            probs_out.append(probs)
        if tokens_only:
            return (torch.stack(tokens, 1),)
        logits, probs = torch.stack(logits_out, 1), torch.stack(probs_out, 1)
        if ids is not None:
            shape = (*logits.shape[:-1], self.vocab_size)
            logits = logits.new_full(shape, -torch.inf).index_copy_(-1, ids, logits)
            if not sample:
                probs = probs.new_zeros(shape).index_copy_(-1, ids, probs)
        return torch.stack(tokens, 1), logits, probs


def confidence_width(probs):
    """Choose one, two or all proposals using only the prefix before each draw."""
    peaks = probs[0, :3].amax(-1)
    reach_two = peaks[0] * peaks[1]
    wide = torch.where(reach_two * peaks[2] >= 0.45, probs.shape[1], 2)
    return torch.where(reach_two < 0.25, 1, wide)


def _verification_result(proposals, count, anchor, return_anchor):
    # One public API boundary: the scheduler needs the accepted length to roll
    # back its host slot, and Verification.tokens is a tuple of Python integers.
    accepted, token, *ids = torch.cat((count, anchor.view(1), proposals)).tolist()
    result = Verification(accepted, (*ids[:accepted], token))
    return (result, anchor.view(())) if return_anchor else result


def accept_greedy(target_logits, proposals, *, return_anchor=False):
    """Select from the rows predicting the proposals and a final bonus."""
    ids = target_logits.argmax(-1)
    proposals = torch.as_tensor(proposals, dtype=torch.long, device=ids.device)
    count = (ids[:-1] == proposals).long().cumprod(0).sum().view(1)
    return _verification_result(proposals, count, ids.index_select(0, count), return_anchor)


def accept_sampled(target_logits, proposals, proposal_probs, *, generator=None,
                   acceptance_uniforms=None, gumbel_noise=None, return_anchor=False):
    """Verify the proposal prefix and sample the rejection residual or bonus."""
    logits = target_logits if isinstance(target_logits, torch.Tensor) else torch.stack(target_logits)
    return accept_sampled_probs(logits.float().softmax(-1), proposals, proposal_probs,
                                generator=generator, acceptance_uniforms=acceptance_uniforms,
                                gumbel_noise=gumbel_noise, return_anchor=return_anchor)


def accept_sampled_probs(target_probs, proposals, proposal_probs, *, generator=None,
                         acceptance_uniforms=None, gumbel_noise=None, return_anchor=False):
    """Verify with the already transformed distributions of the two samplers."""
    n = len(proposals)
    if len(target_probs) != n + 1 or len(proposal_probs) != n:
        raise ValueError('expected one target and proposal distribution per token and a target bonus row')
    if acceptance_uniforms is not None and len(acceptance_uniforms) != n:
        raise ValueError('one acceptance uniform is required per token')

    p = target_probs.float().reshape(n + 1, -1)
    tokens = torch.as_tensor(proposals, dtype=torch.long, device=p.device)
    if n:
        q = proposal_probs
        if not isinstance(q, torch.Tensor):
            q = torch.stack([torch.as_tensor(row, device=p.device) for row in q])
        q = q.to(p).reshape(n, -1)
        if q.shape != p[:-1].shape:
            raise ValueError('invalid proposal distribution shape')
        _check(((tokens >= 0) & (tokens < q.shape[1])).all(), 'invalid sampled token')
        chosen = tokens[:, None]
        chosen_q = q.gather(1, chosen).flatten()
        _check((torch.isfinite(q).all() & (q >= 0).all() & (chosen_q > 0).all()
                & torch.isclose(q.sum(-1), torch.ones_like(chosen_q), atol=1e-5, rtol=1e-5).all()
                & torch.isfinite(p).all()), 'invalid target or proposal distribution')
        if acceptance_uniforms is None:
            u = torch.rand(n, generator=generator, device=q.device)
            u = torch.where(u == 0, torch.nextafter(u, torch.ones_like(u)), u)
        else:
            u = torch.as_tensor(acceptance_uniforms, dtype=q.dtype, device=q.device)
            _check(((u > 0) & (u <= 1)).all(), 'acceptance uniform must be in (0, 1]')
        matches = p[:-1].gather(1, chosen).flatten() > u * chosen_q
        count = matches.long().cumprod(0).sum().view(1)
        row = p.index_select(0, count)[0]
        residual = (row - q.index_select(0, count.clamp_max(n - 1))[0]).clamp_min(0)
        row = torch.where(count == n, row, residual)
    else:
        count = tokens.new_zeros(1)
        row = p[0]
    _check((row > 0).any(), 'invalid rejection residual or bonus')
    if gumbel_noise is None:
        u = torch.rand(row.shape, generator=generator, device=row.device)
        u.clamp_(min=torch.finfo(u.dtype).tiny, max=1 - torch.finfo(u.dtype).eps)
        noise = -(-u.log()).log()
    else:
        noise = torch.as_tensor(gumbel_noise, dtype=row.dtype, device=row.device)
        if noise.shape != row.shape:
            raise ValueError('invalid Gumbel noise')
        _check(torch.isfinite(noise).all(), 'invalid Gumbel noise')
    return _verification_result(tokens, count, (row.log() + noise).argmax(), return_anchor)


def verify_sampled(state, initial_logits, proposals, proposal_probs, step, *, generator=None,
                   acceptance_uniforms=None, gumbel_noise=None):
    """Advance a standalone CPU request through the sampled verification result."""
    if len(proposals) != len(proposal_probs):
        raise ValueError('one proposal distribution is required per token')
    if acceptance_uniforms is not None and len(acceptance_uniforms) != len(proposals):
        raise ValueError('one acceptance uniform is required per token')
    snapshot = deepcopy(state)
    try:
        logits, target_logits = initial_logits, []
        for token in proposals:
            target_logits.append(logits.clone())
            logits = step(state, int(token))
        target_logits.append(logits)
        result = accept_sampled(target_logits, proposals, proposal_probs, generator=generator,
                                acceptance_uniforms=acceptance_uniforms, gumbel_noise=gumbel_noise)
        if result.accepted < len(proposals):
            restore_state(state, snapshot)
            for token in result.tokens:
                logits = step(state, token)
        else:
            logits = step(state, result.tokens[-1])
        return result, logits
    except BaseException:
        restore_state(state, snapshot)
        raise
