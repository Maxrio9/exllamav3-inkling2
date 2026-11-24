"""Single-request decoding over the Inkling model's paged KV and conv cache."""
from contextlib import contextmanager
import math
import time

import torch

from .draft_confidence import DSparkGain, scalar_dspark_verifier
from ..cache.fp16 import CacheLayer_fp16
from ..cache.quant import CacheLayer_quant
from ..constants import PAGE_SIZE
from ..modules.arch_specific.inkling2_dspark import VisitTaps, Verification, accept_greedy, accept_sampled, confidence_width


class Inkling2DecodeRequest:
    """Decode on caller-reserved pages, with a separate recurrent slot per request.

    For DSpark, the last emitted token is pending until the next step uses it
    as the target and draft anchor. Call flush to commit it at the end.
    """
    def __init__(self, model, cache, pages, *, draft=None, temperature=0., seed=None,
                 dspark_adaptive=True, dspark_probe_interval=64):
        pages = tuple(pages)
        total_pages = cache.max_num_tokens // PAGE_SIZE
        if not pages or len(set(pages)) != len(pages) or any(not 0 <= p < total_pages for p in pages):
            raise ValueError('Invalid Inkling request pages')
        if cache.model is not model or not cache.initialized or not issubclass(
            cache.layer_type, (CacheLayer_fp16, CacheLayer_quant)
        ):
            raise ValueError('Inkling requires an initialized target cache')
        if draft is not None and not issubclass(cache.layer_type, CacheLayer_fp16):
            raise ValueError('Inkling DSpark requires FP16 KV for token-by-token equivalence')
        if draft is not None and (draft.target is not model or draft.draft is None):
            raise ValueError('Load DSpark with the request target first')
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError('Temperature must be finite and nonnegative')

        self.model, self.cache, self.draft = model, cache, draft
        self.pages = pages
        self.capacity = min(len(pages) * PAGE_SIZE, model.config.max_position_embeddings)
        self.device = model.embedding.weight.device
        self._sm120_draft = draft is not None and scalar_dspark_verifier(self.device)
        self.scalar_verify = self._sm120_draft and temperature == 0
        self._block_table = torch.tensor([pages], dtype=torch.int32, device=self.device)
        self._lengths = torch.arange(self.capacity, device=self.device, dtype=torch.int32)
        self.vocab_size = model.config.unpadded_vocab_size or model.loop.lm_head.out_features
        self.temperature = temperature
        self.rng = None
        if temperature > 0:
            self.rng = torch.Generator(device=self.device)
            if seed is None:
                self.rng.seed()
            else:
                self.rng.manual_seed(seed)
        self.dspark_gain = None
        if draft is not None and dspark_adaptive:
            self.dspark_gain = DSparkGain(
                dspark_probe_interval, backoff=self._sm120_draft,
                narrow_probes=getattr(model, 'verify_rows_default', False),
                confidence_width=self.rng is None and not self.scalar_verify and hasattr(draft.draft, 'proposal_ids'),
            )
        self._rollback_buffers = {}
        self._features = None
        if draft is not None:
            cfg = draft.draft_config
            self._positions = torch.arange(self.capacity + cfg.block_size - 1, device=self.device)[None]
            self._features = draft.draft.mask_embed.new_empty((1, self.capacity, len(cfg.target_layer_ids), cfg.hidden_size))
        self._state = cache.get_new_state()
        self._pending = self._logits = None

    def __enter__(self):
        if self._state is None:
            raise ValueError('Inkling request is closed')
        return self

    def __exit__(self, *exc):
        if self._state is not None:
            self._state.free()
            self._state = None

    @property
    def position(self):
        return self._state.position

    @property
    def graph_replays(self):
        return self._state.graph_replays

    @property
    def features(self):
        if self._features is not None:
            return self._features[:, :self.position]
        return None

    def _forward(self, ids, *, last_only=False, record_conv=False, prompt_chunk=False):
        params = dict(attn_mode='flash_attn', cache=self.cache, block_table=self._block_table,
                      cache_seqlens=self._lengths[self.position:self.position + 1], recurrent_states=[self._state])
        params['_inkling_prompt_chunk'] = prompt_chunk
        params['_inkling_record_conv'] = record_conv
        params['_inkling_positions'] = self._lengths[None, self.position:self.position + ids.shape[1]]
        if self.draft is None:
            params['_inkling_borrow_logits'] = True
        if last_only:
            params['last_tokens_only'] = 1
        taps = None
        if self.draft is not None:
            taps = VisitTaps(len(self.model.config.layer_map), self.draft.draft_config.target_layer_ids)
            params['inkling_tap_collector'] = taps
        logits = self.model.forward(ids, params)[0, :, :self.vocab_size]
        return logits, taps.result() if taps is not None else None

    def _commit(self, logits, features):
        if features is not None:
            self._features[:, self.position - features.shape[1]:self.position].copy_(features)
        self._logits = logits[-1]

    @torch.inference_mode()
    def prefill(self, ids):
        if self.position or ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= self.capacity:
            raise ValueError('Expected a nonempty prompt fitting an empty Inkling request')
        # The paged kernels consume multiple logical pages per forward. Bound
        # relative-bias and tap activations independently of the request length.
        for chunk in ids.to(self.device).split(2048, dim=1):
            self._commit(*self._forward(chunk, last_only=True, prompt_chunk=True))

    def _scale_logits(self, logits, _previous=()):
        return (logits - logits.amax(dim=-1, keepdim=True)) / self.temperature

    def _pick(self, logits):
        if self.rng is None:
            return logits.argmax().view(1)
        return torch.multinomial(self._scale_logits(logits).softmax(-1), 1, generator=self.rng)

    @contextmanager
    def _transaction(self, length):
        """Save only the KV rows and convolution slot this round can write."""
        position = self.position
        logits, pending = self._logits, self._pending
        # Reuse a block-sized buffer for each global page segment or ring
        # segment. A wrap can split a plane into two disjoint views.
        saved = []
        recurrent, global_rows = {}, []

        def save(key, view, rows):
            if key not in self._rollback_buffers:
                self._rollback_buffers[key] = view.new_empty((rows, *view.shape[1:]))
            original = self._rollback_buffers[key][:view.shape[0]]
            saved.append((view, original))
            return original

        block = self.draft.draft_config.block_size
        for key, layer in self.cache.recurrent_layers.items():
            originals = recurrent[key] = []
            for i, view in enumerate(layer.checkpoint_views(self._state.slot, position, length)):
                # First four planes are complete fixed-shape conv histories.
                originals.append(save(('recurrent', key, i), view, view.shape[0] if i < 4 else block))
        segment = 0
        start, end = position, position + length
        while start < end:
            page_index, offset = divmod(start, PAGE_SIZE)
            stop = min(PAGE_SIZE, offset + end - start)
            page = self.pages[page_index]
            for key, layer in self.cache.layers.items():
                for i, tensor in enumerate(layer.get_tensors()):
                    view = tensor[page, offset:stop]
                    original = save(('global', key, i, segment), view, block)
                    global_rows.append((start - position, view, original))
            segment += 1
            start += stop - offset
        self._state.copy_rows((original, view) for view, original in saved)
        rng_state = self.rng.get_state() if self.rng is not None else None

        def restore(keep=0):
            if keep:
                copies = [pair for key, planes in recurrent.items()
                          for pair in self._state.retained_copies(key, position, length, keep, planes)]
                for first, view, original in global_rows:
                    cut = max(0, keep - first)
                    if cut < len(view):
                        copies.append((view[cut:], original[cut:]))
                self._state.copy_rows(copies)
            else:
                self._state.copy_rows(saved)
            self._state.position, self._state.last_history = position + keep, 0
            self._logits, self._pending = logits, pending

        try:
            yield restore
        except BaseException:
            restore()
            if rng_state is not None:
                self.rng.set_state(rng_state)
            raise

    def _verify(self, count):
        checkpoint_count = count
        with self._transaction(1 + count) as restore:
            query = self._pending.view(1, 1)
            proposals, probabilities = query.view(-1)[:0], ()
            if count:
                choose_width = self.dspark_gain is not None and self.dspark_gain.choose_width(count)
                ids, _, probs = self.draft.propose(
                    self._pending, self.features, self._positions,
                    num_proposals=count, state=self._state, tokens_only=self.rng is None and not choose_width,
                    use_draft_vocab=True,
                    sample=self.rng is not None, generator=self.rng,
                    transform_logits=self._scale_logits if self.rng is not None else None,
                )
                if choose_width:
                    count = int(confidence_width(probs))
                    ids, probs = ids[:, :count], probs[:, :count]
                proposals = ids[0]
                if self.scalar_verify:
                    # Verify one row at a time and stop on the first rejection.
                    # Only canonical tokens have touched the recurrent state.
                    emitted = []
                    expected = proposals.tolist()
                    for proposal in [*expected, None]:
                        logits, features = self._forward(self._pending.view(1, 1))
                        self._commit(logits, features)
                        self._pending = self._pick(logits[-1])
                        token = self._pending.item()
                        emitted.append(token)
                        if token != proposal:
                            break
                    return Verification(len(emitted) - 1, tuple(emitted)), count
                if probs is not None:
                    probabilities = probs[0]
                query = torch.cat((query, ids), dim=1)
            # Retained convolution histories use the original transaction width.
            retain = count == checkpoint_count and count > 0 and self._state.can_retain(1 + count)
            logits, features = self._forward(query, record_conv=retain)
            if self.rng is None:
                result, next_anchor = accept_greedy(logits, proposals, return_anchor=True)
            else:
                result, next_anchor = accept_sampled(self._scale_logits(logits), proposals, probabilities,
                                                     generator=self.rng, return_anchor=True)
            if result.accepted < count:
                keep = 1 + result.accepted
                if retain:
                    restore(keep)
                    logits, features = logits[:keep], features[:, :keep]
                else:
                    restore()
                    logits, features = self._forward(query[:, :keep], last_only=True)
            self._commit(logits, features)
            self._pending = next_anchor.reshape(1)
        return result, count

    @torch.inference_mode()
    def plain_token(self):
        """Commit and return one sampled ID on device for fixed-budget generation."""
        if self.draft is not None or self._logits is None or self.position >= self.capacity:
            raise ValueError('Expected a prefilled plain request with space for one token')
        token = self._pick(self._logits)
        self._commit(*self._forward(token.view(1, 1)))
        return token

    @torch.inference_mode()
    def step(self, remaining, *, batch_fallback=False):
        """Return (verification result, number proposed) within the remaining token budget."""
        if self._logits is None or not 0 < remaining <= self.capacity - self.position - (self._pending is not None):
            raise ValueError('Prefill first and keep the generation budget within the request pages')
        if self._pending is None:
            if self.draft is None:
                token = self.plain_token()
            else:
                token = self._pick(self._logits)
                self._pending = token
            return Verification(0, (token.item(),)), 0

        cfg = self.draft.draft_config
        count = min(cfg.block_size - 1, remaining - 1)
        if self.position + cfg.block_size > cfg.max_position_embeddings:
            count = 0
        if self.dspark_gain is None or not count:
            return self._verify(count)
        start = time.perf_counter()
        if self.dspark_gain.should_draft():
            result, proposed = self._verify(self.dspark_gain.proposal_count(count))
        else:
            # Advance the committed prefix and its taps, leaving the new
            # token pending as the anchor for a later speculative round.
            gain = self.dspark_gain
            length = 1
            if batch_fallback and self._sm120_draft and len(gain.plain_seconds) >= 3:
                length = min(cfg.block_size, remaining, max(1, gain.next_probe - gain.plain_since_probe))
            with self._transaction(length):
                emitted = []
                for _ in range(length):
                    logits, features = self._forward(self._pending.view(1, 1))
                    self._pending = self._pick(logits[-1])
                    self._commit(logits, features)
                    emitted.append(self._pending)
                tokens = (self._pending.item(),) if length == 1 else tuple(torch.cat(emitted).tolist())
                result = Verification(0, tokens)
            proposed = 0
        self.dspark_gain.record(bool(proposed), len(result.tokens), time.perf_counter() - start)
        return result, proposed

    @torch.inference_mode()
    def flush(self):
        if self._pending is not None:
            with self._transaction(1):
                self._commit(*self._forward(self._pending.view(1, 1)))
                self._pending = None
