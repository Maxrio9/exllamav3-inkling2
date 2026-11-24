from __future__ import annotations
from collections import deque
from statistics import median
import math


class DraftConfidenceCalibrator:
    """
    Online mapping from a drafter's per-position confidence score to an observed
    acceptance probability, used to truncate draft blocks at the first position whose estimated
    acceptance falls below a target confidence.

    Scores are collected into fixed-width bins of decayed (tested, accepted) counts. Labels come
    only from positions the verifier actually tested: accepted positions, the first mismatch, and
    the bonus token sampled right after a fully accepted (possibly truncated) window, which
    labels the first position below the current threshold.

    The threshold is the lower edge of the lowest bin such that it and every populated bin above
    it show an acceptance rate >= confidence. Sparse bins (effective count < min_count) are
    skipped rather than trusted. Until burn_in labels have been seen, the threshold is -inf
    (full windows).
    """

    def __init__(
        self,
        confidence: float,
        bin_width: float = 1.0,
        decay: float = 0.995,
        min_count: float = 8.0,
        burn_in: float = 64.0,
    ):
        assert 0.0 < confidence < 1.0, "draft_confidence must be in (0, 1)"
        self.confidence = confidence
        self.bin_width = bin_width
        self.decay = decay
        self.min_count = min_count
        self.burn_in = burn_in

        self.bins = {}  # bin index -> [tested, accepted], exponentially decayed
        self.total = 0.0
        self.cached_threshold = None


    def add_label(self, score: float, accepted: bool):
        idx = math.floor(score / self.bin_width)
        b = self.bins.get(idx)
        if b is None:
            b = self.bins[idx] = [0.0, 0.0]
        b[0] += 1.0
        if accepted:
            b[1] += 1.0
        self.total += 1.0
        self.cached_threshold = None


    def decay_step(self):
        """
        Age the statistics; call once per verification round so the mapping tracks drift in
        output style (prose vs code etc.) over a few hundred rounds.
        """
        for b in self.bins.values():
            b[0] *= self.decay
            b[1] *= self.decay
        self.total *= self.decay
        self.cached_threshold = None

    def threshold(self) -> float:
        """
        Confidence score below which draft positions should be cut. -inf while insufficient
        data has been collected; +inf if even the most confident bins fall short of the target
        (recovery then relies on the bonus-token labels of empty windows).
        """
        if self.cached_threshold is not None:
            return self.cached_threshold
        if self.total < self.burn_in:
            return -math.inf  # not cached; keeps returning -inf only until burn-in completes
        thr = math.inf
        any_populated = False
        for idx in sorted(self.bins.keys(), reverse = True):
            tested, accepted = self.bins[idx]
            if tested < self.min_count:
                continue
            any_populated = True
            if accepted / tested >= self.confidence:
                thr = idx * self.bin_width
            else:
                break
        # No bin has accumulated enough labels to judge: still learning, keep full windows so
        # labels keep flowing (cutting everything would starve the statistics instead). +inf is
        # reserved for the case where populated top bins genuinely fail the target
        if not any_populated:
            thr = -math.inf
        self.cached_threshold = thr
        return thr

    def estimate(self, score: float) -> float:
        """
        Estimated conditional acceptance probability for a drafted position with the given
        confidence score, from the nearest populated bin at or below it (falling back to the
        nearest above). Optimistic 1.0 while no statistics are available, so sequential
        drafting keeps producing full windows (and labels) during the learning phase.
        """
        if self.total < self.burn_in or not self.bins:
            return 1.0
        idx = math.floor(score / self.bin_width)
        populated = sorted(k for k, v in self.bins.items() if v[0] >= self.min_count)
        if not populated:
            return 1.0
        below = [k for k in populated if k <= idx]
        k = below[-1] if below else populated[0]
        tested, accepted = self.bins[k]
        return accepted / tested


class DSparkGain:
    """Compare completed draft rounds to single steps in the same request."""

    def __init__(self, probe_interval, *, backoff=False, confidence_width=False, narrow_probes=False):
        if probe_interval < 1:
            raise ValueError("DSpark probe interval must be positive")
        self.confidence_width = confidence_width
        self.probe_interval = probe_interval
        self.plain_seconds = deque(maxlen=9)
        self.verify_seconds = deque(maxlen=9)
        self.verify_tokens = deque(maxlen=9)
        self.plain_since_probe = 0
        self.backoff = backoff or narrow_probes
        self.next_probe = probe_interval
        self.plain_warmup = self.verify_warmup = 2
        self.drafting = False
        self.losses = 0
        self.narrow_probes = narrow_probes

    def proposal_count(self, count):
        # Limit the cost of a probe until observed acceptance pays for full rounds.
        return min(count, 2) if self.narrow_probes and not self.drafting else count

    def choose_width(self, count):
        # Once the request sustains wide rounds, preserve their throughput.
        return self.confidence_width and not self.drafting and self.verify_warmup == 0 and count >= 3

    def should_draft(self):
        if len(self.verify_seconds) < 3:
            return True
        if len(self.plain_seconds) < 3:
            return False
        return self.drafting or self.plain_since_probe >= self.next_probe

    def record(self, drafted, emitted, seconds):
        if emitted <= 0:
            return
        if not drafted:
            self.plain_since_probe += emitted
            if self.plain_warmup:
                self.plain_warmup -= 1
                return
            self.plain_seconds.append(seconds / emitted)
            if len(self.plain_seconds) == 3 and self.verify_seconds:
                self.drafting = self._profitable()
            return

        probe = bool(self.plain_seconds) and self.plain_since_probe >= self.next_probe
        self.plain_since_probe = 0
        if self.verify_warmup:
            self.verify_warmup -= 1
            return
        if probe:
            self.verify_tokens.clear()
        self.verify_seconds.append(seconds)
        self.verify_tokens.append(emitted)
        if len(self.plain_seconds) < 3:
            return
        profitable = self._profitable()
        if probe:
            # A multi-token match warrants a second observation at a new
            # position. Three consecutive losses are needed to leave drafting.
            self.drafting = profitable or (self.confidence_width and emitted >= 3)
            self.losses = 0
            if self.backoff and not self.drafting:
                self.next_probe = min(4 * self.probe_interval, 2 * self.next_probe)
            else:
                self.next_probe = self.probe_interval
        elif profitable:
            self.drafting = True
            self.losses = 0
        else:
            self.losses += 1
            if self.losses >= 3:
                self.drafting = False

    def _profitable(self):
        # One emission per round is the target bonus/replacement. The others
        # are accepted proposals, averaged over the current sequence phase.
        emitted = sum(self.verify_tokens) / len(self.verify_tokens)
        return emitted * median(self.plain_seconds) >= median(self.verify_seconds)


def scalar_dspark_verifier(device):
    """Width-dependent target rounding on sm_120 can change greedy IDs.

    Opt-in (INKLING_DSPARK_EXACT=1): scalar verification keeps greedy DSpark bit-identical to plain
    decoding but removes its speed-up, so by default the batched verifier is used (REPORT-p4.md).
    """
    import os
    if device.type != "cuda" or os.environ.get("INKLING_DSPARK_EXACT", "0") != "1":
        return False
    import torch
    props = torch.cuda.get_device_properties(device)
    return (props.major, props.minor) == (12, 0) and props.name == "NVIDIA RTX PRO 4500 Blackwell"
