"""
The "smart switch": an accuracy / latency / power-aware policy that decides
which sensing modalities to run for each input.

Three strategies are provided so they can be compared head-to-head (these are
exactly the three conditions in the research plan):

  1. single   -> run ONE modality, always (the unimodal baseline)
  2. all_on   -> run ALL modalities every time and fuse (static late fusion)
  3. adaptive -> run modalities cheapest-first; stop early once the EVIDENCE for
                 one answer is strong enough, and never exceed the latency / power
                 budget (the contribution)

Key idea: each modality's vote is weighted by its reliability (its known
accuracy), and we stop only when the accumulated evidence for the leading
answer crosses a threshold. A single cheap-but-unreliable sensor therefore is
NOT enough on its own; a second sensor that agrees pushes evidence over the line.

Pure Python + NumPy. No torch, no hardware needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.modality_base import Modality, ModalityResult


@dataclass
class GateDecision:
    """Result of running a strategy on one input."""
    final_label: Optional[str]
    final_confidence: float          # agreement posterior in [0, 1]
    evidence: float                  # absolute evidence for the leading label
    modalities_run: List[str]
    results: List[ModalityResult]
    total_latency_ms: float
    total_energy_j: float


def fuse(results: List[ModalityResult]) -> Tuple[Optional[str], float, float]:
    """
    Reliability-weighted vote across the modalities that ran.

    Returns (label, evidence, posterior) where:
      * evidence  = summed reliability*confidence for the winning label (absolute)
      * posterior = evidence / total evidence across all labels (agreement, 0-1)
    """
    scores: Dict[str, float] = defaultdict(float)
    total = 0.0
    for r in results:
        if not r.ran or r.label is None:
            continue
        w = max(r.reliability, 1e-6) * r.confidence
        scores[r.label] += w
        total += w
    if not scores:
        return None, 0.0, 0.0
    best = max(scores, key=scores.get)
    evidence = scores[best]
    posterior = evidence / total if total > 0 else 0.0
    return best, evidence, posterior


def _totals(results: List[ModalityResult]) -> Tuple[float, float]:
    lat = sum(r.latency_ms for r in results if r.ran)
    eng = sum(r.energy_j for r in results if r.ran)
    return lat, eng


class AdaptiveGate:
    """
    Adaptive modality selection.

    Args:
        modalities: list of Modality objects.
        evidence_threshold: stop early once leading-label evidence reaches this.
            Tune so one mediocre sensor isn't enough but two agreeing sensors are.
        latency_budget_ms: optional hard cap on total latency per input.
        power_budget_w: optional cap on the peak power of any single modality
            we are willing to switch on (skips modalities above the cap).
        order: optional explicit order; default is cheapest-first by
            (nominal_latency_ms * nominal_power_w) = rough energy cost.
    """

    def __init__(
        self,
        modalities: List[Modality],
        evidence_threshold: float = 0.75,
        latency_budget_ms: Optional[float] = None,
        power_budget_w: Optional[float] = None,
        order: Optional[List[str]] = None,
    ):
        self.modalities = list(modalities)
        self.evidence_threshold = float(evidence_threshold)
        self.latency_budget_ms = latency_budget_ms
        self.power_budget_w = power_budget_w
        if order:
            idx = {m.name: i for i, m in enumerate(self.modalities)}
            self._ordered = [self.modalities[idx[n]] for n in order if n in idx]
        else:
            self._ordered = sorted(
                self.modalities,
                key=lambda m: m.nominal_latency_ms * max(m.nominal_power_w, 1e-6),
            )

    @staticmethod
    def _frame_for(m: Modality, frame):
        """Allow a per-sensor frame dict {name: subframe} or a single shared frame."""
        if isinstance(frame, dict) and m.name in frame:
            return frame[m.name]
        return frame

    # ---- strategy 3: the contribution ----
    def decide_and_run(self, frame) -> GateDecision:
        results: List[ModalityResult] = []
        spent_ms = 0.0
        for m in self._ordered:
            if self.power_budget_w is not None and m.nominal_power_w > self.power_budget_w:
                continue
            if (self.latency_budget_ms is not None
                    and spent_ms + m.nominal_latency_ms > self.latency_budget_ms
                    and results):
                break

            r = m.infer(self._frame_for(m, frame))
            results.append(r)
            spent_ms += r.latency_ms if r.ran else 0.0

            _, evidence, _ = fuse(results)
            if evidence >= self.evidence_threshold:
                break  # strong enough — stop early, save time and power

        label, evidence, posterior = fuse(results)
        lat, eng = _totals(results)
        return GateDecision(label, posterior, evidence,
                            [r.modality for r in results if r.ran], results, lat, eng)

    # ---- strategy 2: static late fusion ----
    def run_all(self, frame) -> GateDecision:
        results = [m.infer(self._frame_for(m, frame)) for m in self.modalities]
        label, evidence, posterior = fuse(results)
        lat, eng = _totals(results)
        return GateDecision(label, posterior, evidence,
                            [r.modality for r in results if r.ran], results, lat, eng)

    # ---- strategy 1: single modality ----
    def run_single(self, frame, name: str) -> GateDecision:
        m = next(m for m in self.modalities if m.name == name)
        r = m.infer(self._frame_for(m, frame))
        label, evidence, posterior = fuse([r])
        lat, eng = _totals([r])
        return GateDecision(label, posterior, evidence,
                            [r.modality] if r.ran else [], [r], lat, eng)
