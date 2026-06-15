"""
Common interface for every sensing modality (face, audio, gesture, ...).

This is the small shared "shape" that makes an adaptive switch possible:
each modality takes an input, runs its model, and returns a ModalityResult
with a label, a confidence, and the cost (latency / energy) it took to produce.

The SimulatedModality at the bottom lets us run and test the smart switch and
the benchmark harness with NO camera, microphone, or trained model — useful for
development, unit tests, and a runnable demo on any machine (including the Jetson).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import time
import numpy as np


@dataclass
class ModalityResult:
    """One reading from one modality."""
    modality: str
    label: Optional[str]          # predicted class (or None if no detection / not run)
    confidence: float             # 0.0 - 1.0
    latency_ms: float             # how long this reading took
    energy_j: float = 0.0         # estimated energy used for this reading (Joules)
    reliability: float = 1.0      # prior trust in this modality (e.g. its known accuracy)
    ran: bool = True              # False if the modality was skipped or the sensor was dead
    extra: Dict[str, Any] = field(default_factory=dict)


class Modality(ABC):
    """Base class for a sensing modality."""

    #: short name, e.g. "face", "audio", "gesture"
    name: str = "modality"
    #: rough cost profile used by the gate BEFORE running (updated by measurement)
    nominal_latency_ms: float = 10.0
    nominal_power_w: float = 1.0
    #: prior trust in this modality (roughly its known accuracy); used by the gate
    reliability: float = 0.7

    @abstractmethod
    def infer(self, frame: Any) -> ModalityResult:
        """Run the model on one input and return a ModalityResult."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any resources (override if needed)."""
        return None


class SimulatedModality(Modality):
    """
    A fake modality with a configurable accuracy / latency / power profile.

    `infer` expects a dict frame like:
        {"true_label": "happy", "quality": 1.0}
    where `quality` in [0, 1] models degraded conditions (1.0 = perfect,
    lower = harder, e.g. dim light or noise). Lower quality lowers the
    effective accuracy, which is how we test graceful degradation.
    """

    def __init__(
        self,
        name: str,
        classes: List[str],
        accuracy: float,
        latency_ms: float,
        power_w: float,
        latency_jitter: float = 0.15,
        seed: int = 0,
        sleep: bool = False,
    ):
        self.name = name
        self.classes = list(classes)
        self.accuracy = float(accuracy)
        self.nominal_latency_ms = float(latency_ms)
        self.nominal_power_w = float(power_w)
        self.latency_jitter = float(latency_jitter)
        self.sleep = bool(sleep)
        self.reliability = float(accuracy)
        self._rng = np.random.default_rng(seed)

    def infer(self, frame: Any) -> ModalityResult:
        frame = frame or {}
        true_label = frame.get("true_label")
        quality = float(frame.get("quality", 1.0))

        # Effective accuracy drops as conditions degrade.
        eff_acc = max(0.0, min(1.0, self.accuracy * quality))

        # Sample latency around the nominal value (never negative).
        jitter = 1.0 + self._rng.normal(0.0, self.latency_jitter)
        latency_ms = max(0.1, self.nominal_latency_ms * jitter)
        if self.sleep:
            time.sleep(latency_ms / 1000.0)

        # Decide prediction.
        if true_label is not None and self._rng.random() < eff_acc:
            label = true_label
            confidence = float(min(1.0, self._rng.uniform(0.7, 0.99) * (0.5 + 0.5 * quality)))
        else:
            choices = [c for c in self.classes if c != true_label] or self.classes
            label = str(self._rng.choice(choices))
            confidence = float(self._rng.uniform(0.30, 0.65))

        energy_j = self.nominal_power_w * (latency_ms / 1000.0)
        return ModalityResult(
            modality=self.name,
            label=label,
            confidence=confidence,
            latency_ms=latency_ms,
            energy_j=energy_j,
            reliability=self.reliability,
            ran=True,
            extra={"quality": quality, "eff_acc": eff_acc},
        )
