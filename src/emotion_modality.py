"""
Adapter that wraps the existing EmotionEngine (face emotion) in the common
Modality interface, so the adaptive gate can treat it like any other sensor.

EmotionEngine needs torch + OpenCV + MediaPipe, so it is imported lazily here.
"""

from __future__ import annotations

from typing import Optional
import time
import numpy as np

from src.modality_base import Modality, ModalityResult


class EmotionModality(Modality):
    """Face-emotion sensing as a Modality (modality #1)."""

    name = "face"
    nominal_latency_ms = 19.9  # measured p50 on Orin Nano, 15W
    nominal_power_w = 7.8       # measured avg W on Orin Nano, 15W
    reliability = 0.635        # measured FER-2013 test accuracy

    def __init__(self, model_path: Optional[str] = None, smoothing_window: int = 5):
        from src.emotion_engine import EmotionEngine
        from src.config import BEST_MODEL_PATH
        self.engine = EmotionEngine(str(model_path or BEST_MODEL_PATH), smoothing_window=smoothing_window)

    def infer(self, frame) -> ModalityResult:
        """frame: a BGR image (H, W, 3) as used by the rest of the system."""
        img = frame["image"] if isinstance(frame, dict) else frame
        t0 = time.perf_counter()
        state = self.engine.process_frame(np.asarray(img))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if state is None:  # no face found this frame
            return ModalityResult(self.name, None, 0.0, latency_ms,
                                  energy_j=self.nominal_power_w * (latency_ms / 1000.0), reliability=self.reliability, ran=True)
        return ModalityResult(
            modality=self.name,
            label=state.label_name,
            confidence=float(state.confidence),
            latency_ms=latency_ms,
            energy_j=self.nominal_power_w * (latency_ms / 1000.0),
            reliability=self.reliability,
            extra={"valence": state.valence, "arousal": state.arousal},
        )

    def close(self):
        self.engine.close()
