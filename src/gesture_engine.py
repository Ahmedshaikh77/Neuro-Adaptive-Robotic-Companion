"""
Gesture recognition (modality #3), trained on the Jester dataset.

A lightweight 2-D CNN runs over a short stack of frames (T frames treated as
channels) so it stays cheap enough for the Jetson. torch is imported lazily,
so importing this module never fails without torch.
"""

from __future__ import annotations

from typing import List, Optional
import time
import numpy as np

from src.modality_base import Modality, ModalityResult

# A small, useful subset of Jester gestures for companion commands.
DEFAULT_GESTURES = [
    "swipe_left", "swipe_right", "swipe_up", "swipe_down",
    "thumb_up", "thumb_down", "stop", "no_gesture",
]


def build_gesture_model(num_classes: int, n_frames: int = 8):
    """2-D CNN over a stack of n_frames grayscale frames. Requires torch."""
    import torch.nn as nn

    class GestureNet(nn.Module):
        def __init__(self, num_classes: int, n_frames: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(n_frames, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.flatten(1)
            return self.classifier(x)

    return GestureNet(num_classes, n_frames)


class GestureModality(Modality):
    """Gesture modality wrapping the temporal CNN."""

    name = "gesture"
    nominal_latency_ms = 2.3   # measured p50 on Orin Nano, 15W
    nominal_power_w = 6.5       # measured avg W on Orin Nano, 15W
    reliability = 0.50         # PLACEHOLDER: gesture model not yet trained

    def __init__(self, checkpoint_path: Optional[str] = None,
                 gestures: Optional[List[str]] = None,
                 n_frames: int = 8, size: int = 96, device: Optional[str] = None):
        import torch
        from src.config import get_device
        self.gestures = gestures or DEFAULT_GESTURES
        self.n_frames = n_frames
        self.size = size
        self.device = device or get_device()
        self.model = build_gesture_model(len(self.gestures), n_frames).to(self.device)
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        self.model.eval()

    def _preprocess(self, frames):
        """frames: list/array of T grayscale frames -> (1, T, H, W) float tensor."""
        import torch
        arr = np.asarray(frames, dtype=np.float32)
        if arr.ndim == 4:  # (T, H, W, C) -> grayscale
            arr = arr.mean(axis=-1)
        # pad/trim to n_frames
        if arr.shape[0] < self.n_frames:
            arr = np.concatenate([arr, np.repeat(arr[-1:], self.n_frames - arr.shape[0], axis=0)], axis=0)
        arr = arr[: self.n_frames]
        arr = arr / 255.0
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def infer(self, frame) -> ModalityResult:
        import torch
        t0 = time.perf_counter()
        frames = frame["frames"] if isinstance(frame, dict) else frame
        x = self._preprocess(frames)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)
            conf, idx = probs.max(1)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ModalityResult(
            modality=self.name,
            label=self.gestures[int(idx.item())],
            confidence=float(conf.item()),
            latency_ms=latency_ms,
            energy_j=self.nominal_power_w * (latency_ms / 1000.0),
            reliability=self.reliability,
        )
