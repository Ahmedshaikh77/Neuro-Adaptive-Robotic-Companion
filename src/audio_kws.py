"""
On-device keyword spotting (audio modality #2).

This replaces the slow cloud speech-to-text for the "spoken command" task with
a small CNN that runs locally on log-mel features (trained on Google Speech
Commands). Running on-device is what removes the network round-trip that makes
replies feel slow today.

torch is imported lazily, so importing this module never fails on a machine
without torch. You need torch (+ a trained checkpoint) to actually run inference.
"""

from __future__ import annotations

from typing import List, Optional
import time
import numpy as np

from src.modality_base import Modality, ModalityResult

# Default Speech Commands "core" keyword set.
DEFAULT_KEYWORDS = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go", "_silence_", "_unknown_",
]


def build_kws_model(num_classes: int, n_mels: int = 40):
    """Small 2-D CNN over a log-mel spectrogram. Requires torch."""
    import torch.nn as nn

    class KWSNet(nn.Module):
        def __init__(self, num_classes: int, n_mels: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.flatten(1)
            return self.classifier(x)

    return KWSNet(num_classes, n_mels)


def log_mel(wav: np.ndarray, sr: int = 16000, n_mels: int = 40, n_fft: int = 400, hop: int = 160) -> np.ndarray:
    """
    Lightweight log-mel feature without external audio libs.
    Returns an (n_mels, frames) float32 array. Good enough as a scaffold;
    swap in torchaudio/librosa for production-quality features.
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size < n_fft:
        wav = np.pad(wav, (0, n_fft - wav.size))
    # framing + FFT magnitude
    frames = 1 + (len(wav) - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    spec = np.empty((n_fft // 2 + 1, frames), dtype=np.float32)
    for i in range(frames):
        seg = wav[i * hop: i * hop + n_fft] * window
        spec[:, i] = np.abs(np.fft.rfft(seg)).astype(np.float32)
    # triangular mel filterbank
    f = np.linspace(0, sr / 2, spec.shape[0])
    mel_pts = np.linspace(0, 2595 * np.log10(1 + (sr / 2) / 700), n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    fb = np.zeros((n_mels, spec.shape[0]), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        for k, freq in enumerate(f):
            if lo <= freq <= ctr and ctr > lo:
                fb[m - 1, k] = (freq - lo) / (ctr - lo)
            elif ctr <= freq <= hi and hi > ctr:
                fb[m - 1, k] = (hi - freq) / (hi - ctr)
    mel = fb @ spec
    return np.log(mel + 1e-6).astype(np.float32)


class KwsModality(Modality):
    """Keyword-spotting modality wrapping the local CNN."""

    name = "audio"
    nominal_latency_ms = 35.0
    nominal_power_w = 1.2
    reliability = 0.90

    def __init__(self, checkpoint_path: Optional[str] = None,
                 keywords: Optional[List[str]] = None, device: Optional[str] = None):
        import torch
        from src.config import get_device
        self.keywords = keywords or DEFAULT_KEYWORDS
        self.device = device or get_device()
        self.model = build_kws_model(len(self.keywords)).to(self.device)
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        self.model.eval()

    def infer(self, frame) -> ModalityResult:
        import torch
        t0 = time.perf_counter()
        wav = frame["wav"] if isinstance(frame, dict) else frame
        feats = log_mel(np.asarray(wav, dtype=np.float32))
        x = torch.from_numpy(feats).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)
            conf, idx = probs.max(1)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ModalityResult(
            modality=self.name,
            label=self.keywords[int(idx.item())],
            confidence=float(conf.item()),
            latency_ms=latency_ms,
            energy_j=self.nominal_power_w * (latency_ms / 1000.0),
            reliability=self.reliability,
        )
