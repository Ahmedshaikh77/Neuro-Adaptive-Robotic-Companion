"""
Fault injection for robustness tests (the part most HRI papers skip).

Two kinds of helpers:
  * image/audio degraders that physically alter the input
    (low light, sensor noise, occlusion) — useful with the real camera/mic;
  * a `quality` scalar in [0, 1] for the SimulatedModality, so the same
    conditions can be tested with no hardware.

Also `drop_sensor`, which marks a modality's reading as a dead sensor.

NumPy only. OpenCV is NOT required.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional
import numpy as np

from src.modality_base import ModalityResult


# ---------- image degraders (HxWx3 uint8) ----------

def low_light(img: np.ndarray, factor: float = 0.3) -> np.ndarray:
    """Scale brightness down. factor=1.0 unchanged, 0.0 black."""
    return np.clip(img.astype(np.float32) * float(factor), 0, 255).astype(np.uint8)


def add_noise(img: np.ndarray, sigma: float = 25.0, seed: Optional[int] = None) -> np.ndarray:
    """Add Gaussian sensor noise (sigma in 0-255 units)."""
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, size=img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def occlude(img: np.ndarray, frac: float = 0.4, seed: Optional[int] = None) -> np.ndarray:
    """Black out a random rectangle covering ~`frac` of the image area."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    h, w = img.shape[:2]
    bh, bw = int(h * np.sqrt(frac)), int(w * np.sqrt(frac))
    if bh < 1 or bw < 1:
        return out
    y = int(rng.integers(0, max(1, h - bh)))
    x = int(rng.integers(0, max(1, w - bw)))
    out[y:y + bh, x:x + bw] = 0
    return out


# ---------- audio degrader (1-D float waveform) ----------

def add_audio_noise(wav: np.ndarray, snr_db: float = 10.0, seed: Optional[int] = None) -> np.ndarray:
    """Add white noise at a target signal-to-noise ratio (dB)."""
    rng = np.random.default_rng(seed)
    sig_power = float(np.mean(wav.astype(np.float64) ** 2)) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=wav.shape)
    return (wav.astype(np.float64) + noise).astype(wav.dtype if np.issubdtype(wav.dtype, np.floating) else np.float32)


# ---------- condition -> quality scalar (for SimulatedModality) ----------

CONDITION_QUALITY = {
    "clean": 1.0,
    "low_light": 0.45,     # dim room hurts the camera
    "audio_noise": 0.5,    # background noise hurts the mic
    "occlusion": 0.35,     # partly blocked camera
    "dropout": 0.0,        # sensor produces nothing
}


def quality_for(condition: str, modality_name: str) -> float:
    """
    How much a named condition degrades a given modality.
    Conditions are sensor-specific: low light hurts vision, not audio.
    """
    c = condition.lower()
    vision = modality_name in ("face", "gesture")
    audio = modality_name in ("audio", "speech")
    if c in ("low_light", "occlusion"):
        return CONDITION_QUALITY[c] if vision else 1.0
    if c == "audio_noise":
        return CONDITION_QUALITY[c] if audio else 1.0
    if c == "dropout":
        return 0.0
    return 1.0


def drop_sensor(result: ModalityResult) -> ModalityResult:
    """Return a copy of a reading marked as a dead sensor (no output)."""
    return replace(result, label=None, confidence=0.0, ran=False, extra={**result.extra, "dropped": True})
