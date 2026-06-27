#!/usr/bin/env python3
"""
Multi-mode latency+power benchmark for audio, gesture, and face modalities.
Appends one row per modality to results/latency_power.csv for whatever
nvpmodel power mode is currently active.  Re-running on the same mode is safe:
existing (modality, power_mode) pairs are skipped.

Usage (repeat for each mode):
    sudo nvpmodel -m N && sudo jetson_clocks
    python3 bench_modes.py
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.benchmark import benchmark_runs, PowerMonitor

N_RUNS   = 200
N_WARMUP = 10
CSV_PATH = "results/latency_power.csv"
FIELDS   = ["modality", "power_mode", "p50_ms", "p95_ms", "mean_ms", "avg_power_w", "peak_power_w"]

os.makedirs("results", exist_ok=True)


# ── Power-mode detection ────────────────────────────────────────────────────────

def get_power_mode() -> tuple[int, str]:
    """Return (mode_id, label) from `nvpmodel -q`, or (-1, 'unknown') on failure."""
    try:
        out = subprocess.check_output(
            ["nvpmodel", "-q"], stderr=subprocess.STDOUT, text=True
        )
    except Exception as exc:
        print(f"[WARN] nvpmodel unavailable: {exc}")
        return -1, "unknown"

    label   = "unknown"
    mode_id = -1
    for line in out.splitlines():
        line = line.strip()
        m = re.search(r"NV Power Mode:\s*(.+)", line)
        if m:
            label = m.group(1).strip()
        elif re.fullmatch(r"\d+", line):
            mode_id = int(line)
    return mode_id, label


mode_id, mode_label = get_power_mode()
print(f"Current power mode: id={mode_id}  label={mode_label}")

# ── Load CSV to find already-recorded (modality, mode) pairs ───────────────────

existing: set[tuple[str, str]] = set()
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            existing.add((row["modality"], row["power_mode"]))


def already_recorded(modality: str) -> bool:
    return (modality, str(mode_id)) in existing


# ── CUDA warm-up ────────────────────────────────────────────────────────────────

_ = torch.zeros(1).cuda()
print("CUDA driver loaded.\n")

new_rows: list[dict] = []


def _record(modality: str, bm: dict, pwr: dict) -> None:
    new_rows.append(dict(
        modality      = modality,
        power_mode    = mode_id,
        p50_ms        = round(bm["p50"],  3),
        p95_ms        = round(bm["p95"],  3),
        mean_ms       = round(bm["mean"], 3),
        avg_power_w   = round(pwr["avg_w"],  3),
        peak_power_w  = round(pwr["peak_w"], 3),
    ))
    print(f"  p50={bm['p50']:.2f} ms  p95={bm['p95']:.2f} ms  "
          f"avg_w={pwr['avg_w']:.2f} W  peak_w={pwr['peak_w']:.2f} W")


# ── Audio (KWS CNN) ─────────────────────────────────────────────────────────────

AUDIO = "audio"
if already_recorded(AUDIO):
    print(f"[audio] mode {mode_id} already recorded — skipping.")
else:
    print(f"[audio] benchmarking (mode {mode_id}: {mode_label}) …")
    from src.audio_kws import KwsModality
    kws = KwsModality(checkpoint_path=None)
    kws.model.eval()
    wav_dummy = np.zeros(16000, dtype=np.float32)  # 1 s @ 16 kHz

    def run_audio():
        kws.infer({"wav": wav_dummy})

    pm = PowerMonitor(); pm.start()
    bm  = benchmark_runs(run_audio, n=N_RUNS, warmup=N_WARMUP)
    pwr = pm.stop()
    _record(AUDIO, bm, pwr)

# ── Gesture (temporal CNN) ──────────────────────────────────────────────────────

GESTURE = "gesture"
if already_recorded(GESTURE):
    print(f"\n[gesture] mode {mode_id} already recorded — skipping.")
else:
    print(f"\n[gesture] benchmarking (mode {mode_id}: {mode_label}) …")
    from src.gesture_engine import GestureModality
    gest = GestureModality(checkpoint_path=None, n_frames=8, size=96)
    gest.model.eval()
    frames_dummy = np.zeros((8, 96, 96), dtype=np.float32)  # 8 × 96×96 grayscale

    def run_gesture():
        gest.infer({"frames": frames_dummy})

    pm = PowerMonitor(); pm.start()
    bm  = benchmark_runs(run_gesture, n=N_RUNS, warmup=N_WARMUP)
    pwr = pm.stop()
    _record(GESTURE, bm, pwr)

# ── Face (MediaPipe + ResNet-18 random weights) ──────────────────────────────────

FACE_FULL     = "face_mediapipe_resnet18_randweights"
FACE_FALLBACK = "face_resnet18_randweights_only"

if already_recorded(FACE_FULL) or already_recorded(FACE_FALLBACK):
    print(f"\n[face] mode {mode_id} already recorded — skipping.")
else:
    print(f"\n[face] benchmarking MediaPipe + ResNet-18 (mode {mode_id}: {mode_label}) …")
    try:
        import cv2
        import mediapipe as mp
        import torch.nn as nn
        from PIL import Image
        from torchvision import models, transforms

        face_model = models.resnet18(weights=None)
        face_model.fc = nn.Linear(face_model.fc.in_features, 7)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        face_model = face_model.to(device).eval()

        mp_face  = mp.solutions.face_detection
        face_det = mp_face.FaceDetection(min_detection_confidence=0.5, model_selection=0)

        xform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_dummy = np.ones((480, 640, 3), dtype=np.uint8) * 120  # neutral BGR

        def run_face():
            img_rgb = cv2.cvtColor(img_dummy, cv2.COLOR_BGR2RGB)
            res = face_det.process(img_rgb)
            if res.detections:
                bb  = res.detections[0].location_data.relative_bounding_box
                h, w = img_dummy.shape[:2]
                x1 = max(0, int(bb.xmin * w))
                y1 = max(0, int(bb.ymin * h))
                x2 = min(w, int((bb.xmin + bb.width) * w))
                y2 = min(h, int((bb.ymin + bb.height) * h))
                crop = img_dummy[y1:y2, x1:x2]
            else:
                crop = img_dummy[:224, :224]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            pil  = Image.fromarray(gray, mode="L")
            t    = xform(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = face_model(t)

        pm = PowerMonitor(); pm.start()
        bm  = benchmark_runs(run_face, n=N_RUNS, warmup=N_WARMUP)
        pwr = pm.stop()
        face_det.close()
        _record(FACE_FULL, bm, pwr)

    except Exception as exc:
        print(f"  [WARN] MediaPipe pipeline failed ({exc}); falling back to ResNet-18 alone.")
        try:
            import torch.nn as nn
            from torchvision import models

            face_model2 = models.resnet18(weights=None)
            face_model2.fc = nn.Linear(face_model2.fc.in_features, 7)
            device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            face_model2 = face_model2.to(device2).eval()
            dummy_t = torch.randn(1, 3, 224, 224).to(device2)

            def run_face_fallback():
                with torch.no_grad():
                    _ = face_model2(dummy_t)

            pm = PowerMonitor(); pm.start()
            bm  = benchmark_runs(run_face_fallback, n=N_RUNS, warmup=N_WARMUP)
            pwr = pm.stop()
            print("  (label: face_resnet18_randweights_only — MediaPipe not available)")
            _record(FACE_FALLBACK, bm, pwr)

        except Exception as exc2:
            print(f"  [SKIP] face benchmark completely failed: {exc2}")

# ── Append new rows to CSV ───────────────────────────────────────────────────────

if new_rows:
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)
    print(f"\nAppended {len(new_rows)} row(s) to {CSV_PATH}.")
else:
    print(f"\nNo new rows — all modalities for mode {mode_id} already in CSV.")

# ── Print full table ─────────────────────────────────────────────────────────────

print("\n--- Full results/latency_power.csv ---")
with open(CSV_PATH) as f:
    print(f.read())
