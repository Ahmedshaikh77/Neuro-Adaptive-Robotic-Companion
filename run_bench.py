"""
On-device latency + power benchmark for audio, gesture, and face modalities.
Runs under the current nvpmodel power mode (mode 0 / 15W).
sudo was not available to switch modes, so all rows are power_mode=0.
"""

import os, sys, time, csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.benchmark import benchmark_runs, summarize_latencies, PowerMonitor

N_RUNS   = 200
N_WARMUP = 10

os.makedirs("results", exist_ok=True)
rows = []

# ── CUDA warm-up (load driver once) ────────────────────────────────────────────
import torch
_ = torch.zeros(1).cuda()
print("CUDA driver loaded.")

# ── Audio (KWS CNN) ─────────────────────────────────────────────────────────────
print("\n[audio] benchmarking …")
from src.audio_kws import KwsModality, DEFAULT_KEYWORDS, log_mel

kws = KwsModality(checkpoint_path=None)   # random weights
kws.model.eval()

wav_dummy = np.zeros(16000, dtype=np.float32)   # 1 s at 16 kHz

def run_audio():
    kws.infer({"wav": wav_dummy})

pm = PowerMonitor(); pm.start()
bm_audio = benchmark_runs(run_audio, n=N_RUNS, warmup=N_WARMUP)
pwr_audio = pm.stop()
print(f"  audio p50={bm_audio['p50']:.2f}ms  p95={bm_audio['p95']:.2f}ms "
      f"  avg_w={pwr_audio['avg_w']:.2f}  peak_w={pwr_audio['peak_w']:.2f}")

rows.append(dict(
    modality="audio",
    power_mode=0,
    p50_ms=round(bm_audio["p50"], 3),
    p95_ms=round(bm_audio["p95"], 3),
    mean_ms=round(bm_audio["mean"], 3),
    avg_power_w=round(pwr_audio["avg_w"], 3),
    peak_power_w=round(pwr_audio["peak_w"], 3),
))

# ── Gesture (temporal CNN) ──────────────────────────────────────────────────────
print("\n[gesture] benchmarking …")
from src.gesture_engine import GestureModality

gest = GestureModality(checkpoint_path=None, n_frames=8, size=96)
gest.model.eval()

frames_dummy = np.zeros((8, 96, 96), dtype=np.float32)  # 8 grayscale 96×96 frames

def run_gesture():
    gest.infer({"frames": frames_dummy})

pm = PowerMonitor(); pm.start()
bm_gest = benchmark_runs(run_gesture, n=N_RUNS, warmup=N_WARMUP)
pwr_gest = pm.stop()
print(f"  gesture p50={bm_gest['p50']:.2f}ms  p95={bm_gest['p95']:.2f}ms "
      f"  avg_w={pwr_gest['avg_w']:.2f}  peak_w={pwr_gest['peak_w']:.2f}")

rows.append(dict(
    modality="gesture",
    power_mode=0,
    p50_ms=round(bm_gest["p50"], 3),
    p95_ms=round(bm_gest["p95"], 3),
    mean_ms=round(bm_gest["mean"], 3),
    avg_power_w=round(pwr_gest["avg_w"], 3),
    peak_power_w=round(pwr_gest["peak_w"], 3),
))

# ── Face (MediaPipe + ResNet-18 with random weights) ────────────────────────────
print("\n[face] benchmarking MediaPipe + ResNet-18 (random weights) …")
try:
    import cv2
    import mediapipe as mp
    from torchvision import transforms, models
    import torch.nn as nn
    from PIL import Image

    # Build ResNet-18 with 7 FER classes and random weights
    face_model = models.resnet18(weights=None)
    face_model.fc = nn.Linear(face_model.fc.in_features, 7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_model = face_model.to(device).eval()

    mp_face = mp.solutions.face_detection
    face_det = mp_face.FaceDetection(min_detection_confidence=0.5, model_selection=0)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Dummy 480×640 BGR image with a bright patch that may trigger face detection.
    # We do NOT require a face to be detected — we time the full pipeline regardless,
    # matching what real usage does (MediaPipe runs on every frame).
    img_dummy = np.ones((480, 640, 3), dtype=np.uint8) * 120

    def run_face():
        img_rgb = cv2.cvtColor(img_dummy, cv2.COLOR_BGR2RGB)
        res = face_det.process(img_rgb)
        # crop or use whole image if no face found
        if res.detections:
            bbox = res.detections[0].location_data.relative_bounding_box
            h, w = img_dummy.shape[:2]
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))
            crop = img_dummy[y1:y2, x1:x2]
        else:
            crop = img_dummy[:224, :224]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        pil  = Image.fromarray(gray, mode="L")
        t    = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = face_model(t)

    pm = PowerMonitor(); pm.start()
    bm_face = benchmark_runs(run_face, n=N_RUNS, warmup=N_WARMUP)
    pwr_face = pm.stop()
    print(f"  face p50={bm_face['p50']:.2f}ms  p95={bm_face['p95']:.2f}ms "
          f"  avg_w={pwr_face['avg_w']:.2f}  peak_w={pwr_face['peak_w']:.2f}")

    rows.append(dict(
        modality="face_mediapipe_resnet18_randweights",
        power_mode=0,
        p50_ms=round(bm_face["p50"], 3),
        p95_ms=round(bm_face["p95"], 3),
        mean_ms=round(bm_face["mean"], 3),
        avg_power_w=round(pwr_face["avg_w"], 3),
        peak_power_w=round(pwr_face["peak_w"], 3),
    ))

    face_det.close()

except Exception as e:
    print(f"  [SKIP] face benchmark failed: {e}")

# ── Write CSV ───────────────────────────────────────────────────────────────────
csv_path = "results/latency_power.csv"
fields = ["modality", "power_mode", "p50_ms", "p95_ms", "mean_ms", "avg_power_w", "peak_power_w"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"\nResults written to {csv_path}")
print("\n--- CSV contents ---")
with open(csv_path) as f:
    print(f.read())
