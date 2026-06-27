# NeuroBot RUNBOOK: getting the real result data

This is the step-by-step to replace the simulated paper tables with measured
numbers. Training runs on any GPU (your laptop or free Google Colab). The
latency and power numbers must come from the Jetson Orin Nano.

None of this needs an IRB: it uses only public datasets and your own device.

## 0. Setup (training machine)
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    python selftest.py        # sanity check, 18/18 should pass

## 1. Train the three modalities

### Face (FER-2013)
Download FER-2013 from Kaggle and arrange as data/archive/{train,test}/<emotion>/*.jpg
    python -m src.train_fer  --data-root data/archive --epochs 30
    python -m src.eval_fer   --data-root data/archive --checkpoint artifacts/best_fer_resnet.pt
Record the test accuracy and confusion matrix. -> face accuracy row.

### Audio keyword spotting (Speech Commands)
Dataset auto-downloads via torchaudio (~2.3 GB first run).
    python -m src.train_kws --epochs 20 --out artifacts/kws.pt
Record the best val accuracy printed at the end. -> audio accuracy row.

### Gesture (Jester)
Obtain the Jester dataset (note: it has required registration and moved hosts;
if blocked, use any folder-structured gesture set or a subset). Arrange as
data_gesture/{train,test}/<gesture>/<sample_id>/frame_*.jpg
    python -m src.train_gesture --data-root data_gesture --epochs 15 --out artifacts/gesture.pt
Record the best test accuracy. -> gesture accuracy row.

(You can ship a face + audio version first and add gesture later.)

## 2. Set up the Jetson Orin Nano
- Flash JetPack (6.x) and boot.
- Install torch / torchvision / torchaudio from the NVIDIA Jetson wheels,
  NOT from PyPI. See NVIDIA's "PyTorch for Jetson" page for the exact wheels.
- pip install the remaining requirements (mediapipe, opencv-python, etc.).
- Copy this repo and the artifacts/*.pt checkpoints onto the device.

## 3. Measure on the Jetson

### Latency and power (no dataset needed) -> RUNNABLE
    python -m src.eval_strategies --face artifacts/best_fer_resnet.pt \
        --kws artifacts/kws.pt --gesture artifacts/gesture.pt --bench-n 200
Writes results/latency_power.csv with p50/p95 latency and tegrastats power per
modality. These are real on-device numbers for the cost columns of Table 1.
Run it under each power mode (sudo nvpmodel) to report across modes.

### Strategy comparison (Tables 1 and 2)
The three datasets do not share a label space, so the gating comparison uses a
shared "command" task. Review CLASS_TO_COMMAND and the task definition in
src/eval_strategies.py with your advisor, then implement run_condition() with
paired test events (mirrors src/demo_adaptive.py with real inputs). It outputs
accuracy / p50 / p95 / energy / power per strategy, in clean and degraded
(low light, audio noise, occlusion, dropout) conditions.

## 4. Put numbers in the paper
Send the CSVs from results/ and the three accuracy numbers. The illustrative
tables in the paper get replaced with the measured ones.

## Quick reference: what is already built
- Smart switch / fusion:        src/adaptive_gate.py
- Latency + tegrastats power:    src/benchmark.py
- Fault injection:               src/fault_injection.py
- Modality wrappers:             src/emotion_modality.py, audio_kws.py, gesture_engine.py
- Hardware-free demo + tests:    src/demo_adaptive.py, selftest.py
