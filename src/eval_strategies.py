"""
Produce real result data from trained models. Three parts:

  (A) per-modality ACCURACY  -> already printed by train_fer / eval_fer,
      train_kws, and train_gesture at the end of training. Record those numbers.

  (B) per-modality LATENCY (p50/p95) and POWER on the Jetson -> RUNNABLE here,
      needs no dataset. Writes results/latency_power.csv. This is the cost
      profile used in the paper and the reliability/cost inputs to the gate.

  (C) STRATEGY comparison (single / all-on / adaptive), clean and fault-injected
      -> a harness that needs a shared "command" task. The three public datasets
      do not share a label space, so you build paired events from test samples
      that map to a common command vocabulary (CLASS_TO_COMMAND). Review this
      task definition with your advisor; then fill run_condition() with your
      paired samples (mirrors src/demo_adaptive.py, but with real inputs).

Run on the Jetson after training, e.g.:
  python -m src.eval_strategies --face artifacts/best_fer_resnet.pt \
      --kws artifacts/kws.pt --gesture artifacts/gesture.pt --bench-n 200
"""
from __future__ import annotations
import argparse, csv, os
from pathlib import Path
import numpy as np

from src.benchmark import benchmark_runs, PowerMonitor, format_pareto_table

# Face/emotion is intentionally NOT mapped here: emotion is not an intent, so the
# facial modality is excluded from the intent vote (see paper, Sec. IV).
CLASS_TO_COMMAND = {
    "audio": {"yes": "yes", "no": "no", "up": "up", "down": "down",
              "left": "left", "right": "right", "stop": "stop", "go": "go"},
    "gesture": {"thumb_up": "yes", "thumb_down": "no", "swipe_up": "up",
                "swipe_down": "down", "swipe_left": "left", "swipe_right": "right",
                "stop": "stop"},
}
CONDITIONS = ["clean", "low_light", "audio_noise", "occlusion", "dropout"]


def load_modalities(args):
    mods = {}
    if args.face:
        from src.emotion_modality import EmotionModality
        mods["face"] = EmotionModality(args.face)
    if args.kws:
        from src.audio_kws import KwsModality
        mods["audio"] = KwsModality(args.kws)
    if args.gesture:
        from src.gesture_engine import GestureModality
        mods["gesture"] = GestureModality(args.gesture)
    if not mods:
        raise SystemExit("provide at least one of --face --kws --gesture")
    return mods


def dummy_input(name):
    """A correctly-shaped input so the model actually runs for cost measurement.
    For face, prefer passing a real face image to include the full crop+CNN path."""
    if name == "face":
        return (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    if name == "audio":
        return {"wav": (np.random.randn(16000)).astype(np.float32)}
    if name == "gesture":
        return {"frames": (np.random.rand(8, 96, 96) * 255).astype(np.uint8)}
    raise ValueError(name)


def part_b_latency_power(mods, n, warmup, out):
    """RUNNABLE: measure p50/p95 latency and power for each modality on device."""
    rows = []
    for name, m in mods.items():
        sample = dummy_input(name)
        pm = PowerMonitor().start()
        s = benchmark_runs(lambda: m.infer(sample), n=n, warmup=warmup)
        p = pm.stop()
        rows.append({"modality": name, "p50_ms": round(s["p50"], 2),
                     "p95_ms": round(s["p95"], 2), "mean_ms": round(s["mean"], 2),
                     "avg_w": round(p["avg_w"], 3), "peak_w": round(p["peak_w"], 3),
                     "power_available": p["available"], "n": s["n"]})
        print(f"  {name:8s}  p50 {s['p50']:.1f} ms  p95 {s['p95']:.1f} ms  "
              f"power {p['avg_w']:.2f} W ({'tegrastats' if p['available'] else 'no power sensor'})")
    path = Path(out) / "latency_power.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("  wrote", path)
    return rows


def run_condition(gate, events, strategies=None):
    """Run single / all-on / adaptive over a list of paired multimodal events and
    return per-strategy summaries.

    Each event is a dict:
        {"truth": <intent label>,
         "frame": {<modality_name>: <input for that modality>, ...}}

    The modalities on `gate` must emit labels in the SHARED INTENT space (wrap the
    real KWS/gesture modalities so their infer() returns an intent label via
    CLASS_TO_COMMAND, or use SimulatedModality for a protocol dry-run). This is a
    complete evaluator: give it real paired data and it produces the real
    single/all-on/adaptive accuracy, latency, and energy.
    """
    from src.benchmark import StrategyResult
    names = [m.name for m in gate.modalities]
    results = {f"single:{n}": StrategyResult(f"single:{n}") for n in names}
    results["all_on"] = StrategyResult("all_on")
    results["adaptive"] = StrategyResult("adaptive")

    for ev in events:
        truth, frame = ev["truth"], ev["frame"]
        for n in names:
            d = gate.run_single(frame, n)
            results[f"single:{n}"].add(d.final_label, truth, d.total_latency_ms, d.total_energy_j)
        d_all = gate.run_all(frame)
        results["all_on"].add(d_all.final_label, truth, d_all.total_latency_ms, d_all.total_energy_j)
        d_ad = gate.decide_and_run(frame)
        results["adaptive"].add(d_ad.final_label, truth, d_ad.total_latency_ms, d_ad.total_energy_j)

    return {k: v.summary() for k, v in results.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face"); ap.add_argument("--kws"); ap.add_argument("--gesture")
    ap.add_argument("--bench-n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    mods = load_modalities(args)
    print("loaded modalities:", list(mods))

    print("\n[B] Per-modality latency and power on this device:")
    part_b_latency_power(mods, args.bench_n, args.warmup, args.out)

    print("\n[A] Per-modality accuracy: read it from the training/eval scripts")
    print("    (eval_fer.py for face, and the final val/test line of train_kws / train_gesture).")

    print("\n[C] Strategy comparison: define the shared command task with your advisor,")
    print("    then implement run_condition() using your paired test events. The gate,")
    print("    fusion, benchmark, and fault-injection code are all in place to support it.")


if __name__ == "__main__":
    main()
