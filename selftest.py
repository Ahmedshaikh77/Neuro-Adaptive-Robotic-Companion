"""
Self-test for NeuroBot — runs with NO camera, microphone, GPU, or trained model.

Verifies the parts that don't need hardware:
  * all modules import cleanly (torch-based ones import lazily)
  * the simulated modalities, smart switch, fusion, benchmark, and fault
    injection all behave correctly
  * the adaptive gate beats a single weak sensor and costs less than all-on

Run:  python selftest.py
"""

import sys
import numpy as np

PASS, FAIL = "PASS", "FAIL"
results = []

def check(name, cond):
    results.append((name, bool(cond)))
    print(f"  [{PASS if cond else FAIL}] {name}")
    return cond


def main():
    print("NeuroBot self-test")
    print("=" * 60)

    # 1) pure modules import
    print("\n[1] Imports (pure / numpy)")
    from src.modality_base import SimulatedModality, ModalityResult, Modality
    from src.adaptive_gate import AdaptiveGate, fuse
    from src.benchmark import summarize_latencies, benchmark_runs, PowerMonitor, StrategyResult, format_pareto_table
    from src import fault_injection as fi
    check("import core modules", True)

    # 2) torch-based modules import lazily (no torch needed just to import)
    print("\n[2] Imports (lazy torch modules)")
    import importlib
    for m in ("src.audio_kws", "src.gesture_engine", "src.emotion_modality"):
        importlib.import_module(m)
    check("import model modules without torch installed", True)

    # 3) simulated modality
    print("\n[3] Simulated modality")
    face = SimulatedModality("face", ["a", "b", "c"], accuracy=1.0, latency_ms=10, power_w=2.0, seed=1)
    r = face.infer({"true_label": "a", "quality": 1.0})
    check("perfect modality predicts truth", r.label == "a")
    check("latency > 0", r.latency_ms > 0)
    check("energy > 0", r.energy_j > 0)
    check("reliability carried", abs(r.reliability - 1.0) < 1e-9)

    # 4) fusion math
    print("\n[4] Fusion")
    a = ModalityResult("audio", "yes", 0.9, 10, reliability=0.9)
    b = ModalityResult("face", "no", 0.4, 10, reliability=0.66)
    label, evid, post = fuse([a, b])
    check("reliable+confident sensor wins fusion", label == "yes")
    check("posterior in [0,1]", 0.0 <= post <= 1.0)

    # 5) benchmark percentiles
    print("\n[5] Benchmark stats")
    s = summarize_latencies([10, 20, 30, 40, 50])
    check("p50 correct", abs(s["p50"] - 30) < 1e-6)
    check("p95 between max-ish", 40 < s["p95"] <= 50)
    bm = benchmark_runs(lambda: sum(range(1000)), n=20, warmup=2)
    check("benchmark_runs returns n samples", bm["n"] == 20)
    pm = PowerMonitor().start().stop()
    check("power monitor graceful off-Jetson", pm["available"] in (True, False) and pm["avg_w"] >= 0)

    # 6) fault injection
    print("\n[6] Fault injection")
    img = (np.ones((32, 32, 3)) * 200).astype(np.uint8)
    check("low_light darkens image", fi.low_light(img, 0.3).mean() < img.mean())
    check("occlusion zeros some pixels", (fi.occlude(img, 0.5, seed=0) == 0).any())
    check("low_light hurts vision not audio",
          fi.quality_for("low_light", "face") < 1.0 and fi.quality_for("low_light", "audio") == 1.0)

    # 7) end-to-end: adaptive vs single vs all_on
    print("\n[7] Smart switch behaviour")
    mods = [
        SimulatedModality("face", ["x", "y", "z", "w"], 0.66, 18, 2.0, seed=11),
        SimulatedModality("audio", ["x", "y", "z", "w"], 0.92, 35, 1.2, seed=12),
        SimulatedModality("gesture", ["x", "y", "z", "w"], 0.85, 45, 2.5, seed=13),
    ]
    gate = AdaptiveGate(mods, evidence_threshold=0.95, latency_budget_ms=120, power_budget_w=3.0)
    rng = np.random.default_rng(0)
    n = 500
    acc = {k: 0 for k in ("face", "all_on", "adaptive")}
    lat = {k: 0.0 for k in ("all_on", "adaptive")}
    for _ in range(n):
        truth = str(rng.choice(["x", "y", "z", "w"]))
        frame = {m.name: {"true_label": truth, "quality": 1.0} for m in mods}
        if gate.run_single(frame, "face").final_label == truth:
            acc["face"] += 1
        d_all = gate.run_all(frame); acc["all_on"] += d_all.final_label == truth; lat["all_on"] += d_all.total_latency_ms
        d_ad = gate.decide_and_run(frame); acc["adaptive"] += d_ad.final_label == truth; lat["adaptive"] += d_ad.total_latency_ms
    a_face, a_all, a_ad = acc["face"]/n, acc["all_on"]/n, acc["adaptive"]/n
    check(f"adaptive ({a_ad:.0%}) much better than single face ({a_face:.0%})", a_ad > a_face + 0.15)
    check(f"adaptive ({a_ad:.0%}) close to all_on ({a_all:.0%})", a_ad >= a_all - 0.05)
    check(f"adaptive cheaper than all_on ({lat['adaptive']:.0f} < {lat['all_on']:.0f} ms total)",
          lat["adaptive"] < lat["all_on"])

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok in results if ok)
    print(f"{n_pass}/{len(results)} checks passed")
    print("=" * 60)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
