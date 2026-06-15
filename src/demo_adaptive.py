"""
Runnable demo of the adaptive "smart switch" — NO camera, microphone, or trained
model required. It simulates three modalities (face / audio / gesture) with
realistic accuracy/latency/power profiles and compares the three strategies from
the research plan:

    single (each modality alone)  vs  all_on (static fusion)  vs  adaptive

It prints a Pareto table (accuracy vs latency vs power), then repeats under a
degraded condition (dim light) to show graceful degradation.

Run:  python -m src.demo_adaptive
"""

from __future__ import annotations

import numpy as np

from src.modality_base import SimulatedModality
from src.adaptive_gate import AdaptiveGate
from src.benchmark import StrategyResult, format_pareto_table
from src.fault_injection import quality_for

CLASSES = ["yes", "no", "stop", "go", "hello", "bye"]


def build_modalities(seed: int = 0):
    """Three sensors with deliberately different accuracy/speed/power trade-offs."""
    return [
        SimulatedModality("face",    CLASSES, accuracy=0.66, latency_ms=18, power_w=2.0, seed=seed + 1),
        SimulatedModality("audio",   CLASSES, accuracy=0.90, latency_ms=35, power_w=1.2, seed=seed + 2),
        SimulatedModality("gesture", CLASSES, accuracy=0.85, latency_ms=45, power_w=2.5, seed=seed + 3),
    ]


def make_stream(n: int, condition: str, seed: int = 7):
    """A stream of n events; each carries a per-sensor frame with its own quality."""
    rng = np.random.default_rng(seed)
    stream = []
    for _ in range(n):
        truth = str(rng.choice(CLASSES))
        stream.append({
            "truth": truth,
            "frame": {
                "face":    {"true_label": truth, "quality": quality_for(condition, "face")},
                "audio":   {"true_label": truth, "quality": quality_for(condition, "audio")},
                "gesture": {"true_label": truth, "quality": quality_for(condition, "gesture")},
            },
        })
    return stream


def run_condition(condition: str, n: int = 400):
    mods = build_modalities()
    gate = AdaptiveGate(mods, evidence_threshold=0.95, latency_budget_ms=120, power_budget_w=3.0)

    results = {k: StrategyResult(k) for k in
               ("single:face", "single:audio", "single:gesture", "all_on", "adaptive")}

    for ev in make_stream(n, condition):
        truth, frame = ev["truth"], ev["frame"]
        for name in ("face", "audio", "gesture"):
            d = gate.run_single(frame, name)
            results[f"single:{name}"].add(d.final_label, truth, d.total_latency_ms, d.total_energy_j)
        d_all = gate.run_all(frame)
        results["all_on"].add(d_all.final_label, truth, d_all.total_latency_ms, d_all.total_energy_j)
        d_ad = gate.decide_and_run(frame)
        results["adaptive"].add(d_ad.final_label, truth, d_ad.total_latency_ms, d_ad.total_energy_j)

    return [results[k].summary() for k in results]


def main():
    print("=" * 72)
    print("NeuroBot — Adaptive Smart-Switch Demo (simulated, no hardware needed)")
    print("=" * 72)
    for condition in ("clean", "low_light"):
        print(f"\nCondition: {condition.upper()}")
        print(format_pareto_table(run_condition(condition)))
    print("\nReading: 'adaptive' should land near the best accuracy while using")
    print("less time/power than 'all_on'. Under low_light the face sensor degrades,")
    print("so single:face collapses while adaptive leans on audio/gesture.")
    print("=" * 72)


if __name__ == "__main__":
    main()
