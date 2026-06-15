"""
Measurement harness for the paper: per-stage latency (p50 / p95, not just mean)
and on-device power via the Jetson `tegrastats` tool.

Everything degrades gracefully off-Jetson: if `tegrastats` is missing, the power
monitor reports `available=False` and zero power instead of crashing, so the same
code runs on a laptop and on the Orin Nano.

Pure standard library + NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import shutil
import subprocess
import threading
import re
import time
import statistics


def summarize_latencies(samples_ms: List[float]) -> Dict[str, float]:
    """Return p50, p95, mean, min, max and count for a list of latencies (ms)."""
    if not samples_ms:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(samples_ms)
    def pct(p):
        if len(s) == 1:
            return s[0]
        k = (len(s) - 1) * (p / 100.0)
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)
    return {
        "n": len(s),
        "p50": pct(50),
        "p95": pct(95),
        "mean": statistics.fmean(s),
        "min": s[0],
        "max": s[-1],
    }


def benchmark_runs(fn: Callable[[], object], n: int = 100, warmup: int = 5) -> Dict[str, float]:
    """
    Call `fn` n times, measuring wall-clock latency of each call.
    The first `warmup` calls are discarded (model load / cache effects).
    """
    for _ in range(max(0, warmup)):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return summarize_latencies(samples)


# Patterns that match a power field in tegrastats output across JetPack versions,
# e.g. "VDD_IN 4123mW/4500mW" or "POM_5V_IN 3200/3500".
_POWER_PATTERNS = [
    re.compile(r"VDD_IN\s+(\d+)mW"),
    re.compile(r"POM_5V_IN\s+(\d+)mW"),
    re.compile(r"VDD_IN\s+(\d+)/\d+"),
    re.compile(r"POM_5V_IN\s+(\d+)/\d+"),
]


def _parse_power_mw(line: str) -> Optional[float]:
    for pat in _POWER_PATTERNS:
        m = pat.search(line)
        if m:
            return float(m.group(1))
    return None


class PowerMonitor:
    """
    Samples board power via `tegrastats` in a background thread.

    Usage:
        pm = PowerMonitor()
        pm.start()
        ... run workload ...
        stats = pm.stop()   # {"available": bool, "avg_w": float, "peak_w": float, "n": int}
    """

    def __init__(self, interval_ms: int = 100, tegrastats_path: Optional[str] = None):
        self.interval_ms = interval_ms
        self.path = tegrastats_path or shutil.which("tegrastats")
        self.available = self.path is not None
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._samples_mw: List[float] = []
        self._stop = threading.Event()

    def _reader(self):
        try:
            self._proc = subprocess.Popen(
                [self.path, "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except Exception:
            self.available = False
            return
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            mw = _parse_power_mw(line)
            if mw is not None:
                self._samples_mw.append(mw)

    def start(self) -> "PowerMonitor":
        if not self.available:
            return self
        self._stop.clear()
        self._samples_mw = []
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Dict[str, float]:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if not self.available or not self._samples_mw:
            return {"available": self.available, "avg_w": 0.0, "peak_w": 0.0, "n": 0}
        avg_w = statistics.fmean(self._samples_mw) / 1000.0
        peak_w = max(self._samples_mw) / 1000.0
        return {"available": True, "avg_w": avg_w, "peak_w": peak_w, "n": len(self._samples_mw)}


@dataclass
class StrategyResult:
    """Accumulated metrics for one strategy over a stream of inputs."""
    name: str
    n: int = 0
    correct: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    energies_j: List[float] = field(default_factory=list)
    powers_w: List[float] = field(default_factory=list)

    def add(self, predicted, truth, latency_ms, energy_j):
        self.n += 1
        if truth is not None and predicted == truth:
            self.correct += 1
        self.latencies_ms.append(latency_ms)
        self.energies_j.append(energy_j)
        if latency_ms > 0:
            self.powers_w.append(energy_j / (latency_ms / 1000.0))

    def summary(self) -> Dict[str, float]:
        lat = summarize_latencies(self.latencies_ms)
        return {
            "strategy": self.name,
            "accuracy": (self.correct / self.n) if self.n else 0.0,
            "p50_ms": lat["p50"],
            "p95_ms": lat["p95"],
            "avg_energy_mj": (statistics.fmean(self.energies_j) * 1000.0) if self.energies_j else 0.0,
            "avg_power_w": statistics.fmean(self.powers_w) if self.powers_w else 0.0,
            "n": self.n,
        }


def format_pareto_table(summaries: List[Dict[str, float]]) -> str:
    """Pretty ASCII table comparing strategies on accuracy / latency / power."""
    head = f"{'strategy':<18}{'acc':>7}{'p50 ms':>9}{'p95 ms':>9}{'energy mJ':>11}{'power W':>9}"
    lines = [head, "-" * len(head)]
    for s in summaries:
        lines.append(
            f"{s['strategy']:<18}{s['accuracy']*100:>6.1f}%"
            f"{s['p50_ms']:>9.1f}{s['p95_ms']:>9.1f}{s['avg_energy_mj']:>11.1f}{s['avg_power_w']:>9.2f}"
        )
    return "\n".join(lines)
