"""Runtime profiler."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter_ns

import numpy as np


@dataclass(slots=True)
class TimingStat:
    count: int = 0
    totalNs: int = 0
    minNs: int | None = None
    maxNs: int | None = None
    samplesNs: list[int] = field(default_factory=list)

    @property
    def meanNs(self) -> float:
        return float(self.totalNs / self.count) if self.count else 0.0


@dataclass(slots=True)
class RuntimeProfiler:
    """Low-overhead stage profiler.

    ``enabled=False`` makes ``track`` a no-op context manager and avoids all
    timestamp calls. Samples are retained so evaluation artifacts can report
    percentiles rather than relying on a mean that hides tail latency.
    """

    enabled: bool = True
    stats: dict[str, TimingStat] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    frameRows: list[dict[str, object]] = field(default_factory=list)
    _frame: dict[str, object] | None = field(default=None, init=False, repr=False)

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = perf_counter_ns()
        try:
            yield
        finally:
            elapsed = perf_counter_ns() - start
            self.record(name, elapsed)

    def startFrame(self, frameIndex: int, **metadata: object) -> None:
        if not self.enabled:
            return
        if self._frame is not None:
            raise RuntimeError("profiler frame already active")
        self._frame = {"frameIndex": int(frameIndex), "stages": {}, **metadata}

    def finishFrame(self) -> None:
        if not self.enabled:
            return
        if self._frame is None:
            raise RuntimeError("profiler frame is not active")
        row = dict(self._frame)
        stages = row.get("stages", {})
        row["stages"] = {name: float(sum(values)) for name, values in stages.items()}
        self.frameRows.append(row)
        self._frame = None

    def annotateFrame(self, **metadata: object) -> None:
        if not self.enabled:
            return
        if self._frame is None:
            raise RuntimeError("profiler frame is not active")
        self._frame.update(metadata)

    def appendFrameMetadata(self, name: str, value: object) -> None:
        if not self.enabled:
            return
        if self._frame is None:
            raise RuntimeError("profiler frame is not active")
        values = self._frame.setdefault(name, [])
        if not isinstance(values, list):
            raise RuntimeError(f"profiler frame metadata is not appendable: {name}")
        values.append(value)

    def record(self, name: str, elapsedNs: int) -> None:
        if not self.enabled:
            return
        stat = self.stats.setdefault(name, TimingStat())
        stat.count += 1
        stat.totalNs += int(elapsedNs)
        stat.minNs = int(elapsedNs) if stat.minNs is None else min(stat.minNs, int(elapsedNs))
        stat.maxNs = int(elapsedNs) if stat.maxNs is None else max(stat.maxNs, int(elapsedNs))
        stat.samplesNs.append(int(elapsedNs))
        if self._frame is not None:
            stages = self._frame.setdefault("stages", {})
            stages.setdefault(name, []).append(int(elapsedNs) / 1_000_000.0)

    def summarize(self) -> dict[str, dict[str, float]]:
        if not self.enabled:
            return {}
        return {
            name: {
                "count": float(stat.count),
                "meanNs": stat.meanNs,
                "minNs": float(stat.minNs or 0),
                "maxNs": float(stat.maxNs or 0),
                "p50Ns": _percentile(stat.samplesNs, 50),
                "p95Ns": _percentile(stat.samplesNs, 95),
                "p99Ns": _percentile(stat.samplesNs, 99),
            }
            for name, stat in self.stats.items()
        }

    def summarizeFrames(self) -> dict[str, dict[str, float]]:
        """Summarize per-frame stage totals rather than individual stage calls."""
        if not self.enabled:
            return {}
        stageNames = sorted(
            {
                name
                for row in self.frameRows
                for name in row.get("stages", {})
            }
        )
        result: dict[str, dict[str, float]] = {}
        for name in stageNames:
            valuesNs = [
                float(row.get("stages", {}).get(name, 0.0)) * 1_000_000.0
                for row in self.frameRows
            ]
            result[name] = {
                "count": float(len(valuesNs)),
                "meanNs": float(np.mean(valuesNs)) if valuesNs else 0.0,
                "p50Ns": _percentile(valuesNs, 50),
                "p95Ns": _percentile(valuesNs, 95),
                "p99Ns": _percentile(valuesNs, 99),
            }
        return result


def _percentile(values: list[int] | list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def measure(func, *args, **kwargs):
    start = perf_counter_ns()
    value = func(*args, **kwargs)
    return value, perf_counter_ns() - start


__all__ = ["RuntimeProfiler", "TimingStat", "measure"]
