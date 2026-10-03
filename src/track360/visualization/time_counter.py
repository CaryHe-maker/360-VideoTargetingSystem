"""Processing-only runtime measurement for tracking artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns


@dataclass(slots=True)
class TimeCounter:
    """Accumulate only the tracking-processing intervals for one run."""

    _startedAtNs: int | None = field(default=None, init=False, repr=False)
    _startedAtUtc: datetime | None = field(default=None, init=False, repr=False)
    _processingStartedAtNs: int | None = field(default=None, init=False, repr=False)
    _processingElapsedNs: int = field(default=0, init=False, repr=False)

    def start(self) -> None:
        """Start the artifact lifecycle without starting a processing interval."""
        if self._startedAtNs is not None:
            raise RuntimeError("time counter has already started")
        self._startedAtNs = perf_counter_ns()
        self._startedAtUtc = datetime.now(UTC)

    def startProcessing(self) -> None:
        """Start one interval that belongs to the remote tracking workload."""
        if self._startedAtNs is None:
            self.start()
        if self._processingStartedAtNs is not None:
            raise RuntimeError("time counter processing interval has already started")
        self._processingStartedAtNs = perf_counter_ns()

    def stopProcessing(self) -> None:
        """Stop the current tracking interval without measuring surrounding work."""
        if self._processingStartedAtNs is None:
            raise RuntimeError("time counter processing interval has not started")
        self._processingElapsedNs += max(0, perf_counter_ns() - self._processingStartedAtNs)
        self._processingStartedAtNs = None

    def stop(self, outputPath: str | Path) -> Path:
        """Stop measuring and atomically write the timing artifact."""
        if self._startedAtNs is None or self._startedAtUtc is None:
            raise RuntimeError("time counter has not started")

        if self._processingStartedAtNs is not None:
            self.stopProcessing()
        finishedAtUtc = datetime.now(UTC)
        elapsedNs = self._processingElapsedNs
        output = Path(outputPath)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "track360.time.v1",
            "scope": "tracking_processing",
            "elapsedNanoseconds": elapsedNs,
            "elapsedMilliseconds": elapsedNs / 1_000_000,
            "elapsedSeconds": elapsedNs / 1_000_000_000,
            "startedAtUtc": self._startedAtUtc.isoformat(),
            "finishedAtUtc": finishedAtUtc.isoformat(),
        }
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
        return output


__all__ = ["TimeCounter"]
