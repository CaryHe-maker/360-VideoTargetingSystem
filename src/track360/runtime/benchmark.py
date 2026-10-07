"""Run tracking methods over a 360VOT dataset and score the result files.

Output layout under one root, ready for the official toolkit as well::

    <root>/bbox/<method>/<sequence>.txt      <root>/reports/<method>/run.json
    <root>/bfov/<method>/<sequence>.txt      <root>/reports/<method>/<sequence>.json
"""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from track360.backends import ARTrackSession, createArtrackSession
from track360.core.config import AppConfig, ModelConfig
from track360.core.errors import DecodeError, ProtocolError
from track360.core.protocols import SphericalGeometry
from track360.core.types import BBoxXYWH, FramePacket, ResultSource, TrackResult, TrackStatus
from track360.datasets.vot360 import Vot360Dataset, Vot360DataSource
from track360.evaluation.vot360_metrics import Vot360Scores, evaluateVot360, loadTrackerResults
from track360.geometry import SphericalGeometryImpl
from track360.io.vot360_results import (
    BBOX_DIRECTORY,
    BFOV_DIRECTORY,
    ResultCollector,
    resultPaths,
    writeSequenceResults,
)
from track360.runtime.driver import buildRuntime, closeRuntime, runTracking
from track360.runtime.reproducibility import collectRunMetadata, seedEverything

LOGGER = logging.getLogger("track360.benchmark")
REPORT_DIRECTORY = "reports"

METHODS: dict[str, str] = {
    "ours": "Track360: one perspective view that follows the predicted BFoV",
    "b0": "ARTrackV2 tracking directly on the ERP frame, no spherical handling",
}

SessionFactory = Callable[[ModelConfig], ARTrackSession]


@dataclass(frozen=True, slots=True)
class SequenceReport:
    sequence: str
    status: str  # done | skipped | failed
    frames: int = 0
    seconds: float = 0.0
    fps: float = 0.0
    p50LatencyMs: float = 0.0
    p95LatencyMs: float = 0.0
    invalidFrames: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    method: str
    reports: tuple[SequenceReport, ...]

    @property
    def failures(self) -> tuple[SequenceReport, ...]:
        return tuple(report for report in self.reports if report.status == "failed")


@dataclass(slots=True)
class _TimedCollector(ResultCollector):
    """Collect results and the wall-clock time each one was committed."""

    writeTimes: list[float] = field(default_factory=list)

    def write(self, result: TrackResult) -> None:
        ResultCollector.write(self, result)
        self.writeTimes.append(perf_counter())


class _SharedSession:
    """Keep one loaded model across sequences: each runtime closes its backend."""

    def __init__(self, session: ARTrackSession) -> None:
        self._session = session

    # The protocol members are spelled out because runtime protocol checks do not
    # look through ``__getattr__``; optional batch methods are forwarded by it.
    @property
    def supportsOnlineTemplates(self) -> bool:
        return bool(self._session.supportsOnlineTemplates)

    def encodeTemplate(self, rgb: Any, bbox: BBoxXYWH) -> Any:
        return self._session.encodeTemplate(rgb, bbox)

    def infer(self, rgb: Any, templateFeatures: Sequence[object]) -> Any:
        return self._session.infer(rgb, templateFeatures)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def close(self) -> None:
        return None


def runBenchmark(
    *,
    datasetRoot: str | Path,
    outputRoot: str | Path,
    method: str,
    config: AppConfig,
    labelRoot: str | Path | None = None,
    sequences: Sequence[str] | None = None,
    maxFrames: int | None = None,
    resume: bool = True,
    shard: tuple[int, int] = (0, 1),
    sessionFactory: SessionFactory | None = None,
) -> BenchmarkSummary:
    """Track every selected sequence with one method, one failure never stopping the rest.

    ``shard=(i, n)`` keeps every n-th sequence starting at i, so several processes can
    split a run.  With ``resume`` a sequence whose result files are already complete is
    skipped.  ``labelRoot`` supplies labels for sequences shipped without them, such as
    the tune set.
    """
    if method not in METHODS:
        raise ProtocolError(f"unknown method '{method}'; available: {', '.join(METHODS)}")
    dataset = Vot360Dataset(datasetRoot, labelRoot)
    names = _selectSequences(dataset, sequences, shard)
    output = Path(outputRoot)
    reportRoot = output / REPORT_DIRECTORY / method
    reportRoot.mkdir(parents=True, exist_ok=True)
    seedEverything(config.reproducibility)
    _writeJson(
        reportRoot / "run.json",
        {
            "method": method,
            "description": METHODS[method],
            "datasetRoot": str(dataset.root),
            "labelRoot": None if dataset.labelRoot is None else str(dataset.labelRoot),
            "sequences": list(names),
            "maxFrames": maxFrames,
            "shard": list(shard),
            **collectRunMetadata(config),
        },
    )
    session = (
        sessionFactory(config.model)
        if sessionFactory is not None
        else createArtrackSession(config)
    )
    reports: list[SequenceReport] = []
    try:
        for position, name in enumerate(names, start=1):
            report = _runOne(dataset, output, method, config, name, session, maxFrames, resume)
            reports.append(report)
            _writeJson(reportRoot / f"{name}.json", asdict(report))
            LOGGER.info(
                "[%d/%d] %s %s: frames=%d fps=%.2f p50=%.0fms p95=%.0fms invalid=%d%s",
                position,
                len(names),
                method,
                name,
                report.frames,
                report.fps,
                report.p50LatencyMs,
                report.p95LatencyMs,
                report.invalidFrames,
                f" {report.status}: {report.error}" if report.error else f" {report.status}",
            )
    finally:
        session.close()
    summary = BenchmarkSummary(method=method, reports=tuple(reports))
    if summary.failures:
        LOGGER.error(
            "%d sequence(s) failed: %s",
            len(summary.failures),
            ", ".join(report.sequence for report in summary.failures),
        )
    return summary


def evaluateResults(
    *,
    datasetRoot: str | Path,
    outputRoot: str | Path,
    labelRoot: str | Path | None = None,
    methods: Sequence[str] | None = None,
    allowPartial: bool = False,
    only: Sequence[str] | None = None,
) -> dict[str, dict[str, Vot360Scores]]:
    """Score every method under ``outputRoot`` in both result representations.

    Each method is scored on the sequences it has result files for.  A result file
    shorter than its sequence is an error unless ``allowPartial`` is set, which
    truncates the ground truth to match and is only meant for smoke runs.  ``only``
    restricts scoring to the named sequences, for example those with one attribute; a
    method with no result for any of them is left out.
    """
    dataset = Vot360Dataset(datasetRoot, labelRoot)
    output = Path(outputRoot)
    scores: dict[str, dict[str, Vot360Scores]] = {}
    for representation in (BBOX_DIRECTORY, BFOV_DIRECTORY):
        root = output / representation
        if not root.is_dir():
            continue
        available = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
        for method in methods if methods is not None else available:
            if method not in available:
                raise DecodeError(f"no {representation} results for method '{method}' in {root}")
            results = loadTrackerResults(root / method)
            if only is not None:
                results = {name: rows for name, rows in results.items() if name in only}
                if not results:
                    continue
            groundTruth = {}
            for name, rows in results.items():
                sequence = dataset.sequence(name)
                truth, present = sequence.groundTruth(representation)
                sequence.close()
                if allowPartial and len(rows) < len(truth):
                    truth, present = truth[: len(rows)], present[: len(rows)]
                groundTruth[name] = (truth, present)
            frameWidthPx, frameHeightPx = dataset.sequence(next(iter(results))).frameSize
            scores.setdefault(method, {})[representation] = evaluateVot360(
                groundTruth,
                results,
                representation,
                frameWidthPx=frameWidthPx,
                frameHeightPx=frameHeightPx,
            )
    if not scores:
        raise DecodeError(f"no result directories under {output}")
    return scores


def _selectSequences(
    dataset: Vot360Dataset, sequences: Sequence[str] | None, shard: tuple[int, int]
) -> tuple[str, ...]:
    index, count = shard
    if count <= 0 or not 0 <= index < count:
        raise ProtocolError(f"invalid shard {index}/{count}")
    names = dataset.sequenceNames if sequences is None else tuple(sequences)
    unknown = [name for name in names if name not in dataset.sequenceNames]
    if unknown:
        raise DecodeError(f"unknown 360VOT sequences: {', '.join(unknown)}")
    return tuple(names[index::count])


def _runOne(
    dataset: Vot360Dataset,
    outputRoot: Path,
    method: str,
    config: AppConfig,
    name: str,
    session: ARTrackSession,
    maxFrames: int | None,
    resume: bool,
) -> SequenceReport:
    source = Vot360DataSource(maxFrames=maxFrames, labelRoot=dataset.labelRoot)
    try:
        source.open(str(dataset.root), name)
        frameCount = source.frameCount
        if resume and _isComplete(outputRoot, method, name, frameCount):
            return SequenceReport(sequence=name, status="skipped", frames=frameCount)
        collector = _TimedCollector()
        started = perf_counter()
        if method == "b0":
            _trackErpDirect(source, session, config, collector)
        else:
            _trackSpherical(source, session, config, collector)
        seconds = perf_counter() - started
        collector.finalize(frameCount)
        frameWidthPx = source.sequence.frameSize[0]
        writeSequenceResults(outputRoot, method, name, collector.results, frameWidthPx)
        latencies = np.diff(collector.writeTimes) * 1000.0
        return SequenceReport(
            sequence=name,
            status="done",
            frames=frameCount,
            seconds=seconds,
            fps=frameCount / seconds if seconds > 0.0 else 0.0,
            p50LatencyMs=float(np.percentile(latencies, 50)) if latencies.size else 0.0,
            p95LatencyMs=float(np.percentile(latencies, 95)) if latencies.size else 0.0,
            invalidFrames=sum(1 for result in collector.results if not result.valid),
        )
    except Exception as error:
        LOGGER.exception("sequence %s failed", name)
        return SequenceReport(
            sequence=name, status="failed", error=f"{type(error).__name__}: {error}"
        )
    finally:
        source.close()


def _trackSpherical(
    source: Vot360DataSource,
    session: ARTrackSession,
    config: AppConfig,
    collector: ResultCollector,
) -> None:
    shared = _SharedSession(session)
    runtime = buildRuntime(config, artrackSessionFactory=lambda _: shared)
    try:
        runTracking(
            source=source,
            initialBfov=source.sequence.initialBfov(),
            geometry=runtime.geometry,
            controller=runtime.controller,
            backend=runtime.backend,
            sink=collector,
            scoreCalibration=runtime.scoreCalibration,
            useMotionScore=runtime.useMotionScore,
        )
    finally:
        closeRuntime(runtime)


def _trackErpDirect(
    source: Vot360DataSource,
    session: ARTrackSession,
    config: AppConfig,
    collector: ResultCollector,
) -> None:
    """Baseline B0: the plain tracker loop on full ERP frames.

    The search region follows the previous box in image coordinates.  Nothing handles
    the seam or the distortion, which is the point of the baseline.  A sequence-level
    session also gets the previous boxes, as the upstream tracker feeds them.
    """
    geometry = SphericalGeometryImpl(boundarySamplesPerEdge=config.geometry.boundarySamplesPerEdge)
    frame = _requireFrame(source.read())
    height, width = frame.rgb.shape[:2]
    state = _largerSideOfSeam(source.sequence.initialBbox(), width)
    template = session.encodeTemplate(frame.rgb, state)
    collector.write(_erpResult(frame, state, 1.0, geometry, ResultSource.INITIAL))
    historyLength = int(getattr(session, "trajectoryLength", 0))
    history: deque[BBoxXYWH] = deque([state] * historyLength, maxlen=max(1, historyLength))
    while (frame := source.read()) is not None:
        extra = {"trajectories": (tuple(history),)} if historyLength else {}
        prediction = session.inferBatch(
            (frame.rgb,), (template,), priorBoxes=(state,), **extra
        )[0]
        state = prediction.bbox
        history.append(state)
        collector.write(
            _erpResult(
                frame, state, prediction.modelScore, geometry, ResultSource.OBSERVED_CONFIRMED
            )
        )


def _erpResult(
    frame: FramePacket,
    box: BBoxXYWH,
    confidence: float,
    geometry: SphericalGeometry,
    source: ResultSource,
) -> TrackResult:
    height, width = frame.rgb.shape[:2]
    return TrackResult(
        sequenceId=frame.sequenceId,
        frameIndex=frame.frameIndex,
        bbox=box,
        bfov=geometry.bboxToBfov(box, width, height),
        confidence=confidence,
        status=TrackStatus.TRACKING,
        valid=True,
        resultSource=source,
    )


def _largerSideOfSeam(box: BBoxXYWH, frameWidthPx: int) -> BBoxXYWH:
    """Keep the larger in-frame part of a seam-crossing box: B0 cannot wrap."""
    overflow = box.xPx + box.widthPx - frameWidthPx
    if overflow <= 0.0:
        return box
    inside = box.widthPx - overflow
    if inside >= overflow:
        return BBoxXYWH(box.xPx, box.yPx, inside, box.heightPx)
    return BBoxXYWH(0.0, box.yPx, overflow, box.heightPx)


def _requireFrame(frame: FramePacket | None) -> FramePacket:
    if frame is None:
        raise DecodeError("360VOT sequence is empty")
    return frame


def _isComplete(outputRoot: Path, method: str, name: str, frameCount: int) -> bool:
    for path in resultPaths(outputRoot, method, name):
        try:
            with path.open("r", encoding="utf-8") as stream:
                if sum(1 for _ in stream) != frameCount:
                    return False
        except OSError:
            return False
    return True


def _writeJson(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = [
    "METHODS",
    "BenchmarkSummary",
    "SequenceReport",
    "evaluateResults",
    "runBenchmark",
]
