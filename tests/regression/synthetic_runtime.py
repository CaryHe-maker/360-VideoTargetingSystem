"""Deterministic CPU end-to-end scenarios for golden regression tests.

A synthetic ERP sequence and a color-threshold fake ARTrack session drive the real
``buildRuntime`` / ``runTracking`` path, so the controller, geometry and template
logic run exactly as in production without a GPU or model weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from track360.backends.artrack_model import ARTrackPrediction, ARTrackTemplate
from track360.core.config import AppConfig, ModelConfig
from track360.core.types import BBoxXYWH, FrameIndex, FramePacket, SequenceId, TrackResult
from track360.runtime.driver import buildRuntime, closeRuntime, runTracking

FRAME_WIDTH = 960
FRAME_HEIGHT = 480
FRAME_INTERVAL_NS = 33_333_333
SCENARIOS = ("seam", "large")


class FakeARTrackSession:
    """Locate the saturated-red target in each view and score it by its coverage."""

    supportsOnlineTemplates = True

    def __init__(self, config: ModelConfig) -> None:
        del config
        self.calls: list[dict[str, Any]] = []

    def encodeTemplate(self, rgb: NDArray[np.uint8], bbox: BBoxXYWH) -> ARTrackTemplate:
        self.calls.append({"op": "encodeTemplate", "shape": list(rgb.shape[:2])})
        return ARTrackTemplate(tensor=None, bbox=bbox)

    def infer(
        self, rgb: NDArray[np.uint8], templateFeatures: Sequence[object]
    ) -> ARTrackPrediction:
        return self.inferBatchWithFovs((rgb,), templateFeatures, ((0.0, 0.0),))[0]

    def inferBatchWithFovs(
        self,
        rgbs: Sequence[NDArray[np.uint8]],
        templateFeatures: Sequence[object],
        imageFovs: Sequence[tuple[float, float]],
    ) -> tuple[ARTrackPrediction, ...]:
        self.calls.append(
            {
                "op": "inferBatch",
                "views": len(rgbs),
                "templates": len(templateFeatures),
                "fovs": [[float(fov[0]), float(fov[1])] for fov in imageFovs],
            }
        )
        return tuple(_locate(rgb, len(templateFeatures)) for rgb in rgbs)

    def inferBatch(
        self,
        rgbs: Sequence[NDArray[np.uint8]],
        templateFeatures: Sequence[object],
        *,
        imageFovs: Sequence[tuple[float, float]] | None = None,
        priorBoxes: Sequence[BBoxXYWH] | None = None,
    ) -> tuple[ARTrackPrediction, ...]:
        """Search around a prior box: ERP frames, or views planned with a prior."""
        if imageFovs is not None:
            # A perspective view with a search prior (aligned search).
            self.calls.append(
                {
                    "op": "inferBatch",
                    "views": len(rgbs),
                    "templates": len(templateFeatures),
                    "fovs": [[float(fov[0]), float(fov[1])] for fov in imageFovs],
                    "priors": priorBoxes is not None,
                }
            )
        else:
            self.calls.append(
                {"op": "inferErp", "views": len(rgbs), "priors": priorBoxes is not None}
            )
        return tuple(_locate(rgb, len(templateFeatures)) for rgb in rgbs)

    def close(self) -> None:
        self.calls.append({"op": "close"})


class _MemorySource:
    def __init__(self, frames: Sequence[FramePacket]) -> None:
        self._frames = list(frames)

    def open(self, uri: str) -> None:
        del uri

    def read(self) -> FramePacket | None:
        return self._frames.pop(0) if self._frames else None

    def close(self) -> None:
        self._frames.clear()


class _MemorySink:
    def __init__(self) -> None:
        self.results: list[TrackResult] = []

    def open(self, destination: str) -> None:
        del destination

    def write(self, result: TrackResult) -> None:
        self.results.append(result)

    def finalize(self, expectedFrameCount: int) -> None:
        del expectedFrameCount


def runScenario(config: AppConfig, scenario: str) -> dict[str, Any]:
    """Run one synthetic scenario through the production runtime and return its trace."""
    frames, initialBox = buildScenario(scenario)
    sessions: list[FakeARTrackSession] = []

    def sessionFactory(modelConfig: ModelConfig) -> FakeARTrackSession:
        session = FakeARTrackSession(modelConfig)
        sessions.append(session)
        return session

    runtime = buildRuntime(config, artrackSessionFactory=sessionFactory)
    sink = _MemorySink()
    try:
        runTracking(
            source=_MemorySource(frames),
            initialBox=initialBox,
            geometry=runtime.geometry,
            controller=runtime.controller,
            backend=runtime.backend,
            sink=sink,
            scoreCalibration=runtime.scoreCalibration,
            useMotionScore=runtime.useMotionScore,
        )
    finally:
        closeRuntime(runtime)
    return {
        "scenario": scenario,
        "results": [_resultRecord(result) for result in sink.results],
        "backendCalls": sessions[0].calls,
    }


def buildScenario(scenario: str) -> tuple[list[FramePacket], BBoxXYWH]:
    if scenario == "seam":
        # A small target crosses the +/-180 degree seam, disappears for four
        # frames, and is later joined by a same-colored distractor.
        frameCount = 40
        size = (20.0, 14.0)

        def target(index: int) -> tuple[float, float] | None:
            if 18 <= index <= 21:
                return None
            return 150.0 + 2.0 * index, 10.0 - 0.5 * index

        def distractor(index: int) -> tuple[float, float] | None:
            return (target(index)[0] - 55.0, 25.0) if 26 <= index <= 31 else None

    elif scenario == "large":
        # A target covering more than 10% of the frame exercises the weak-hold path.
        frameCount = 30
        size = (150.0, 60.0)

        def target(index: int) -> tuple[float, float] | None:
            if index in {8, 9, 10, 17, 18}:
                return None
            return -20.0 + 1.5 * index, 5.0

        def distractor(index: int) -> tuple[float, float] | None:
            del index
            return None

    else:
        raise ValueError(f"unknown scenario: {scenario}")

    frames: list[FramePacket] = []
    background = _background()
    for index in range(frameCount):
        rgb = background.copy()
        other = distractor(index)
        if other is not None:
            _drawTarget(rgb, other, (12.0, 9.0))
        center = target(index)
        if center is not None:
            _drawTarget(rgb, center, size)
        rgb.setflags(write=False)
        frames.append(
            FramePacket(
                SequenceId(f"synthetic-{scenario}"),
                FrameIndex(index),
                index * FRAME_INTERVAL_NS,
                rgb,
            )
        )
    first = target(0)
    assert first is not None
    return frames, _erpBox(first, size)


def _background() -> NDArray[np.uint8]:
    yy, xx = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH]
    rgb = np.empty((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    rgb[..., 0] = 20 + (xx // 48 + yy // 48) % 2 * 20
    rgb[..., 1] = 60 + (xx * 120 // FRAME_WIDTH)
    rgb[..., 2] = 60 + (yy * 120 // FRAME_HEIGHT)
    return rgb


def _erpBox(centerDeg: tuple[float, float], sizeDeg: tuple[float, float]) -> BBoxXYWH:
    widthPx = sizeDeg[0] / 360.0 * FRAME_WIDTH
    heightPx = sizeDeg[1] / 180.0 * FRAME_HEIGHT
    centerX = ((centerDeg[0] + 180.0) % 360.0) / 360.0 * FRAME_WIDTH
    centerY = (0.5 - centerDeg[1] / 180.0) * FRAME_HEIGHT
    return BBoxXYWH(centerX - widthPx / 2.0, centerY - heightPx / 2.0, widthPx, heightPx)


def _drawTarget(
    rgb: NDArray[np.uint8],
    centerDeg: tuple[float, float],
    sizeDeg: tuple[float, float],
) -> None:
    box = _erpBox(centerDeg, sizeDeg)
    top = max(0, int(round(box.yPx)))
    bottom = min(FRAME_HEIGHT, int(round(box.yPx + box.heightPx)))
    left = int(round(box.xPx))
    columns = np.arange(left, left + int(round(box.widthPx))) % FRAME_WIDTH
    rgb[top:bottom, columns] = (255, 0, 0)


def _locate(rgb: NDArray[np.uint8], templateCount: int) -> ARTrackPrediction:
    height, width = rgb.shape[:2]
    mask = (rgb[..., 0] > 160) & (rgb[..., 1] < 90) & (rgb[..., 2] < 90)
    count = int(mask.sum())
    if count < 4:
        box = BBoxXYWH(width * 0.375, height * 0.375, width * 0.25, height * 0.25)
        return ARTrackPrediction(box, 0.30, 0.30, 0.30)
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    left, right = int(columns[0]), int(columns[-1]) + 1
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    touchesEdge = left == 0 or top == 0 or right == width or bottom == height
    fill = count / float((right - left) * (bottom - top))
    coverage = count / float(width * height)
    score = 0.38 + 0.12 * fill + 0.20 * min(coverage * 4.0, 1.0) + 0.01 * templateCount
    if touchesEdge:
        score -= 0.06
    score = float(np.clip(score, 0.0, 1.0))
    box = BBoxXYWH(float(left), float(top), float(right - left), float(bottom - top))
    return ARTrackPrediction(box, score, score, score)


def _resultRecord(result: TrackResult) -> dict[str, Any]:
    return {
        "frameIndex": int(result.frameIndex),
        "bbox": [
            result.bbox.xPx,
            result.bbox.yPx,
            result.bbox.widthPx,
            result.bbox.heightPx,
        ],
        "bfov": [
            result.bfov.center.yawRad,
            result.bfov.center.pitchRad,
            result.bfov.horizontalFovRad,
            result.bfov.verticalFovRad,
        ],
        "confidence": result.confidence,
        "status": result.status.name,
        "valid": result.valid,
        "resultSource": result.resultSource.name,
    }
