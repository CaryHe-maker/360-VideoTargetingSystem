"""Spherical tracking metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import acos, pi

import numpy as np

from track360.core.types import BFoV
from track360.geometry.projection_math import cameraBasis, makeSphericalPoint


def centerAngularErrorRad(first: BFoV, second: BFoV) -> float:
    dot = np.clip(
        first.center.x * second.center.x
        + first.center.y * second.center.y
        + first.center.z * second.center.z,
        -1.0,
        1.0,
    )
    return float(acos(dot))


def bfovSphericalIoU(
    first: BFoV,
    second: BFoV,
    samplesYaw: int = 256,
    samplesPitch: int = 128,
) -> float:
    if samplesYaw <= 0 or samplesPitch <= 0:
        raise ValueError("sample counts must be positive")
    yaws = np.linspace(-pi, pi, samplesYaw, endpoint=False, dtype=np.float64) + pi / samplesYaw
    pitches = np.linspace(-pi / 2.0, pi / 2.0, samplesPitch, endpoint=False, dtype=np.float64)
    yawGrid, pitchGrid = np.meshgrid(yaws, pitches)
    weights = np.cos(pitchGrid)
    points = np.vectorize(makeSphericalPoint)(yawGrid, pitchGrid)
    insideFirst = _contains(first, points)
    insideSecond = _contains(second, points)
    intersection = float(np.sum(weights[insideFirst & insideSecond]))
    union = float(np.sum(weights[insideFirst | insideSecond]))
    if union <= 0.0:
        return 0.0
    return float(np.clip(intersection / union, 0.0, 1.0))


@dataclass(slots=True)
class SphericalMetrics:
    centerErrorsRad: list[float] = field(default_factory=list)
    ious: list[float] = field(default_factory=list)

    def update(self, prediction: BFoV, target: BFoV) -> None:
        self.centerErrorsRad.append(centerAngularErrorRad(prediction, target))
        self.ious.append(bfovSphericalIoU(prediction, target))

    def summarize(self) -> dict[str, float]:
        if not self.centerErrorsRad:
            return {"meanCenterErrorRad": 0.0, "meanIoU": 0.0}
        return {
            "meanCenterErrorRad": float(np.mean(self.centerErrorsRad)),
            "meanIoU": float(np.mean(self.ious)),
        }


def summarizeTrackResults(results: Sequence[tuple[BFoV, BFoV]]) -> dict[str, float]:
    if not results:
        return {"meanCenterErrorRad": 0.0, "meanIoU": 0.0, "sampleCount": 0.0}
    metric = SphericalMetrics()
    for prediction, target in results:
        metric.update(prediction, target)
    summary = metric.summarize()
    summary["sampleCount"] = float(len(results))
    return summary


def _contains(bfov: BFoV, points) -> np.ndarray:
    forward, right, up = cameraBasis(bfov)
    flat = np.asarray(
        [(point.x, point.y, point.z) for point in np.asarray(points, dtype=object).reshape(-1)],
        dtype=np.float64,
    )
    dotsForward = flat @ forward
    dotsRight = flat @ right
    dotsUp = flat @ up
    horizontal = np.abs(np.arctan2(dotsRight, dotsForward))
    vertical = np.abs(np.arctan2(dotsUp, dotsForward))
    mask = (horizontal <= bfov.horizontalFovRad / 2.0) & (vertical <= bfov.verticalFovRad / 2.0)
    return mask.reshape(np.asarray(points, dtype=object).shape)


__all__ = ["SphericalMetrics", "bfovSphericalIoU", "centerAngularErrorRad", "summarizeTrackResults"]
