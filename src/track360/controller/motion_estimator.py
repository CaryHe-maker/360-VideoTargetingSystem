"""Windowed spherical motion prediction for the DTC control thread."""

from __future__ import annotations

from collections import deque
from math import asin, atan2

import numpy as np

from track360.controller.state_model import MotionPrediction, MotionSample
from track360.core.errors import ProtocolError
from track360.core.protocols import MotionEstimator as MotionEstimatorProtocol
from track360.core.types import MotionState3D, SphericalPoint
from track360.geometry.projection_math import makeSphericalPoint


class SphericalMotionEstimator(MotionEstimatorProtocol):
    """Robust constant-velocity estimator with a bounded reliable-measurement window.

    The legacy ``initialize/predict/update`` protocol remains available.  The controller uses
    ``predictDetailed`` and ``resetFromMeasurement`` so invalid output predictions never enter
    the measurement history.
    """

    def __init__(
        self,
        alpha: float = 0.70,
        beta: float = 0.20,
        windowLength: int = 5,
        maxPredictionHorizon: int = 3,
        minSamplesForVelocity: int = 2,
        maxTangentSpanRad: float = 1.20,
        huberDeltaRad: float = 0.15,
        processNoiseRadPerSec: float = 0.04,
        maxAngularSpeedRadPerSec: float = 2.0,
        maxLogScaleRatePerSec: float = 1.0,
    ) -> None:
        if not 0.0 < alpha <= 1.0 or not 0.0 < beta <= 1.0:
            raise ValueError("alpha and beta must be in (0, 1]")
        if windowLength < 2 or maxPredictionHorizon < 1:
            raise ValueError("windowLength and maxPredictionHorizon must be positive")
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._windowLength = int(windowLength)
        self._maxPredictionHorizon = int(maxPredictionHorizon)
        self._minSamplesForVelocity = int(minSamplesForVelocity)
        self._maxTangentSpanRad = float(maxTangentSpanRad)
        self._huberDeltaRad = float(huberDeltaRad)
        self._processNoiseRadPerSec = float(processNoiseRadPerSec)
        self._maxAngularSpeedRadPerSec = float(maxAngularSpeedRadPerSec)
        self._maxLogScaleRatePerSec = float(maxLogScaleRatePerSec)
        self._samples: deque[MotionSample] = deque(maxlen=self._windowLength)
        self._initialized = False
        self._lastPrediction: MotionPrediction | None = None
        self._timestampNs = 0

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def samples(self) -> tuple[MotionSample, ...]:
        return tuple(self._samples)

    def initialize(
        self,
        point: SphericalPoint,
        timestampNs: int,
    ) -> MotionState3D:
        _requireTimestamp(timestampNs)
        self._samples.clear()
        self._appendSample(
            MotionSample(
                frameIndex=0,
                timestampNs=timestampNs,
                center=point,
                horizontalSizeRad=0.0,
                verticalSizeRad=0.0,
                confidence=1.0,
            )
        )
        self._initialized = True
        self._timestampNs = int(timestampNs)
        return self._legacyState(self.predictDetailed(timestampNs, 0))

    def resetFromMeasurement(
        self,
        point: SphericalPoint,
        timestampNs: int,
        frameIndex: int,
        confidence: float,
        horizontalSizeRad: float = 0.0,
        verticalSizeRad: float = 0.0,
    ) -> MotionPrediction:
        """Rebase after reacquisition and discard stale velocity hypotheses."""
        _requireTimestamp(timestampNs)
        if not 0.0 <= confidence <= 1.0:
            raise ProtocolError("motion confidence must be in [0, 1]")
        self._samples.clear()
        self._appendSample(
            MotionSample(
                frameIndex=frameIndex,
                timestampNs=timestampNs,
                center=point,
                horizontalSizeRad=max(0.0, horizontalSizeRad),
                verticalSizeRad=max(0.0, verticalSizeRad),
                confidence=confidence,
            )
        )
        self._initialized = True
        self._timestampNs = int(timestampNs)
        return self.predictDetailed(timestampNs, 0)

    def predict(self, timestampNs: int) -> MotionState3D:
        prediction = self.predictDetailed(timestampNs, 0)
        return self._legacyState(prediction)

    def predictDetailed(self, timestampNs: int, horizonFrames: int = 1) -> MotionPrediction:
        _requireTimestamp(timestampNs)
        self._requireInitialized()
        if timestampNs < self._timestampNs:
            raise ProtocolError("motion timestamp must be monotonic")
        horizon = max(0, min(self._maxPredictionHorizon, int(horizonFrames)))
        latest = self._samples[-1]
        dt = max((timestampNs - latest.timestampNs) / 1e9, 0.0)
        velocity, residual = self._fitVelocity()
        center = _advanceOnSphere(latest.center, velocity, dt)
        width, height, scaleLogVelocity = self._fitScale()
        width = max(0.0, width * np.exp(scaleLogVelocity[0] * dt))
        height = max(0.0, height * np.exp(scaleLogVelocity[1] * dt))
        uncertainty = max(
            0.01,
            residual + self._processNoiseRadPerSec * dt + 0.02 * len(self._samples) ** -0.5,
        )
        confidence = float(np.clip(latest.confidence * np.exp(-0.25 * dt), 0.0, 1.0))
        degraded: list[str] = []
        if len(self._samples) < self._minSamplesForVelocity:
            degraded.append("insufficient_motion_samples")
        scaleUncertainty = 0.25 if len(self._samples) < 2 else 0.10
        # One confirmed point is already useful as a position/scale anchor even though it
        # cannot define velocity.  Grow reliability continuously with history maturity instead
        # of switching from the neutral 0.5 fallback to a fully active model at sample N.
        historyMaturity = min(1.0, len(self._samples) / self._minSamplesForVelocity)
        reliability = float(
            np.clip(
                confidence * np.exp(-uncertainty / 0.35) * historyMaturity,
                0.0,
                1.0,
            )
        )
        result = MotionPrediction(
            sourceRevision=0,
            targetFrameIndex=0,
            horizonFrames=horizon,
            center=center,
            horizontalSizeRad=width,
            verticalSizeRad=height,
            tangentVelocityRadPerSec=(float(velocity[0]), float(velocity[1])),
            angularUncertaintyRad=float(uncertainty),
            scaleUncertainty=scaleUncertainty,
            confidence=confidence,
            centerCovarianceRad2=(
                (float(uncertainty * uncertainty), 0.0),
                (0.0, float(uncertainty * uncertainty)),
            ),
            scaleCovarianceLog2=(
                (float(scaleUncertainty * scaleUncertainty), 0.0),
                (0.0, float(scaleUncertainty * scaleUncertainty)),
            ),
            reliability=reliability,
            degradedReasons=tuple(degraded),
            sampleCount=len(self._samples),
        )
        self._lastPrediction = result
        return result

    def update(
        self,
        point: SphericalPoint,
        timestampNs: int,
        observationConfidence: float,
    ) -> MotionState3D:
        _requireTimestamp(timestampNs)
        self._requireInitialized()
        if timestampNs < self._timestampNs:
            raise ProtocolError("motion timestamp must be monotonic")
        if not 0.0 <= observationConfidence <= 1.0:
            raise ProtocolError("observationConfidence must be in [0, 1]")
        self._appendSample(
            MotionSample(
                frameIndex=self._samples[-1].frameIndex + 1,
                timestampNs=timestampNs,
                center=point,
                horizontalSizeRad=self._samples[-1].horizontalSizeRad,
                verticalSizeRad=self._samples[-1].verticalSizeRad,
                confidence=observationConfidence,
            )
        )
        self._timestampNs = int(timestampNs)
        return self._legacyState(self.predictDetailed(timestampNs, 0))

    def recordMeasurement(
        self,
        *,
        frameIndex: int,
        timestampNs: int,
        point: SphericalPoint,
        confidence: float,
        horizontalSizeRad: float,
        verticalSizeRad: float,
    ) -> MotionPrediction:
        """Append one confirmed measurement; prediction-only frames never call this."""
        _requireTimestamp(timestampNs)
        self._requireInitialized()
        if timestampNs < self._timestampNs:
            raise ProtocolError("motion timestamp must be monotonic")
        self._appendSample(
            MotionSample(
                frameIndex=frameIndex,
                timestampNs=timestampNs,
                center=point,
                horizontalSizeRad=max(0.0, horizontalSizeRad),
                verticalSizeRad=max(0.0, verticalSizeRad),
                confidence=confidence,
            )
        )
        self._timestampNs = int(timestampNs)
        return self.predictDetailed(timestampNs, 0)

    def predictWithProvisionalMeasurement(
        self,
        *,
        frameIndex: int,
        timestampNs: int,
        point: SphericalPoint,
        confidence: float,
        horizontalSizeRad: float,
        verticalSizeRad: float,
    ) -> MotionPrediction:
        """Predict from an R1 measurement without committing it to motion history."""
        _requireTimestamp(timestampNs)
        self._requireInitialized()
        savedSamples = tuple(self._samples)
        savedTimestamp = self._timestampNs
        savedPrediction = self._lastPrediction
        try:
            self._appendSample(
                MotionSample(
                    frameIndex=frameIndex,
                    timestampNs=timestampNs,
                    center=point,
                    horizontalSizeRad=max(0.0, horizontalSizeRad),
                    verticalSizeRad=max(0.0, verticalSizeRad),
                    confidence=confidence,
                )
            )
            self._timestampNs = int(timestampNs)
            return self.predictDetailed(timestampNs, 0)
        finally:
            self._samples.clear()
            self._samples.extend(savedSamples)
            self._timestampNs = savedTimestamp
            self._lastPrediction = savedPrediction

    def _appendSample(self, sample: MotionSample) -> None:
        self._samples.append(sample)

    def _fitVelocity(self) -> tuple[np.ndarray, float]:
        if len(self._samples) < self._minSamplesForVelocity:
            return np.zeros(2, dtype=np.float64), 0.0
        latest = self._samples[-1]
        times = np.asarray(
            [(s.timestampNs - latest.timestampNs) / 1e9 for s in self._samples], dtype=np.float64
        )
        origin = np.asarray((latest.center.x, latest.center.y, latest.center.z), dtype=np.float64)
        points = np.asarray(
            [(s.center.x, s.center.y, s.center.z) for s in self._samples], dtype=np.float64
        )
        # Use a local tangent basis at the latest point.  This keeps yaw wrap and polar motion
        # continuous, including at the poles where the usual yaw-derived east vector vanishes.
        east, north = _tangentBasis(origin)
        coords = np.column_stack(((points - origin) @ east, (points - origin) @ north))
        weights = np.asarray([max(0.05, s.confidence) for s in self._samples], dtype=np.float64)
        design = np.column_stack((np.ones(len(times)), times))
        velocity = np.zeros(2, dtype=np.float64)
        residual = 0.0
        for axis in range(2):
            solution, axisWeights = _robustLineFit(
                design,
                coords[:, axis],
                weights,
                self._huberDeltaRad,
            )
            velocity[axis] = float(solution[1])
            residual += float(
                np.average(
                    np.abs(coords[:, axis] - design @ solution),
                    weights=axisWeights,
                )
            )
        speed = float(np.linalg.norm(velocity))
        if speed > self._maxAngularSpeedRadPerSec:
            velocity *= self._maxAngularSpeedRadPerSec / speed
        if residual > self._maxTangentSpanRad:
            return np.zeros(2, dtype=np.float64), residual
        return velocity, min(residual, self._huberDeltaRad)

    def _fitScale(self) -> tuple[float, float, tuple[float, float]]:
        samples = [
            s for s in self._samples if s.horizontalSizeRad > 0.0 and s.verticalSizeRad > 0.0
        ]
        if not samples:
            return 0.0, 0.0, (0.0, 0.0)
        if len(samples) < 2:
            return samples[-1].horizontalSizeRad, samples[-1].verticalSizeRad, (0.0, 0.0)
        times = np.asarray(
            [(s.timestampNs - samples[-1].timestampNs) / 1e9 for s in samples], dtype=np.float64
        )
        design = np.column_stack((np.ones(len(times)), times))
        values = []
        for field in ("horizontalSizeRad", "verticalSizeRad"):
            y = np.log(np.asarray([getattr(s, field) for s in samples], dtype=np.float64))
            solution = np.linalg.lstsq(design, y, rcond=None)[0]
            rate = float(
                np.clip(
                    solution[1],
                    -self._maxLogScaleRatePerSec,
                    self._maxLogScaleRatePerSec,
                )
            )
            values.append((float(np.exp(solution[0])), rate))
        return values[0][0], values[1][0], (values[0][1], values[1][1])

    def _legacyState(self, prediction: MotionPrediction) -> MotionState3D:
        return prediction.motionState

    def _requireInitialized(self) -> None:
        if not self._initialized or not self._samples:
            raise ProtocolError("motion estimator has not been initialized")


def _advanceOnSphere(point: SphericalPoint, velocity: np.ndarray, dt: float) -> SphericalPoint:
    vector = np.asarray((point.x, point.y, point.z), dtype=np.float64)
    east, north = _tangentBasis(vector)
    moved = vector + (velocity[0] * east + velocity[1] * north) * dt
    norm = float(np.linalg.norm(moved))
    if norm <= 1e-12 or not np.isfinite(norm):
        return point
    moved /= norm
    return makeSphericalPoint(atan2(moved[0], moved[2]), asin(float(np.clip(moved[1], -1.0, 1.0))))


def _tangentBasis(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable orthonormal tangent basis for any unit sphere position."""
    east = np.asarray((-vector[2], 0.0, vector[0]), dtype=np.float64)
    eastNorm = float(np.linalg.norm(east))
    if eastNorm <= 1e-8:
        east = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    else:
        east /= eastNorm
    north = np.cross(vector, east)
    northNorm = float(np.linalg.norm(north))
    if northNorm <= 1e-12 or not np.isfinite(northNorm):
        raise ProtocolError("motion tangent basis requires a finite unit sphere position")
    return east, north / northNorm


def _robustLineFit(
    design: np.ndarray,
    values: np.ndarray,
    baseWeights: np.ndarray,
    huberDelta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an intercept and slope with deterministic Huber reweighting."""
    weights = baseWeights.copy()
    solution = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(3):
        root = np.sqrt(weights)
        solution = np.linalg.lstsq(
            design * root[:, None],
            values * root,
            rcond=None,
        )[0]
        residuals = np.abs(values - design @ solution)
        huberWeights = np.ones_like(residuals)
        outliers = residuals > huberDelta
        huberWeights[outliers] = huberDelta / residuals[outliers]
        weights = np.maximum(1e-6, baseWeights * huberWeights)
    return solution, weights


def _requireTimestamp(timestampNs: int) -> None:
    if isinstance(timestampNs, bool) or not isinstance(timestampNs, int) or timestampNs < 0:
        raise ProtocolError("motion timestampNs must be a non-negative integer")


MotionEstimatorImpl = SphericalMotionEstimator

__all__ = ["MotionEstimatorImpl", "SphericalMotionEstimator"]
