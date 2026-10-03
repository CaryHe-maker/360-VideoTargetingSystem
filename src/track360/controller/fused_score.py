"""Calibration and composition for backend, motion, and single-candidate scores."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import acos, atan2, exp, isfinite, log, log1p, pi

import numpy as np

from track360.controller.score_calibration import (
    ScoreCalibration,
)
from track360.core.errors import ProtocolError
from track360.core.types import (
    BFoV,
    LocalObservation,
    MotionState3D,
    SphericalPoint,
)

MOTION_SCALE_WEIGHT = 0.35
MOTION_MAX_D2 = 25.0
MOTION_CENTER_MEASUREMENT_STD_RAD = 0.025
MOTION_SCALE_MEASUREMENT_STD_LOG = 0.08
VIEW_MOTION_ANGLE_STEP_RAD = pi / 6.0
VIEW_MOTION_SCORE_DROP_PER_STEP = 0.10


@dataclass(frozen=True, slots=True)
class MotionScore:
    rawScore: float
    probability: float
    effectiveProbability: float
    reliability: float
    squaredDistance: float


def calibrateBackendFusedScore(
    score: float,
    calibration: ScoreCalibration,
) -> float:
    """Calibrate a backend probability with a monotonic beta map."""
    value = float(score)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ProtocolError(f"backend fusedScore must be in [0, 1], actual={score}")
    if value == 0.0 or value == 1.0:
        return value

    alpha = calibration.appearance.alpha
    beta = calibration.appearance.beta
    intercept = calibration.appearance.intercept
    calibratedLogit = intercept + alpha * log(value) - beta * log1p(-value)
    if calibratedLogit >= 0.0:
        return 1.0 / (1.0 + exp(-calibratedLogit))
    exponential = exp(calibratedLogit)
    return exponential / (1.0 + exponential)


def calibrateLocalAppearanceProbabilities(
    observations: Sequence[LocalObservation],
    calibration: ScoreCalibration,
) -> tuple[LocalObservation, ...]:
    """Attach appearance probabilities without overwriting backend evidence."""
    return tuple(
        replace(
            observation,
            appearanceProbability=calibrateBackendFusedScore(
                observation.fusedScore,
                calibration,
            ),
        )
        for observation in observations
    )


def calibrateMotionScore(rawScore: float) -> float:
    """Identity calibration placeholder frozen until a calibration set is available."""
    value = float(rawScore)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ProtocolError(f"rawMotionScore must be in [0, 1], actual={rawScore}")
    return value


def scoreMotionConsistency(
    candidate: BFoV,
    prediction: MotionState3D | None,
) -> MotionScore:
    """Map covariance-normalized center and scale residuals to a probability."""
    if prediction is None:
        return MotionScore(0.5, 0.5, 0.5, 0.0, 0.0)

    origin = np.asarray(prediction.position, dtype=np.float64)
    originNorm = float(np.linalg.norm(origin))
    if originNorm <= 1e-12:
        raise ProtocolError("motion prediction position must be non-zero")
    origin /= originNorm
    candidateVector = np.asarray(
        (candidate.center.x, candidate.center.y, candidate.center.z), dtype=np.float64
    )
    east = np.asarray((-origin[2], 0.0, origin[0]), dtype=np.float64)
    if float(np.linalg.norm(east)) <= 1e-8:
        east = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    else:
        east /= np.linalg.norm(east)
    north = np.cross(origin, east)
    north /= max(float(np.linalg.norm(north)), 1e-12)
    forward = float(candidateVector @ origin)
    centerResidual = np.asarray(
        (
            atan2(float(candidateVector @ east), forward),
            atan2(float(candidateVector @ north), forward),
        ),
        dtype=np.float64,
    )
    centerCovariance = np.asarray(prediction.centerCovarianceRad2, dtype=np.float64)
    centerCovariance += np.eye(2) * MOTION_CENTER_MEASUREMENT_STD_RAD**2
    centerD2 = _quadraticDistance(centerResidual, centerCovariance)

    scaleD2 = 0.0
    if (
        prediction.horizontalSizeRad > 0.0
        and prediction.verticalSizeRad > 0.0
        and candidate.horizontalFovRad > 0.0
        and candidate.verticalFovRad > 0.0
    ):
        scaleResidual = np.log(
            np.asarray(
                (
                    candidate.horizontalFovRad / prediction.horizontalSizeRad,
                    candidate.verticalFovRad / prediction.verticalSizeRad,
                ),
                dtype=np.float64,
            )
        )
        scaleCovariance = np.asarray(prediction.scaleCovarianceLog2, dtype=np.float64)
        scaleCovariance += np.eye(2) * MOTION_SCALE_MEASUREMENT_STD_LOG**2
        scaleD2 = _quadraticDistance(scaleResidual, scaleCovariance)

    squaredDistance = centerD2 + MOTION_SCALE_WEIGHT * scaleD2
    rawScore = exp(-0.5 * min(float(squaredDistance), MOTION_MAX_D2))
    probability = calibrateMotionScore(rawScore)
    reliability = float(np.clip(prediction.reliability, 0.0, 1.0))
    effective = reliability * probability + (1.0 - reliability) * 0.5
    return MotionScore(
        rawScore=float(rawScore),
        probability=float(probability),
        effectiveProbability=float(np.clip(effective, 0.0, 1.0)),
        reliability=reliability,
        squaredDistance=float(squaredDistance),
    )


def scoreViewCenterMotion(
    viewCenter: SphericalPoint,
    prediction: MotionState3D | None,
) -> MotionScore:
    """Score one local view center against this frame's predicted spherical position.

    This is a same-frame spatial prior: 0 degrees maps to 1.0 and every additional
    30 degrees continuously subtracts 0.1.  It intentionally does not blend around 0.5,
    because all views in the frame must remain directly comparable to the same prediction.
    """
    if prediction is None:
        return MotionScore(0.5, 0.5, 0.5, 0.0, 0.0)

    predicted = np.asarray(prediction.position, dtype=np.float64)
    predictedNorm = float(np.linalg.norm(predicted))
    if predictedNorm <= 1e-12:
        raise ProtocolError("motion prediction position must be non-zero")
    predicted /= predictedNorm
    center = np.asarray((viewCenter.x, viewCenter.y, viewCenter.z), dtype=np.float64)
    centerNorm = float(np.linalg.norm(center))
    if centerNorm <= 1e-12:
        raise ProtocolError("local view center must be non-zero")
    center /= centerNorm

    angleRad = acos(float(np.clip(center @ predicted, -1.0, 1.0)))
    score = 1.0 - (
        angleRad / VIEW_MOTION_ANGLE_STEP_RAD * VIEW_MOTION_SCORE_DROP_PER_STEP
    )
    score = float(np.clip(score, 0.0, 1.0))
    return MotionScore(
        rawScore=score,
        probability=score,
        effectiveProbability=score,
        reliability=float(np.clip(prediction.reliability, 0.0, 1.0)),
        squaredDistance=float(angleRad * angleRad),
    )


def composeSingleScore(
    appearanceProbability: float,
    motionProbability: float,
    calibration: ScoreCalibration,
) -> float:
    """Compose the score consumed by candidate ranking and two-box fusion."""
    for name, value in (
        ("appearanceProbability", appearanceProbability),
        ("motionProbability", motionProbability),
    ):
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ProtocolError(f"{name} must be in [0, 1], actual={value}")
    return float(
        np.clip(
            calibration.appearanceWeight * appearanceProbability
            + calibration.motionWeight * motionProbability,
            0.0,
            1.0,
        )
    )


def _quadraticDistance(residual: np.ndarray, covariance: np.ndarray) -> float:
    covariance = covariance + np.eye(covariance.shape[0]) * 1e-9
    try:
        solved = np.linalg.solve(covariance, residual)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(covariance) @ residual
    return max(0.0, float(residual @ solved))


__all__ = [
    "MotionScore",
    "calibrateBackendFusedScore",
    "calibrateLocalAppearanceProbabilities",
    "calibrateMotionScore",
    "composeSingleScore",
    "scoreMotionConsistency",
    "scoreViewCenterMotion",
]
