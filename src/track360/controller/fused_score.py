"""Calibration and composition for backend, motion, and single-candidate scores."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import acos, exp, isfinite, log, log1p, pi

import numpy as np

from track360.controller.score_calibration import (
    ScoreCalibration,
)
from track360.core.errors import ProtocolError
from track360.core.types import (
    LocalObservation,
    MotionState3D,
    SphericalPoint,
)

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


__all__ = [
    "MotionScore",
    "calibrateBackendFusedScore",
    "calibrateLocalAppearanceProbabilities",
    "composeSingleScore",
    "scoreViewCenterMotion",
]
