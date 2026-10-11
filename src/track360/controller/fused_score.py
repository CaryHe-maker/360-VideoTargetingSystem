"""Scores attached to an observation before the controller judges it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import acos, exp, isfinite, log, log1p, pi

import numpy as np

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


def backendScoreProbability(score: float) -> float:
    """The backend score as the probability the controller works with.

    Mathematically the score itself: the logit of the score put back through the
    logistic function.  The round trip is kept because it moves the last digits, and
    the number weights each frame in the motion model: dropping it would change the
    recorded baselines without changing anything else (it is what is left of a score
    calibration that was never fitted).
    """
    value = float(score)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ProtocolError(f"backend fusedScore must be in [0, 1], actual={score}")
    if value == 0.0 or value == 1.0:
        return value
    logit = 0.0 + 1.0 * log(value) - 1.0 * log1p(-value)
    if logit >= 0.0:
        return 1.0 / (1.0 + exp(-logit))
    exponential = exp(logit)
    return exponential / (1.0 + exponential)


def withScoreProbability(
    observations: Sequence[LocalObservation],
) -> tuple[LocalObservation, ...]:
    """Attach the score probability without overwriting the backend's own numbers."""
    return tuple(
        replace(
            observation,
            appearanceProbability=backendScoreProbability(observation.fusedScore),
        )
        for observation in observations
    )


def scoreViewCenterMotion(
    viewCenter: SphericalPoint,
    prediction: MotionState3D | None,
) -> MotionScore:
    """Score one local view center against this frame's predicted spherical position.

    This is a same-frame spatial prior: 0 degrees maps to 1.0 and every additional
    30 degrees continuously subtracts 0.1.  It is recorded with the observation; the
    controller's decisions do not use it.
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


__all__ = [
    "MotionScore",
    "backendScoreProbability",
    "scoreViewCenterMotion",
    "withScoreProbability",
]
