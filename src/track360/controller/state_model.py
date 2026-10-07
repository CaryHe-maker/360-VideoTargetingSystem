"""Immutable state records shared by the controller components."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    MotionState3D,
    SequenceId,
    SphericalPoint,
)


class TrackMode(Enum):
    INIT = auto()
    TRACKING = auto()
    UNCERTAIN = auto()
    LOST = auto()


class TransitionReason(Enum):
    INITIALIZED = auto()
    RELIABLE_MEASUREMENT = auto()
    WEAK_MEASUREMENT = auto()
    HARD_MISS = auto()


@dataclass(frozen=True, slots=True)
class MotionSample:
    frameIndex: FrameIndex
    timestampNs: int
    center: SphericalPoint
    horizontalSizeRad: float
    verticalSizeRad: float
    confidence: float


@dataclass(frozen=True, slots=True)
class MotionPrediction:
    """Prediction plus uncertainty used by planning and candidate scoring."""

    sourceRevision: int
    targetFrameIndex: FrameIndex
    horizonFrames: int
    center: SphericalPoint
    horizontalSizeRad: float
    verticalSizeRad: float
    tangentVelocityRadPerSec: tuple[float, float]
    angularUncertaintyRad: float
    scaleUncertainty: float
    confidence: float
    centerCovarianceRad2: tuple[tuple[float, float], tuple[float, float]]
    scaleCovarianceLog2: tuple[tuple[float, float], tuple[float, float]]
    reliability: float
    degradedReasons: tuple[str, ...] = ()
    sampleCount: int = 0

    @property
    def motionState(self) -> MotionState3D:
        """Expose the legacy public motion type without losing V2 uncertainty."""
        return MotionState3D(
            position=(self.center.x, self.center.y, self.center.z),
            velocity=(
                self.tangentVelocityRadPerSec[0],
                self.tangentVelocityRadPerSec[1],
                0.0,
            ),
            confidence=self.confidence,
            horizontalSizeRad=self.horizontalSizeRad,
            verticalSizeRad=self.verticalSizeRad,
            angularUncertaintyRad=self.angularUncertaintyRad,
            scaleUncertainty=self.scaleUncertainty,
            reliability=self.reliability,
            centerCovarianceRad2=self.centerCovarianceRad2,
            scaleCovarianceLog2=self.scaleCovarianceLog2,
        )


@dataclass(frozen=True, slots=True)
class StateObservation:
    """What one frame's search view yielded, before the state transition."""

    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    evaluatedMode: TrackMode
    predictedCenter: SphericalPoint
    viewId: int | None
    localBox: BBoxXYWH | None
    measuredBfov: BFoV | None
    measuredBbox: BBoxXYWH | None
    proposedOutputBfov: BFoV
    proposedOutputBbox: BBoxXYWH
    backendScore: float
    motionScore: float
    scaleScore: float
    stateScore: float
    measurementAccepted: bool
    uncertainThreshold: float = 0.0
    lostThreshold: float = 0.0

    @property
    def hasCandidate(self) -> bool:
        return self.measuredBfov is not None


@dataclass(slots=True)
class ScoreGroup:
    """The bounded score history used to derive the next-frame state thresholds."""

    capacity: int = 10
    values: deque[float] = field(default_factory=lambda: deque(maxlen=10))

    def append(self, score: float) -> None:
        value = float(score)
        if not 0.0 <= value <= 1.0:
            raise ValueError("StateScore must be in [0, 1]")
        self.values.append(value)

    def thresholds(self) -> tuple[float, float] | None:
        """Return ``(UT, LT)`` from the scores already committed."""
        if len(self.values) < 2:
            return None
        ordered = sorted(self.values, reverse=True)
        if len(ordered) < self.capacity:
            highest = ordered[0]
            lowest = ordered[-1]
            return (0.5 * highest + 0.5 * lowest, 0.2 * highest + 0.8 * lowest)
        return (ordered[4], ordered[7])


@dataclass(frozen=True, slots=True)
class TransitionDecision:
    action: str
    nextMode: TrackMode
    reason: TransitionReason
    acceptMeasurement: bool


__all__ = [
    "MotionPrediction",
    "MotionSample",
    "ScoreGroup",
    "StateObservation",
    "TrackMode",
    "TransitionDecision",
    "TransitionReason",
]
