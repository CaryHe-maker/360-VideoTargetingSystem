"""Immutable state records shared by the controller components."""

from __future__ import annotations

from dataclasses import dataclass
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
    # Right after a jump to a scan candidate, before it is confirmed.
    PROBATION = auto()


class TransitionReason(Enum):
    INITIALIZED = auto()
    RELIABLE_MEASUREMENT = auto()
    WEAK_MEASUREMENT = auto()
    HARD_MISS = auto()
    RELEASED = auto()
    BACKEND_LOW = auto()
    MOTION_LOW = auto()
    APPEARANCE_LOW = auto()
    DOUBT_HELD = auto()
    APPEARANCE_CONFIRMED_LOSS = auto()
    ON_PROBATION = auto()
    PROBATION_PASSED = auto()
    PROBATION_FAILED = auto()


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
    # The weighted mean of the backend, appearance and motion scores.
    stateScore: float
    measurementAccepted: bool
    # Similarity of the box to the target's appearance; None: not measured.
    appearanceScore: float | None = None
    uncertainThreshold: float = 0.0
    # The two residuals behind the motion score: distance from the predicted
    # position in predicted target sizes, and the log of the size ratio.
    motionOffset: float | None = None
    motionLogScale: float | None = None

    @property
    def hasCandidate(self) -> bool:
        return self.measuredBfov is not None


@dataclass(frozen=True, slots=True)
class TransitionDecision:
    action: str
    nextMode: TrackMode
    reason: TransitionReason
    acceptMeasurement: bool


__all__ = [
    "MotionPrediction",
    "MotionSample",
    "StateObservation",
    "TrackMode",
    "TransitionDecision",
    "TransitionReason",
]
