"""Immutable state records and bounded controller transaction data for V2."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto

from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    MotionState3D,
    ProjectedObservation,
    ResultSource,
    SequenceId,
    SphericalPoint,
    TrackStatus,
)


class TrackMode(Enum):
    INIT = auto()
    TRACKING = auto()
    UNCERTAIN = auto()
    LOST = auto()
    TERMINATED = auto()


class AttemptKind(Enum):
    PRIMARY = auto()
    ESCALATION = auto()


class EvidenceLevel(Enum):
    CONFIRMED = auto()
    WEAK = auto()
    REJECTED = auto()
    REACQUIRED = auto()


class MeasurementEvidence(Enum):
    RELIABLE_FUSED = auto()
    RELIABLE_SINGLE = auto()
    WEAK = auto()
    MISSING = auto()


class EvaluationReason(StrEnum):
    NO_ELIGIBLE_CLUSTER = "no_eligible_cluster"
    INSUFFICIENT_VIEW_SUPPORT = "insufficient_view_support"
    BELOW_UNCERTAIN_THRESHOLD = "below_uncertain_threshold"
    SOURCE_CONFIDENCE_BELOW_THRESHOLD = "source_confidence_below_threshold"


class TransitionReason(Enum):
    INITIALIZED = auto()
    RELIABLE_MEASUREMENT = auto()
    WEAK_MEASUREMENT = auto()
    HARD_MISS = auto()
    PATIENCE_EXHAUSTED = auto()
    RECOVERY_PROGRESS = auto()
    REACQUIRED = auto()
    RECOVERY_EXHAUSTED = auto()
    END_OF_STREAM = auto()
    EXTERNAL_RESET = auto()


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
class ConfirmedTargetState:
    frameIndex: FrameIndex
    bbox: BBoxXYWH
    bfov: BFoV
    confidence: float


@dataclass(frozen=True, slots=True)
class StateInstance:
    stateId: int
    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    mode: TrackMode
    enteredFrom: TrackMode | None
    entryReason: TransitionReason
    prediction: MotionPrediction | None
    searchSeedCenter: SphericalPoint | None
    recoveryEpochId: int
    modeAgeFrames: int
    stableStreak: int
    weakStreak: int
    missStreak: int

    @property
    def publicStatus(self) -> TrackStatus:
        if self.mode is TrackMode.LOST:
            return TrackStatus.LOST
        if self.mode is TrackMode.UNCERTAIN:
            return TrackStatus.UNCERTAIN
        return TrackStatus.TRACKING


@dataclass(frozen=True, slots=True)
class EvaluatedCandidate:
    bfov: BFoV
    bbox: BBoxXYWH
    confidence: float
    sourceViewIds: tuple[int, ...]
    fused: bool
    overlapRate: float | None
    minSourceConfidence: float | None
    sourceConfidencePassed: bool
    representativeViewId: int
    representativeLocalBox: BBoxXYWH | None


@dataclass(frozen=True, slots=True)
class StateObservation:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    transactionId: int
    stateId: int
    attemptIndex: int
    evaluatedMode: TrackMode
    isFinalAttempt: bool
    appearanceOnlyScoring: bool
    successRate: float
    fusionThreshold: float
    overlapThreshold: float
    fusionSourceMinConfidence: float
    bestCandidate: EvaluatedCandidate | None
    predictedCenter: SphericalPoint
    searchSeedCenter: SphericalPoint
    measuredBfov: BFoV | None
    measuredBbox: BBoxXYWH | None
    measuredCenter: SphericalPoint | None
    proposedOutputBfov: BFoV
    proposedOutputBbox: BBoxXYWH
    proposedResultSource: ResultSource
    candidateCount: int
    eligibleCandidateCount: int
    clusterCount: int
    sourceViewIds: tuple[int, ...]
    representativeViewId: int | None
    representativeLocalBox: BBoxXYWH | None
    selectedIsFused: bool
    selectedOverlapRate: float | None
    selectedMinSourceConfidence: float | None
    selectedSourceConfidencePassed: bool
    fusedCandidateCount: int
    outputEligible: bool
    supportViewCount: int
    backendScore: float
    motionScore: float
    scaleScore: float
    supportScore: float
    agreementScore: float
    stateScore: float
    evidence: MeasurementEvidence
    hardGatePassed: bool
    supported: bool
    escalationRecommended: bool
    reacquired: bool
    rejectionReasons: tuple[EvaluationReason, ...] = ()
    rawMotionScore: float | None = None
    motionProbability: float | None = None
    motionReliability: float = 0.0
    motionSampleCount: int = 0
    motionDegradedReasons: tuple[str, ...] = ()
    uncertainThreshold: float = 0.0
    lostThreshold: float = 0.0
    measurementAccepted: bool = False

    @property
    def resultSource(self) -> ResultSource:
        """Compatibility alias for diagnostics written before the V2 field was explicit."""
        return self.proposedResultSource


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    kind: AttemptKind
    attemptIndex: int
    plan: object
    observations: tuple[ProjectedObservation, ...]
    evaluation: StateObservation


@dataclass(slots=True)
class RecoveryMemory:
    epochId: int = 0
    startedFrameIndex: FrameIndex | None = None
    framesSpent: int = 0
    globalScanPhase: int = 0
    coveredCells: set[tuple[int, int]] = field(default_factory=set)
    attemptedPlanKeys: set[tuple[int, int, int, int, int, int]] = field(default_factory=set)
    bestSeedCenter: SphericalPoint | None = None
    bestSeedScore: float = 0.0
    bestSeedFrameIndex: FrameIndex | None = None
    lastGlobalScanFrameIndex: FrameIndex | None = None

    def reset(self, frameIndex: FrameIndex) -> None:
        self.epochId += 1
        self.startedFrameIndex = frameIndex
        self.framesSpent = 0
        self.globalScanPhase = 0
        self.coveredCells.clear()
        self.attemptedPlanKeys.clear()
        self.bestSeedCenter = None
        self.bestSeedScore = 0.0
        self.bestSeedFrameIndex = None
        self.lastGlobalScanFrameIndex = None


@dataclass(slots=True)
class FrameTransaction:
    transactionId: int
    frame: object
    state: StateInstance
    startingMode: TrackMode
    attemptIndex: int = 0
    completedAttempts: int = 0
    remainingViews: int = 0
    attempts: list[AttemptRecord] = field(default_factory=list)
    recoveryMemory: RecoveryMemory | None = None
    refinementCenters: tuple[SphericalPoint, ...] = ()
    provisionalPrediction: MotionPrediction | None = None
    provisionalPredictionRevision: int = 0


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
    resetMotionHistory: bool = False
    resetRecoveryEpoch: bool = False


def newMotionHistory(maxlen: int) -> deque[MotionSample]:
    return deque(maxlen=maxlen)


__all__ = [
    "AttemptKind",
    "AttemptRecord",
    "ConfirmedTargetState",
    "EvaluatedCandidate",
    "EvidenceLevel",
    "EvaluationReason",
    "FrameTransaction",
    "MeasurementEvidence",
    "MotionPrediction",
    "MotionSample",
    "RecoveryMemory",
    "ScoreGroup",
    "StateInstance",
    "StateObservation",
    "TrackMode",
    "TransitionDecision",
    "TransitionReason",
    "newMotionHistory",
]
