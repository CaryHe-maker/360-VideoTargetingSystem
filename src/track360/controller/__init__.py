"""Controller components for spherical RGB tracking."""

from track360.controller.decision_gate import DecisionGate, FrameAggregate, ScoredObservation
from track360.controller.fused_score import (
    MotionScore,
    calibrateBackendFusedScore,
    calibrateLocalAppearanceProbabilities,
    calibrateMotionScore,
    composeSingleScore,
    scoreMotionConsistency,
    scoreViewCenterMotion,
)
from track360.controller.fusor import (
    FUSION_AGREEMENT_BONUS_WEIGHT,
    FUSION_MAX_SCORE_GAIN,
    FUSION_OVERLAP_RATE,
    FUSION_SCORE_CAP,
    FusionBoxMode,
    Fusor,
    fuse,
)
from track360.controller.motion_estimator import MotionEstimatorImpl, SphericalMotionEstimator
from track360.controller.recovery_planner import PlannedView, RecoveryPlanner, ViewSpecType1
from track360.controller.score_calibration import (
    UNCALIBRATED_STAGE3_SCORE_CALIBRATION,
    BetaCalibration,
    ScoreCalibration,
    loadScoreCalibration,
)
from track360.controller.state_evaluator import StateEvaluator
from track360.controller.state_machine import StateUpdate, TrackStateMachine
from track360.controller.state_model import (
    EvaluatedCandidate,
    EvidenceLevel,
    MeasurementEvidence,
    MotionPrediction,
    RecoveryMemory,
    StateInstance,
    StateObservation,
    TrackMode,
)
from track360.controller.template_policy import TemplateDecision, TemplatePolicy
from track360.controller.track_controller import TrackControllerImpl

__all__ = [
    "DecisionGate",
    "FUSION_AGREEMENT_BONUS_WEIGHT",
    "FUSION_MAX_SCORE_GAIN",
    "FUSION_OVERLAP_RATE",
    "FUSION_SCORE_CAP",
    "FusionBoxMode",
    "Fusor",
    "MotionScore",
    "BetaCalibration",
    "FrameAggregate",
    "fuse",
    "MotionEstimatorImpl",
    "PlannedView",
    "RecoveryPlanner",
    "ScoreCalibration",
    "ViewSpecType1",
    "calibrateBackendFusedScore",
    "calibrateLocalAppearanceProbabilities",
    "calibrateMotionScore",
    "composeSingleScore",
    "UNCALIBRATED_STAGE3_SCORE_CALIBRATION",
    "ScoredObservation",
    "StateEvaluator",
    "StateInstance",
    "StateObservation",
    "EvidenceLevel",
    "EvaluatedCandidate",
    "MeasurementEvidence",
    "MotionPrediction",
    "RecoveryMemory",
    "SphericalMotionEstimator",
    "scoreMotionConsistency",
    "scoreViewCenterMotion",
    "loadScoreCalibration",
    "StateUpdate",
    "TemplateDecision",
    "TemplatePolicy",
    "TrackStateMachine",
    "TrackMode",
    "TrackControllerImpl",
]
