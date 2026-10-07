"""Turn the observation of a frame's search view into a measurement decision."""

from __future__ import annotations

import numpy as np

from track360.controller.state_model import MotionPrediction, StateObservation, TrackMode
from track360.core.config import BackendTuningConfig, TrackingConfig
from track360.core.errors import ProtocolError
from track360.core.protocols import SphericalGeometry
from track360.core.types import BFoV, ProjectedObservation, SearchPlan


class StateEvaluator:
    """Decide whether the observation is accepted as this frame's measurement.

    The evaluator owns measurement eligibility.  The state machine receives only the
    resulting StateScore for next-state selection.
    """

    def __init__(
        self,
        trackingConfig: TrackingConfig,
        backendTuning: BackendTuningConfig | None = None,
    ) -> None:
        self._tracking = trackingConfig
        self._tuning = backendTuning or BackendTuningConfig()

    def evaluate(
        self,
        *,
        mode: TrackMode,
        plan: SearchPlan,
        observation: ProjectedObservation | None,
        prediction: MotionPrediction,
        predictedBfov: BFoV,
        geometry: SphericalGeometry,
        frameWidthPx: int,
        frameHeightPx: int,
    ) -> StateObservation:
        if observation is None:
            return StateObservation(
                sequenceId=plan.sequenceId,
                frameIndex=plan.frameIndex,
                stateRevision=plan.stateRevision,
                evaluatedMode=mode,
                predictedCenter=prediction.center,
                viewId=None,
                localBox=None,
                measuredBfov=None,
                measuredBbox=None,
                proposedOutputBfov=predictedBfov,
                proposedOutputBbox=geometry.bfovToBbox(predictedBfov, frameWidthPx, frameHeightPx),
                backendScore=0.0,
                motionScore=0.0,
                scaleScore=0.0,
                stateScore=0.0,
                measurementAccepted=False,
            )
        if observation.viewId != plan.view.viewId:
            raise ProtocolError("projected observation does not belong to the planned view")
        stateScore = _singleScore(observation)
        # ARTrack's score is a localization-quality signal, not a calibrated probability:
        # by default the returned box is the measurement whatever its raw score.
        accepted = (
            self._tuning.acceptAnyCandidate or stateScore >= self._tracking.candidateMinScore
        )
        return StateObservation(
            sequenceId=plan.sequenceId,
            frameIndex=plan.frameIndex,
            stateRevision=plan.stateRevision,
            evaluatedMode=mode,
            predictedCenter=prediction.center,
            viewId=observation.viewId,
            localBox=observation.localBox,
            measuredBfov=observation.bfov,
            measuredBbox=observation.bbox,
            proposedOutputBfov=observation.bfov,
            proposedOutputBbox=observation.bbox,
            backendScore=float(
                observation.backendFusedScore
                if observation.backendFusedScore is not None
                else observation.fusedScore
            ),
            motionScore=observation.motionScore,
            scaleScore=observation.scaleScore,
            stateScore=stateScore,
            measurementAccepted=accepted,
        )


def _singleScore(observation: ProjectedObservation) -> float:
    return float(
        np.clip(
            observation.singleScore
            if observation.singleScore is not None
            else observation.fusedScore,
            0.0,
            1.0,
        )
    )


__all__ = ["StateEvaluator"]
