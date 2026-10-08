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
        backendScore = _singleScore(observation)
        motionScore = motionAgreement(
            observation.bfov,
            prediction,
            self._tuning.motionOffsetScale,
            self._tuning.motionSizeScale,
        )
        appearanceScore = (
            None
            if observation.appearanceSimilarity is None
            else float(np.clip(observation.appearanceSimilarity, 0.0, 1.0))
        )
        stateScore = fuseStateScore(backendScore, appearanceScore, motionScore, self._tuning)
        # ARTrack's score is a localization-quality signal, not a calibrated probability:
        # by default the returned box is the measurement whatever its raw score.
        accepted = (
            self._tuning.acceptAnyCandidate
            or backendScore >= self._tracking.candidateMinScore
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
            motionScore=motionScore,
            scaleScore=observation.scaleScore,
            stateScore=stateScore,
            measurementAccepted=accepted,
            appearanceScore=appearanceScore,
            uncertainThreshold=self._tuning.uncertainScore,
        )


def motionAgreement(
    measured: BFoV, prediction: MotionPrediction, offsetScale: float, sizeScale: float
) -> float:
    """How well a box agrees with the motion prediction, in [0, 1].

    1 when the box is where and as large as predicted.  It falls off with the angle
    between the two, counted in predicted target sizes, and with the log of the
    ratio of their sizes; ``offsetScale`` and ``sizeScale`` are the widths of the
    two fall-offs.  Without a predicted size only the position counts.
    """
    cosine = (
        measured.center.x * prediction.center.x
        + measured.center.y * prediction.center.y
        + measured.center.z * prediction.center.z
    )
    angle = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    horizontal, vertical = prediction.horizontalSizeRad, prediction.verticalSizeRad
    if horizontal <= 0.0 or vertical <= 0.0:
        size = float(np.sqrt(measured.horizontalFovRad * measured.verticalFovRad))
        return float(np.exp(-0.5 * (angle / size / offsetScale) ** 2))
    size = float(np.sqrt(horizontal * vertical))
    logScale = 0.5 * float(
        np.log(measured.horizontalFovRad * measured.verticalFovRad / (horizontal * vertical))
    )
    return float(
        np.exp(-0.5 * (angle / size / offsetScale) ** 2)
        * np.exp(-0.5 * (logScale / sizeScale) ** 2)
    )


def fuseStateScore(
    backendScore: float,
    appearanceScore: float | None,
    motionScore: float,
    tuning: BackendTuningConfig,
) -> float:
    """Weighted mean of the three scores; a missing appearance score is left out."""
    total = tuning.stateBackendWeight * backendScore + tuning.stateMotionWeight * motionScore
    weight = tuning.stateBackendWeight + tuning.stateMotionWeight
    if appearanceScore is not None:
        total += tuning.stateAppearanceWeight * appearanceScore
        weight += tuning.stateAppearanceWeight
    return float(np.clip(total / weight, 0.0, 1.0))


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


__all__ = ["StateEvaluator", "fuseStateScore", "motionAgreement"]
