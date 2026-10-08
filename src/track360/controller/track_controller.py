"""Single-view controller for spherical RGB tracking."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from math import asin, atan2, pi
from typing import TYPE_CHECKING

from track360.controller.motion_estimator import SphericalMotionEstimator
from track360.controller.state_evaluator import StateEvaluator
from track360.controller.state_machine import TrackStateMachine
from track360.controller.state_model import (
    MotionPrediction,
    StateObservation,
    TrackMode,
    TransitionDecision,
)
from track360.controller.template_policy import TemplateDecision, TemplatePolicy
from track360.controller.view_planner import TRAJECTORY_LENGTH, ViewPlanner
from track360.core.config import (
    AppConfig,
    BackendTuningConfig,
    GeometryConfig,
    MotionConfig,
    TrackingConfig,
)
from track360.core.errors import ProtocolError
from track360.core.protocols import SphericalGeometry
from track360.core.protocols import (
    TrackController as TrackControllerProtocol,
)
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    FramePacket,
    InitializationPlan,
    MotionState3D,
    ProjectedObservation,
    ResultSource,
    SearchPlan,
    TemplateCommand,
    TemplateCommandKind,
    TrackResult,
    TrackStatus,
)
from track360.geometry.projection_math import (
    makeSphericalPoint,
)

if TYPE_CHECKING:
    from track360.core.protocols import MotionEstimator


_MAX_HORIZONTAL_SIZE_RAD = 2.0 * pi - 1e-6
_MAX_VERTICAL_SIZE_RAD = pi - 1e-6


@dataclass(frozen=True, slots=True)
class _PendingFrame:
    frame: FramePacket
    plan: SearchPlan
    prediction: MotionPrediction
    predictedBfov: BFoV
    mode: TrackMode


class TrackControllerImpl(TrackControllerProtocol):
    """Single-writer controller: one search view per frame, one commit per frame."""

    def __init__(
        self,
        geometry: SphericalGeometry,
        config: AppConfig | None = None,
        *,
        geometryConfig: GeometryConfig | None = None,
        trackingConfig: TrackingConfig | None = None,
        motionConfig: MotionConfig | None = None,
        backendTuning: BackendTuningConfig | None = None,
        motionEstimator: MotionEstimator | None = None,
    ) -> None:
        if config is not None:
            geometryConfig = config.geometry
            trackingConfig = config.tracking
            motionConfig = config.motion
            backendTuning = config.backendTuning
        if geometryConfig is None or trackingConfig is None:
            raise ValueError("geometryConfig and trackingConfig are required")
        motionConfig = motionConfig or MotionConfig()
        backendTuning = backendTuning or BackendTuningConfig()
        self._backendTuning = backendTuning
        self._geometry = geometry
        self._geometryConfig = geometryConfig
        self._trackingConfig = trackingConfig
        self._motionMinSamples = motionConfig.minSamplesForVelocity
        self._motion: MotionEstimator = motionEstimator or SphericalMotionEstimator(
            windowLength=trackingConfig.windowLength,
            maxPredictionHorizon=trackingConfig.maxPredictionHorizon,
            minSamplesForVelocity=motionConfig.minSamplesForVelocity,
            maxTangentSpanRad=motionConfig.maxTangentSpanRad,
            huberDeltaRad=motionConfig.huberDeltaRad,
            processNoiseRadPerSec=motionConfig.processNoiseRadPerSec,
            maxAngularSpeedRadPerSec=motionConfig.maxAngularSpeedRadPerSec,
            maxLogScaleRatePerSec=motionConfig.maxLogScaleRatePerSec,
        )
        self._evaluator = StateEvaluator(trackingConfig, backendTuning)
        self._planner = ViewPlanner(geometryConfig, trackingConfig, backendTuning)
        self._stateMachine = TrackStateMachine(trackingConfig)
        self._templatePolicy = TemplatePolicy(trackingConfig, backendTuning)

        self._initialized = False
        self._sequenceId: str | None = None
        self._lastFrameIndex = -1
        self._stateRevision = -1
        self._backendRevision = 0
        self._mode = TrackMode.INIT
        self._stableFrames = 0
        self._lastFrame: FramePacket | None = None
        self._initialBox: BBoxXYWH | None = None
        self._currentBox: BBoxXYWH | None = None
        self._currentBfov: BFoV | None = None
        # The target's BFoV after each of the last frames, oldest first.
        self._trajectory: deque[BFoV] = deque(maxlen=TRAJECTORY_LENGTH)
        self._pendingTemplate = TemplateDecision(TemplateCommandKind.KEEP)
        self._pending: _PendingFrame | None = None
        self._initialPlan: InitializationPlan | None = None
        self._lastStateObservation: StateObservation | None = None
        self._lastTransition: TransitionDecision | None = None
        self._lastPipelineProfile: dict[str, object] = {}

    @property
    def status(self) -> TrackStatus | None:
        if not self._initialized:
            return None
        return _publicStatus(self._mode)

    @property
    def stateRevision(self) -> int:
        return max(0, self._stateRevision)

    @property
    def lastStateObservation(self) -> StateObservation | None:
        return self._lastStateObservation

    @property
    def lastTransition(self) -> TransitionDecision | None:
        return self._lastTransition

    @property
    def lastPipelineProfile(self) -> dict[str, object]:
        return dict(self._lastPipelineProfile)

    def buildInitialization(
        self,
        frame: FramePacket,
        initialBox: BBoxXYWH | None = None,
        *,
        initialBfov: BFoV | None = None,
    ) -> InitializationPlan:
        """Plan the template crop from the frame-0 target, given as an ERP box or a BFoV."""
        if self._initialized or self._initialPlan is not None:
            raise ProtocolError("controller is already initialized or has a pending initialization")
        if int(frame.frameIndex) != 0:
            raise ProtocolError("initialization must use frameIndex 0")
        if (initialBox is None) == (initialBfov is None):
            raise ProtocolError("initialization requires exactly one of initialBox or initialBfov")
        frameWidthPx, frameHeightPx = frame.rgb.shape[1], frame.rgb.shape[0]
        if initialBfov is not None:
            objectBfov = initialBfov
            initialBox = self._geometry.bfovToBbox(initialBfov, frameWidthPx, frameHeightPx)
        else:
            assert initialBox is not None
            objectBfov = self._geometry.bboxToBfov(initialBox, frameWidthPx, frameHeightPx)
        templateView = self._planner.templateView(objectBfov)
        assert templateView.priorBox is not None
        plan = InitializationPlan(
            sequenceId=frame.sequenceId,
            frameIndex=FrameIndex(0),
            stateRevision=0,
            templateView=replace(templateView, priorBox=None),
            templateBox=templateView.priorBox,
        )
        self._initialPlan = plan
        self._lastFrame = frame
        self._sequenceId = str(frame.sequenceId)
        self._initialBox = initialBox
        self._currentBox = initialBox
        self._currentBfov = objectBfov
        return plan

    def commitInitialization(
        self,
        plan: InitializationPlan,
    ) -> TrackResult:
        if self._initialPlan != plan:
            raise ProtocolError("initialization response does not match the pending plan")
        if self._lastFrame is None or self._initialBox is None or self._currentBfov is None:
            raise ProtocolError("initialization frame state is incomplete")
        if hasattr(self._motion, "resetFromMeasurement"):
            self._motion.resetFromMeasurement(  # type: ignore[attr-defined]
                self._currentBfov.center,
                self._lastFrame.timestampNs,
                0,
                1.0,
                self._currentBfov.horizontalFovRad,
                self._currentBfov.verticalFovRad,
            )
        else:
            self._motion.initialize(
                self._currentBfov.center,
                self._lastFrame.timestampNs,
            )
        self._stateMachine.initialize()
        self._trajectory.clear()
        self._trajectory.extend([self._currentBfov] * TRAJECTORY_LENGTH)
        self._initialized = True
        self._initialPlan = None
        self._stateRevision = 0
        self._lastFrameIndex = 0
        self._mode = TrackMode.TRACKING
        return TrackResult(
            sequenceId=self._lastFrame.sequenceId,
            frameIndex=FrameIndex(0),
            bbox=self._initialBox,
            bfov=self._currentBfov,
            confidence=1.0,
            status=TrackStatus.TRACKING,
            valid=True,
            resultSource=ResultSource.INITIAL,
        )

    def beginFrame(self, frame: FramePacket) -> SearchPlan:
        """Plan the single search view of ``frame`` around the predicted target."""
        self._requireInitialized()
        if self._pending is not None:
            raise ProtocolError("a frame is already awaiting its observation")
        self._requireFrameOrder(frame)
        if self._currentBox is None or self._currentBfov is None:
            raise ProtocolError("controller target state is incomplete")
        prediction = self._predictDetailed(frame)
        predictedBfov = self._planner.contextBfov(
            prediction.center,
            frame.rgb.shape[1],
            frame.rgb.shape[0],
            self._initialBox or self._currentBox,
            self._currentBox,
            prediction.angularUncertaintyRad,
        )
        # Size the view from the predicted target; before the estimator exposes an
        # angular scale, the last committed BFoV is the basis.
        targetBfov = self._currentBfov
        viewCenter = prediction.center
        if not self._backendTuning.predictiveSearch:
            # The view follows the last committed target, as the tracker's own loop does;
            # the motion estimate is left to the backend's trajectory input.
            viewCenter = self._currentBfov.center
        elif prediction.horizontalSizeRad > 0.0 and prediction.verticalSizeRad > 0.0:
            # Extrapolating a growing target can leave the range a BFoV can express.
            targetBfov = BFoV(
                prediction.center,
                min(prediction.horizontalSizeRad, _MAX_HORIZONTAL_SIZE_RAD),
                min(prediction.verticalSizeRad, _MAX_VERTICAL_SIZE_RAD),
            )
        plan = SearchPlan(
            sequenceId=frame.sequenceId,
            frameIndex=frame.frameIndex,
            stateRevision=self._stateRevision + 1,
            view=self._planner.searchView(
                viewCenter,
                targetBfov.horizontalFovRad,
                targetBfov.verticalFovRad,
                tuple(self._trajectory),
            ),
            templateCommand=TemplateCommand(
                kind=self._pendingTemplate.kind,
                frameIndex=frame.frameIndex,
                viewId=self._pendingTemplate.viewId,
                localBox=self._pendingTemplate.localBox,
                expectedRevision=self._backendRevision + 1,
            ),
            predictedMotion=prediction.motionState,
        )
        self._pending = _PendingFrame(
            frame=frame,
            plan=plan,
            prediction=prediction,
            predictedBfov=predictedBfov,
            mode=self._mode,
        )
        self._pendingTemplate = TemplateDecision(TemplateCommandKind.KEEP)
        return plan

    def consume(
        self,
        plan: SearchPlan,
        observation: ProjectedObservation | None,
    ) -> TrackResult:
        """Commit the frame from the observation of its search view (``None``: no box)."""
        self._requireInitialized()
        pending = self._pending
        if pending is None or pending.plan != plan:
            raise ProtocolError("search response does not match the pending plan")
        self._backendRevision = plan.templateCommand.expectedRevision
        evaluation = self._evaluator.evaluate(
            mode=pending.mode,
            plan=plan,
            observation=observation,
            prediction=pending.prediction,
            predictedBfov=pending.predictedBfov,
            geometry=self._geometry,
            frameWidthPx=pending.frame.rgb.shape[1],
            frameHeightPx=pending.frame.rgb.shape[0],
        )
        thresholds = self._stateMachine.scoreGroup.thresholds()
        if thresholds is not None:
            evaluation = replace(
                evaluation,
                uncertainThreshold=thresholds[0],
                lostThreshold=thresholds[1],
            )
        decision = self._stateMachine.transition(
            pending.mode,
            evaluation.stateScore,
            measurementAccepted=evaluation.measurementAccepted,
        )
        result = self._commit(pending, evaluation, decision)
        self._lastPipelineProfile = {
            "frameIndex": int(plan.frameIndex),
            "finalMeasurementAccepted": bool(decision.acceptMeasurement),
            "finalStateRevision": int(plan.stateRevision),
        }
        self._stateMachine.recordScore(evaluation.stateScore)
        self._lastStateObservation = evaluation
        self._lastTransition = decision
        return result

    def commitFallback(
        self,
        frame: FramePacket,
        *,
        backendRevision: int | None = None,
        reason: str = "frame_error",
    ) -> TrackResult:
        """Advance past one failed frame and emit an invalid, zero-scored result.

        Result files must contain one output per input frame. This method
        keeps the last confirmed target state, clears the pending plan, and
        advances protocol revisions so tracking can resume.
        """
        self._requireInitialized()
        if self._sequenceId != str(frame.sequenceId):
            raise ProtocolError("fallback frame sequence does not match controller state")
        expectedFrameIndex = self._lastFrameIndex + 1
        if int(frame.frameIndex) != expectedFrameIndex:
            raise ProtocolError(
                "fallback frame order mismatch: "
                f"expected={expectedFrameIndex}, actual={int(frame.frameIndex)}"
            )
        if self._currentBox is None or self._currentBfov is None:
            raise ProtocolError("fallback frame requires a confirmed target state")

        plannedRevision = (
            self._pending.plan.stateRevision
            if self._pending is not None
            else self._stateRevision + 1
        )
        self._stateRevision = max(self._stateRevision + 1, plannedRevision)
        if backendRevision is not None:
            self._backendRevision = max(self._backendRevision, int(backendRevision))
        self._lastFrameIndex = int(frame.frameIndex)
        self._lastFrame = frame
        self._pending = None
        self._pendingTemplate = TemplateDecision(TemplateCommandKind.KEEP)
        self._stableFrames = 0
        self._lastStateObservation = None
        self._lastTransition = None
        self._lastPipelineProfile = {
            "pipelineFrameFallback": True,
            "frameIndex": int(frame.frameIndex),
            "reason": reason,
            "finalStateRevision": int(self._stateRevision),
        }
        return TrackResult(
            sequenceId=frame.sequenceId,
            frameIndex=frame.frameIndex,
            bbox=self._currentBox,
            bfov=self._currentBfov,
            confidence=0.0,
            status=_publicStatus(self._mode),
            valid=False,
            resultSource=ResultSource.MOTION_PREDICTED,
        )

    def _commit(
        self,
        pending: _PendingFrame,
        evaluation: StateObservation,
        decision: TransitionDecision,
    ) -> TrackResult:
        hasCandidate = evaluation.hasCandidate
        accepted = decision.acceptMeasurement and hasCandidate
        holdingWeak = False
        outputBfov = evaluation.proposedOutputBfov
        outputBox = evaluation.proposedOutputBbox
        outputConfidence = (
            evaluation.stateScore if hasCandidate else max(0.0, pending.prediction.confidence * 0.5)
        )
        if accepted:
            self._recordMeasurement(pending, outputBfov, outputConfidence)
            source = ResultSource.OBSERVED_CONFIRMED
            self._currentBox = outputBox
            self._currentBfov = outputBfov
        else:
            source = (
                ResultSource.OBSERVED_WEAK_BLEND if hasCandidate else ResultSource.MOTION_PREDICTED
            )
            assert self._currentBox is not None and self._currentBfov is not None
            currentArea = self._currentBox.widthPx * self._currentBox.heightPx
            frameArea = float(pending.frame.rgb.shape[1] * pending.frame.rgb.shape[0])
            if self._backendTuning.holdWeakBox and currentArea >= 0.10 * frameArea:
                outputBox = self._currentBox
                outputBfov = self._currentBfov
                holdingWeak = True
            # Break the motion/acceptance bootstrap cycle without committing a weak box as the
            # public target state.  A single bounded provisional observation supplies the second
            # timestamped point required for velocity fitting; subsequent weak frames do not
            # continue polluting the history.
            if (
                evaluation.measuredBfov is not None
                and pending.prediction.sampleCount < self._motionMinSamples
                and pending.mode in {TrackMode.TRACKING, TrackMode.UNCERTAIN}
            ):
                self._recordMeasurement(
                    pending,
                    evaluation.measuredBfov,
                    max(self._trackingConfig.candidateMinScore, evaluation.stateScore),
                )
        self._mode = decision.nextMode
        if accepted and self._mode is TrackMode.TRACKING:
            self._stableFrames += 1
        else:
            self._stableFrames = 0
        self._pendingTemplate = self._templatePolicy.decide(
            _publicStatus(self._mode),
            self._stableFrames,
            evaluation,
        )
        assert self._currentBfov is not None
        self._trajectory.append(self._currentBfov)
        self._stateRevision = pending.plan.stateRevision
        self._lastFrameIndex = int(pending.frame.frameIndex)
        self._lastFrame = pending.frame
        self._pending = None
        return TrackResult(
            sequenceId=pending.frame.sequenceId,
            frameIndex=pending.frame.frameIndex,
            bbox=outputBox,
            bfov=outputBfov,
            confidence=outputConfidence,
            status=_publicStatus(self._mode),
            valid=accepted or holdingWeak,
            resultSource=source,
        )

    def _recordMeasurement(
        self, pending: _PendingFrame, bfov: BFoV, confidence: float
    ) -> None:
        if hasattr(self._motion, "recordMeasurement"):
            self._motion.recordMeasurement(  # type: ignore[attr-defined]
                frameIndex=int(pending.frame.frameIndex),
                timestampNs=pending.frame.timestampNs,
                point=bfov.center,
                confidence=confidence,
                horizontalSizeRad=bfov.horizontalFovRad,
                verticalSizeRad=bfov.verticalFovRad,
            )
        else:
            self._motion.update(bfov.center, pending.frame.timestampNs, confidence)

    def _predictDetailed(self, frame: FramePacket) -> MotionPrediction:
        if hasattr(self._motion, "predictDetailed"):
            prediction = self._motion.predictDetailed(  # type: ignore[attr-defined]
                frame.timestampNs,
                1,
            )
            return replace(
                prediction,
                sourceRevision=max(0, self._stateRevision),
                targetFrameIndex=frame.frameIndex,
            )
        motion = self._motion.predict(frame.timestampNs)
        center = _motionCenter(motion)
        return MotionPrediction(
            sourceRevision=max(0, self._stateRevision),
            targetFrameIndex=frame.frameIndex,
            horizonFrames=1,
            center=center,
            horizontalSizeRad=self._currentBfov.horizontalFovRad,
            verticalSizeRad=self._currentBfov.verticalFovRad,
            tangentVelocityRadPerSec=(0.0, 0.0),
            angularUncertaintyRad=0.05,
            scaleUncertainty=0.10,
            confidence=motion.confidence,
            centerCovarianceRad2=motion.centerCovarianceRad2,
            scaleCovarianceLog2=motion.scaleCovarianceLog2,
            reliability=motion.reliability,
        )

    def _requireInitialized(self) -> None:
        if not self._initialized:
            raise ProtocolError("controller has not been initialized")

    def _requireFrameOrder(self, frame: FramePacket) -> None:
        if self._sequenceId != str(frame.sequenceId):
            raise ProtocolError("frame sequence does not match controller sequence")
        if int(frame.frameIndex) != self._lastFrameIndex + 1:
            raise ProtocolError(
                f"frame index must be {self._lastFrameIndex + 1}, actual={frame.frameIndex}"
            )


def _motionCenter(motion: MotionState3D):
    x, y, z = motion.position
    return makeSphericalPoint(atan2(x, z), asin(max(-1.0, min(1.0, y))))


def _publicStatus(mode: TrackMode) -> TrackStatus:
    if mode is TrackMode.UNCERTAIN:
        return TrackStatus.UNCERTAIN
    if mode is TrackMode.LOST:
        return TrackStatus.LOST
    return TrackStatus.TRACKING


__all__ = ["TrackControllerImpl"]
