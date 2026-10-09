"""Single-view controller for spherical RGB tracking."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import asin, atan2, degrees, pi, sqrt
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
from track360.controller.view_planner import (
    REFINE_VIEW_ID_BASE,
    SCAN_VIEW_ID_BASE,
    TRAJECTORY_LENGTH,
    ViewPlanner,
)
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
    SphericalPoint,
    TemplateCommand,
    TemplateCommandKind,
    TrackResult,
    TrackStatus,
    ViewSpec,
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
        self._stateMachine = TrackStateMachine(backendTuning)
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
        self._lastPipelineProfile: dict[str, object] = {}
        # Loss handling: where the target was last trusted, and how far the scan
        # around that place has got.
        self._lastGoodBfov: BFoV | None = None
        self._scanCursor = 0
        self._lastFrameSuspect = False
        self._lastFrameReacquired = False
        self._suspectFrameCount = 0
        self._scanFrameCount = 0
        self._reacquiredFrames: list[int] = []
        # What a lost track does: nothing (the state is only judged) or jump to a candidate.
        self._actions = backendTuning.lossActions if backendTuning.lossHandling else "none"
        # Scan views that may still be spent; starts full.
        self._scanTokens = backendTuning.scanBudgetBurst
        # Frames planned since the track was last declared lost, and what the last
        # plan's scan was (for the state trace).
        self._lostFrames = 0
        self._lastScan = ""
        self._lastFrameTrace: dict[str, object] = {}

    @property
    def lossStatistics(self) -> dict[str, object]:
        """What loss handling did over the sequence so far."""
        return {
            "suspectFrames": self._suspectFrameCount,
            "scanFrames": self._scanFrameCount,
            "reacquiredAt": list(self._reacquiredFrames),
        }

    @property
    def lastFrameTrace(self) -> dict[str, object]:
        """Scores, state and decisions of the last frame, for the state trace."""
        return dict(self._lastFrameTrace)

    @property
    def lastFrameSuspect(self) -> bool:
        """The last frame's box was doubted: the tracker should not learn from it."""
        return self._lastFrameSuspect

    @property
    def lastFrameReacquired(self) -> bool:
        """The last frame jumped to a scan candidate: the tracker restarts from it."""
        return self._lastFrameReacquired

    @property
    def suspectFrames(self) -> int:
        return self._stateMachine.untrustedFrames

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
        self._lastGoodBfov = self._currentBfov
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
        mainView = self._planner.searchView(
            viewCenter,
            targetBfov.horizontalFovRad,
            targetBfov.verticalFovRad,
            tuple(self._trajectory),
        )
        scanViews: tuple[ViewSpec, ...] = ()
        scanRefine = False
        self._lastScan = ""
        lost = self._mode is TrackMode.LOST
        lostFrames = self._lostFrames if lost else 0
        self._lostFrames = lostFrames + 1 if lost else 0
        scanCount = self._backendTuning.scanViewsPerFrame
        if self._backendTuning.scanBudgetPerFrame > 0.0:
            # Every frame earns a share of a scan view; a lost track spends the whole
            # ones it has saved, so the average cost per frame stays bounded.
            self._scanTokens = min(
                self._backendTuning.scanBudgetBurst,
                self._scanTokens + self._backendTuning.scanBudgetPerFrame,
            )
            scanCount = min(scanCount, int(self._scanTokens))
        searching = self._actions != "none" and lost and self._lastGoodBfov is not None
        if searching and self._backendTuning.scanMode == "zoom":
            scanViews, scanRefine = self._zoomScan(mainView, lostFrames, scanCount)
            if scanViews:
                self._scanTokens -= 2 if scanRefine else 1
                self._scanFrameCount += 1
        elif (
            self._actions != "none"
            and self._mode is TrackMode.LOST
            and self._lastGoodBfov is not None
            and scanCount > 0
        ):
            scanViews, self._scanCursor = self._planner.scanViews(
                self._lastGoodBfov, self._scanCursor, scanCount
            )
            self._scanTokens -= len(scanViews)
            self._scanFrameCount += 1
            self._lastScan = "tiles"
        plan = SearchPlan(
            sequenceId=frame.sequenceId,
            frameIndex=frame.frameIndex,
            stateRevision=self._stateRevision + 1,
            view=mainView,
            templateCommand=TemplateCommand(
                kind=self._pendingTemplate.kind,
                frameIndex=frame.frameIndex,
                viewId=self._pendingTemplate.viewId,
                localBox=self._pendingTemplate.localBox,
                expectedRevision=self._backendRevision + 1,
            ),
            predictedMotion=prediction.motionState,
            scanViews=scanViews,
            scanRefine=scanRefine,
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

    def _zoomScan(
        self, mainView: ViewSpec, lostFrames: int, budget: int
    ) -> tuple[tuple[ViewSpec, ...], bool]:
        """The scan of a lost frame under ``scanMode: zoom``, and whether it is refined.

        The first lost frame looks again at the view the tracker is in, without
        the tracker's memory.  Later frames take one enlarged view, the larger the
        longer the track has been lost; the box found there is not a candidate
        itself but the centre of a view of the normal size (see ``refinementView``).
        """
        tuning = self._backendTuning
        trusted = self._lastGoodBfov
        assert trusted is not None and self._currentBfov is not None
        if lostFrames == 0 and tuning.zoomInPlace:
            if budget < 1:
                return (), False
            self._lastScan = "1x"
            return (replace(mainView, viewId=SCAN_VIEW_ID_BASE, trajectory=()),), False
        if budget < 2:
            return (), False
        if lostFrames >= tuning.zoomLastAfterFrames:
            scale = tuning.zoomLastScale
        elif tuning.zoomMidScale > 0.0 and lostFrames >= tuning.zoomMidAfterFrames:
            scale = tuning.zoomMidScale
        else:
            scale = tuning.zoomFirstScale
        centre = trusted.center if tuning.zoomCentre == "trusted" else self._currentBfov.center
        self._lastScan = f"{scale:g}x"
        view = self._planner.probeView(
            centre,
            trusted.horizontalFovRad,
            trusted.verticalFovRad,
            scale,
            SCAN_VIEW_ID_BASE,
        )
        return (view,), True

    def refinementView(self, center: SphericalPoint, index: int = 0) -> ViewSpec:
        """A view of the normal size around a place an enlarged scan view pointed at."""
        trusted = self._lastGoodBfov or self._currentBfov
        assert trusted is not None
        return self._planner.probeView(
            center,
            trusted.horizontalFovRad,
            trusted.verticalFovRad,
            1.0,
            REFINE_VIEW_ID_BASE + index,
        )

    def consume(
        self,
        plan: SearchPlan,
        observation: ProjectedObservation | None,
        candidates: Sequence[ProjectedObservation] = (),
    ) -> TrackResult:
        """Commit the frame from the observation of its search view (``None``: no box).

        ``candidates`` are the boxes found in the plan's scan views.  When one of
        them looks like the template clearly more than the tracked box does, the
        track jumps to it.
        """
        self._requireInitialized()
        pending = self._pending
        if pending is None or pending.plan != plan:
            raise ProtocolError("search response does not match the pending plan")
        self._lastFrameSuspect = False
        self._lastFrameReacquired = False
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
        trace = self._traceOf(pending, evaluation)
        if self._actions != "none":
            chosen, trace["candidates"] = self._reacquisitionCandidate(
                observation, candidates
            )
            if chosen is not None:
                result = self._reacquire(pending, chosen)
                trace.update(
                    modeAfter=self._mode.name, reason="JUMP", action="jump", **_place(result.bfov)
                )
                self._lastFrameTrace = trace
                return result
        self._backendRevision = plan.templateCommand.expectedRevision
        decision = self._stateMachine.transition(
            pending.mode,
            evaluation.stateScore,
            measurementAccepted=evaluation.measurementAccepted,
            backendScore=evaluation.backendScore if evaluation.hasCandidate else None,
            appearanceScore=evaluation.appearanceScore,
        )
        result = self._commit(pending, evaluation, decision)
        trace.update(
            modeAfter=decision.nextMode.name,
            reason=decision.reason.name,
            untrusted=self._stateMachine.untrustedFrames,
            calm=self._stateMachine.calmFrames,
            relative=self._stateMachine.relativeSignal,
            **_place(result.bfov),
        )
        self._lastPipelineProfile = {
            "frameIndex": int(plan.frameIndex),
            "finalMeasurementAccepted": bool(decision.acceptMeasurement),
            "finalStateRevision": int(plan.stateRevision),
        }
        self._lastStateObservation = evaluation
        if self._backendTuning.lossHandling:
            if decision.nextMode is TrackMode.TRACKING:
                self._scanCursor = 0
                self._lastGoodBfov = self._currentBfov
            else:
                self._suspectFrameCount += 1
                # Without actions the run must stay what it is without loss handling.
                self._lastFrameSuspect = self._actions != "none"
        self._lastFrameTrace = trace
        return result

    def _traceOf(self, pending: _PendingFrame, evaluation: StateObservation) -> dict[str, object]:
        return {
            "frame": int(pending.frame.frameIndex),
            "modeBefore": pending.mode.name,
            "modeAfter": pending.mode.name,
            "reason": "",
            "action": "",
            "hasBox": int(evaluation.hasCandidate),
            "backend": evaluation.backendScore,
            "appearance": evaluation.appearanceScore,
            "motion": evaluation.motionScore,
            "motionOffset": evaluation.motionOffset,
            "motionLogScale": evaluation.motionLogScale,
            "stateScore": evaluation.stateScore,
            "untrusted": self._stateMachine.untrustedFrames,
            "calm": self._stateMachine.calmFrames,
            "scanViews": len(pending.plan.scanViews),
            "scan": self._lastScan,
            "relative": None,
            "candidates": [],
        }

    def _restartMotion(self, bfov: BFoV, frame: FramePacket) -> None:
        if hasattr(self._motion, "resetFromMeasurement"):
            self._motion.resetFromMeasurement(  # type: ignore[attr-defined]
                bfov.center,
                frame.timestampNs,
                int(frame.frameIndex),
                1.0,
                bfov.horizontalFovRad,
                bfov.verticalFovRad,
            )
        else:
            self._motion.initialize(bfov.center, frame.timestampNs)

    def _reacquisitionCandidate(
        self,
        observation: ProjectedObservation | None,
        candidates: Sequence[ProjectedObservation],
    ) -> tuple[ProjectedObservation | None, list[dict[str, object]]]:
        """The scan candidate to jump to, if any, and why each one was or was not it."""
        tuning = self._backendTuning
        current = -1.0
        if observation is not None and observation.appearanceSimilarity is not None:
            current = observation.appearanceSimilarity
        best: ProjectedObservation | None = None
        records: list[dict[str, object]] = []
        bestRecord: dict[str, object] | None = None
        for candidate in candidates:
            similarity = candidate.appearanceSimilarity
            score = _observationScore(candidate)
            if similarity is None or similarity < tuning.reacquireSimilarity:
                verdict = "low_similarity"
            elif similarity < current + tuning.reacquireMargin:
                verdict = "low_margin"
            elif score < tuning.reacquireScore:
                verdict = "low_score"
            else:
                verdict = "not_best"
            record: dict[str, object] = {
                "view": int(candidate.viewId),
                "similarity": None if similarity is None else round(float(similarity), 4),
                "score": round(score, 4),
                "verdict": verdict,
                **_place(candidate.bfov),
            }
            records.append(record)
            if verdict == "not_best" and (
                best is None or similarity > (best.appearanceSimilarity or -1.0)
            ):
                best, bestRecord = candidate, record
        if bestRecord is not None:
            bestRecord["verdict"] = "accepted"
        return best, records

    def _reacquire(
        self, pending: _PendingFrame, candidate: ProjectedObservation
    ) -> TrackResult:
        """Restart the track from a scan candidate."""
        plan, frame = pending.plan, pending.frame
        score = _observationScore(candidate)
        self._backendRevision = plan.templateCommand.expectedRevision
        self._lastGoodBfov = candidate.bfov
        self._stateMachine.reset()
        self._scanCursor = 0
        self._restartMotion(candidate.bfov, frame)
        self._currentBfov = candidate.bfov
        self._currentBox = candidate.bbox
        self._trajectory.clear()
        self._trajectory.extend([candidate.bfov] * TRAJECTORY_LENGTH)
        self._lastFrameReacquired = True
        self._reacquiredFrames.append(int(frame.frameIndex))
        self._mode = TrackMode.TRACKING
        self._stableFrames = 0
        self._pendingTemplate = TemplateDecision(TemplateCommandKind.KEEP)
        self._stateRevision = plan.stateRevision
        self._lastFrameIndex = int(frame.frameIndex)
        self._lastFrame = frame
        self._pending = None
        self._lastStateObservation = None
        self._lastPipelineProfile = {
            "frameIndex": int(frame.frameIndex),
            "reacquired": True,
            "finalStateRevision": int(plan.stateRevision),
        }
        return TrackResult(
            sequenceId=frame.sequenceId,
            frameIndex=frame.frameIndex,
            bbox=candidate.bbox,
            bfov=candidate.bfov,
            confidence=score,
            status=_publicStatus(self._mode),
            valid=True,
            resultSource=ResultSource.OBSERVED_CONFIRMED,
        )

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
        # The published confidence stays the backend's own score: the state score is
        # an internal judgement and is reported through the track status.
        outputConfidence = (
            evaluation.backendScore
            if hasCandidate
            else max(0.0, pending.prediction.confidence * 0.5)
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
                    max(self._trackingConfig.candidateMinScore, evaluation.backendScore),
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


def _observationScore(observation: ProjectedObservation) -> float:
    value = (
        observation.singleScore
        if observation.singleScore is not None
        else observation.fusedScore
    )
    return float(min(1.0, max(0.0, value)))


def _motionCenter(motion: MotionState3D):
    x, y, z = motion.position
    return makeSphericalPoint(atan2(x, z), asin(max(-1.0, min(1.0, y))))


def _angularSize(bfov: BFoV) -> float:
    return sqrt(bfov.horizontalFovRad * bfov.verticalFovRad)


def _place(bfov: BFoV) -> dict[str, float]:
    """Where a box is and how large, in degrees, for the state trace."""
    center = bfov.center
    return {
        "yawDeg": round(degrees(atan2(center.x, center.z)), 3),
        "pitchDeg": round(degrees(asin(max(-1.0, min(1.0, center.y)))), 3),
        "sizeDeg": round(degrees(_angularSize(bfov)), 3),
    }


def _publicStatus(mode: TrackMode) -> TrackStatus:
    if mode is TrackMode.UNCERTAIN:
        return TrackStatus.UNCERTAIN
    if mode is TrackMode.LOST:
        return TrackStatus.LOST
    return TrackStatus.TRACKING


__all__ = ["TrackControllerImpl"]
