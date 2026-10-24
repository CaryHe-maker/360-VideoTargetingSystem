"""Single-view controller for spherical RGB tracking."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import acos, asin, atan2, degrees, pi, sqrt
from typing import TYPE_CHECKING, Any

from track360.controller.motion_estimator import SphericalMotionEstimator
from track360.controller.state_evaluator import StateEvaluator
from track360.controller.state_machine import TrackStateMachine
from track360.controller.state_model import (
    MotionPrediction,
    StateObservation,
    TrackMode,
    TransitionDecision,
)
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
# Search views a lost frame may take when the budget is not limited.
_UNLIMITED_SCAN_VIEWS = 4
# The loss actions that look for the target and may jump to what they find.
_SEARCHING_ACTIONS = ("jump", "move")


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

        self._initialized = False
        self._sequenceId: str | None = None
        self._lastFrameIndex = -1
        self._stateRevision = -1
        self._mode = TrackMode.INIT
        self._lastFrame: FramePacket | None = None
        self._initialBox: BBoxXYWH | None = None
        self._currentBox: BBoxXYWH | None = None
        self._currentBfov: BFoV | None = None
        # The target's BFoV after each of the last frames, oldest first.
        self._trajectory: deque[BFoV] = deque(maxlen=TRAJECTORY_LENGTH)
        self._pending: _PendingFrame | None = None
        self._initialPlan: InitializationPlan | None = None
        self._lastStateObservation: StateObservation | None = None
        self._lastPipelineProfile: dict[str, object] = {}
        # Loss handling: where the target was last trusted.
        self._lastGoodBfov: BFoV | None = None
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
        self._scanPrepaid = 0
        self._lastFrameTrace: dict[str, object] = {}
        # With a learned candidate judge, the score a candidate needs from it.
        self._judgeThreshold: float | None = None
        # Evaluation only (E052): a partial oracle that corrects some decisions.
        self._oracle: Any | None = None
        # Experimental (E053): the camera's rotation, to turn the search centre with it.
        self._camera: Any | None = None
        self._lastGoodFrame = 0
        self._lastScanCentre: SphericalPoint | None = None
        # Experimental (E053): targets at least this large search the whole sphere.
        self._sphereScanRad: float | None = None
        self._lastScanSphere = False
        # Experimental (E061): two-frame confirmation of candidates below the taking
        # score: (low, window frames, tolerance in sizes) and the candidates held.
        self._confirm: tuple[float, int, float] | None = None
        self._held: list[tuple[int, SphericalPoint, float]] = []
        # Experimental (E064): a watch window after a doubted track is trusted again:
        # (frames, freeze the appearance memory, keep searching), the frames left, and
        # the place trusted before the doubt began.
        self._postReleaseOptions: tuple[int, bool, bool] | None = None
        self._postRelease = 0
        self._doubtAnchor: tuple[BFoV, int] | None = None
        self._releaseAnchor: tuple[BFoV, int] | None = None
        self._trustedStreak = 0
        self._mainViewSize = (0.0, 0.0)
        # Experimental (E054): every this many lost frames, a detector looks at the
        # whole sphere and normal views are taken at the best ``_globalCount`` places.
        self._globalEvery = 0
        self._globalCount = 0
        # With ``_globalAfter`` > 0: from that many lost frames on, every search is one.
        self._globalAfter = 0
        self._globalPending = False

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
        self._lastGoodFrame = 0
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
        if prediction.horizontalSizeRad > 0.0 and prediction.verticalSizeRad > 0.0:
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
        self._mainViewSize = (targetBfov.horizontalFovRad, targetBfov.verticalFovRad)
        scanViews: tuple[ViewSpec, ...] = ()
        scanRefine = False
        self._lastScan = ""
        self._lastScanCentre = None
        self._lastScanSphere = False
        self._globalPending = False
        lost = self._mode is TrackMode.LOST
        lostFrames = self._lostFrames if lost else 0
        self._lostFrames = lostFrames + 1 if lost else 0
        scanCount = _UNLIMITED_SCAN_VIEWS
        if self._backendTuning.scanBudgetPerFrame > 0.0:
            # Every frame earns a share of a scan view; a lost track spends the whole
            # ones it has saved, so the average cost per frame stays bounded.
            self._scanTokens = min(
                self._backendTuning.scanBudgetBurst,
                self._scanTokens + self._backendTuning.scanBudgetPerFrame,
            )
            scanCount = min(scanCount, int(self._scanTokens))
        searching = self._actions in _SEARCHING_ACTIONS and lost and self._lastGoodBfov is not None
        periodic = self._globalEvery > 0 and lostFrames > 0 and lostFrames % self._globalEvery == 0
        late = self._globalAfter > 0 and lostFrames >= self._globalAfter
        globalFrame = searching and (periodic or late) and scanCount >= self._globalCount
        if searching and late and not globalFrame:
            # Saving up for the next detection: no enlarged search meanwhile.
            searching = False
        if globalFrame:
            # The views are chosen after the detector has looked (``refinementView``).
            self._globalPending = True
            self._lastScan = "global"
            self._scanPrepaid = self._globalCount
            self._scanTokens -= self._scanPrepaid
            self._scanFrameCount += 1
        elif searching or self._watching():
            anchor = None
            if not searching:
                anchor = self._releaseAnchor
                lostFrames = max(1, int(frame.frameIndex) - anchor[1])
            scanViews, scanRefine = self._zoomScan(
                mainView, lostFrames, scanCount, int(frame.frameIndex), anchor
            )
            if scanViews:
                self._scanPrepaid = len(scanViews) + (1 if scanRefine else 0)
                self._scanTokens -= self._scanPrepaid
                self._scanFrameCount += 1
        plan = SearchPlan(
            sequenceId=frame.sequenceId,
            frameIndex=frame.frameIndex,
            stateRevision=self._stateRevision + 1,
            view=mainView,
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
        return plan

    def _watching(self) -> bool:
        return bool(
            self._postRelease > 0
            and self._postReleaseOptions is not None
            and self._postReleaseOptions[2]
            and self._releaseAnchor is not None
            and self._mode is TrackMode.TRACKING
        )

    def _zoomScan(
        self,
        mainView: ViewSpec,
        lostFrames: int,
        budget: int,
        frameIndex: int = -1,
        anchor: tuple[BFoV, int] | None = None,
    ) -> tuple[tuple[ViewSpec, ...], bool]:
        """The search views of a lost frame, and whether what they find is refined.

        The first lost frame looks again at the view the tracker is in, without
        the tracker's memory.  Later frames take one enlarged view around the place
        the track was last trusted, larger once the track has been lost for a while;
        the box found there is not a candidate itself but the centre of a view of the
        normal size (see ``refinementView``).
        """
        tuning = self._backendTuning
        trusted = self._lastGoodBfov if anchor is None else anchor[0]
        trustedFrame = self._lastGoodFrame if anchor is None else anchor[1]
        assert trusted is not None
        if lostFrames == 0:
            if budget < 1:
                return (), False
            self._lastScan = "1x"
            return (replace(mainView, viewId=SCAN_VIEW_ID_BASE, trajectory=()),), False
        limited = tuning.scanBudgetPerFrame > 0.0
        if budget < 1 or (limited and int(self._scanTokens) < 2):
            return (), False
        scale = (
            tuning.zoomLastScale
            if lostFrames >= tuning.zoomLastAfterFrames
            else tuning.zoomFirstScale
        )
        self._lastScan = f"{scale:g}x"
        centre = trusted.center
        if self._camera is not None and frameIndex > trustedFrame + 1:
            # The camera may have turned since the place was trusted; the frame before
            # this one is the latest whose rotation is known when the search is planned.
            centre = _turned(centre, self._camera.rotation(trustedFrame, frameIndex - 1))
        view = self._planner.probeView(
            centre,
            trusted.horizontalFovRad,
            trusted.verticalFovRad,
            scale,
            SCAN_VIEW_ID_BASE,
        )
        if self._sphereScanRad is not None and _angularSize(trusted) >= self._sphereScanRad:
            view = self._planner.sphereView(
                centre, trusted.horizontalFovRad, trusted.verticalFovRad, SCAN_VIEW_ID_BASE
            )
            self._lastScanSphere = True
        if self._oracle is not None:
            moved = self._oracle.scanCentre(frameIndex, centre, 0.5 * view.bfov.horizontalFovRad)
            if moved is not centre:
                centre = moved
                view = self._planner.probeView(
                    centre,
                    trusted.horizontalFovRad,
                    trusted.verticalFovRad,
                    scale,
                    SCAN_VIEW_ID_BASE,
                )
        self._lastScanCentre = view.bfov.center
        return (view,), True

    def useCameraMotion(self, camera: Any | None) -> None:
        """Experimental (E053): centre enlarged searches where the trusted place has turned to.

        ``camera.rotation(a, b)`` takes a direction seen in frame ``a`` to frame ``b``.
        """
        self._camera = camera

    def useGlobalSearch(self, every: int, count: int, after: int = 0) -> None:
        """Experimental (E054): a whole-sphere detection every ``every`` lost frames, or,
        with ``after`` > 0, in place of every enlarged search once the track has been
        lost that long."""
        self._globalEvery, self._globalCount = int(every), int(count)
        self._globalAfter = int(after)

    @property
    def lastGoodFrame(self) -> int:
        """The index of the last frame whose box was trusted (or jumped to)."""
        return self._lastGoodFrame

    @property
    def globalSearchPending(self) -> bool:
        """The frame being planned looks for the target on the whole sphere."""
        return self._globalPending

    @property
    def trustedBfov(self) -> BFoV | None:
        """Where and how large the target was when last trusted."""
        return self._lastGoodBfov

    def useTwoFrameConfirmation(self, low: float, window: int, tolerance: float) -> None:
        """Experimental (E061): take a candidate scoring in [low, reacquireScore) when the
        next search with candidates, within ``window`` frames, finds one scoring at least
        ``low`` where it should be (turned with the camera, within ``tolerance`` sizes)."""
        self._confirm = (float(low), int(window), float(tolerance))

    def usePostReleaseWatch(self, frames: int, freeze: bool, search: bool) -> None:
        """Experimental (E064): for ``frames`` frames after a doubted track is trusted
        again, keep the appearance memory as it was (``freeze``) and go on searching
        around the place trusted before the doubt (``search``)."""
        self._postReleaseOptions = (int(frames), bool(freeze), bool(search))

    @property
    def memoryFrozen(self) -> bool:
        """The appearance memory must not learn from this frame (E064 watch window)."""
        return bool(
            self._postRelease > 0 and self._postReleaseOptions and self._postReleaseOptions[1]
        )

    @property
    def cameraAnchorFrame(self) -> int:
        """The frame the camera's rotation is needed from."""
        if self._postRelease > 0 and self._releaseAnchor is not None:
            return self._releaseAnchor[1]
        return self._lastGoodFrame

    @property
    def trustedStreak(self) -> int:
        """Frames in a row the track has been trusted (E064)."""
        return self._trustedStreak

    @property
    def lastRelativeSignal(self) -> float | None:
        return self._stateMachine.relativeSignal

    @property
    def isLost(self) -> bool:
        return self._mode is TrackMode.LOST

    def recentredMainView(self, plan: SearchPlan, center: SphericalPoint) -> ViewSpec:
        """The frame's tracking view, centred elsewhere (experimental, E061)."""
        view = self._planner.searchView(
            center, self._mainViewSize[0], self._mainViewSize[1], tuple(self._trajectory)
        )
        return replace(view, viewId=plan.view.viewId)

    def useSphereScan(self, minSizeRad: float | None) -> None:
        """Experimental (E053): enlarged searches of targets this large take the whole sphere."""
        self._sphereScanRad = None if minSizeRad is None else float(minSizeRad)

    def useOracle(self, oracle: Any | None) -> None:
        """Evaluation only (E052): let ground truth correct a share of the decisions."""
        self._oracle = oracle

    def useCandidateJudge(self, threshold: float | None) -> None:
        """Take search candidates by their judge score (experimental, E051); None: off."""
        self._judgeThreshold = None if threshold is None else float(threshold)

    @property
    def currentScan(self) -> str:
        """The scan of the frame being planned: "1x", "2x", "4x" or ""."""
        return self._lastScan

    def recentredView(self, center: SphericalPoint, sizeRad: float) -> ViewSpec:
        """A normal-size search view centred on a candidate of angular size ``sizeRad``."""
        side = min(max(sizeRad, 1e-4), pi - 1e-3)
        return self._planner._searchView(center, side, side)

    def canSpendScan(self, forwards: int) -> bool:
        """Whether the scan budget holds this many more passes (always without a budget)."""
        if self._backendTuning.scanBudgetPerFrame <= 0.0:
            return True
        return self._scanTokens >= forwards

    def chargeScan(self, forwards: int) -> None:
        """Spend scan budget on passes outside the plan (experimental, E061)."""
        if self._backendTuning.scanBudgetPerFrame > 0.0:
            self._scanTokens -= forwards

    def settleScan(self, forwards: int) -> None:
        """Charge the scan of the frame what it really cost."""
        if self._backendTuning.scanBudgetPerFrame > 0.0 and self._scanPrepaid:
            self._scanTokens -= forwards - self._scanPrepaid
        self._scanPrepaid = 0

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

        ``candidates`` are the boxes found in the plan's search views.  When one of
        them scores high enough without the tracker's memory, the track jumps to it.
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
        frameIndex = int(pending.frame.frameIndex)
        trackerBox = observation.bbox if observation is not None else None
        forced = (
            self._oracle.forceDoubt(frameIndex, trackerBox) if self._oracle is not None else False
        )
        if self._actions in _SEARCHING_ACTIONS:
            chosen, trace["candidates"] = self._reacquisitionCandidate(
                candidates, frameIndex, trackerBox
            )
            if chosen is not None:
                result = self._reacquire(pending, chosen)
                trace.update(
                    modeAfter=self._mode.name, reason="JUMP", action="jump", **_place(result.bfov)
                )
                self._lastFrameTrace = trace
                return result
        decision = self._stateMachine.transition(
            pending.mode,
            evaluation.stateScore,
            measurementAccepted=evaluation.measurementAccepted,
            backendScore=evaluation.backendScore if evaluation.hasCandidate else None,
            appearanceScore=evaluation.appearanceScore,
            forceUntrusted=forced,
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
            self._watchBookkeeping(pending.mode, decision.nextMode)
            if decision.nextMode is TrackMode.TRACKING:
                self._lastGoodBfov = self._currentBfov
                self._lastGoodFrame = int(plan.frameIndex)
                self._held = []
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
            **_scanPlace(self._lastScanCentre),
            "scanSphere": int(self._lastScanSphere),
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
        candidates: Sequence[ProjectedObservation],
        frameIndex: int = -1,
        trackerBox: BBoxXYWH | None = None,
    ) -> tuple[ProjectedObservation | None, list[dict[str, object]]]:
        """The search candidate to jump to, if any, and what became of each one.

        A candidate is taken when its score without the tracker's memory reaches
        ``reacquireScore``: the only signal found to tell a re-found target from
        another object (evaluation log E026, E034, E036).  A candidate on the box the
        tracker already has is taken as well: the jump then only restarts the
        tracker's memory, which is what helped in E030 and E031.
        """
        best: ProjectedObservation | None = None
        records: list[dict[str, object]] = []
        bestRecord: dict[str, object] | None = None
        judged = self._judgeThreshold is not None
        for candidate in candidates:
            similarity = candidate.appearanceSimilarity
            score = _observationScore(candidate)
            if judged:
                # The learned judge decides instead of the score (evaluation log E051).
                verdict = (
                    "low_score"
                    if candidate.judgeScore is None or candidate.judgeScore < self._judgeThreshold
                    else "not_best"
                )
            else:
                verdict = "low_score" if score < self._backendTuning.reacquireScore else "not_best"
            if verdict == "not_best" and self._actions == "move" and self._onCurrentBox(candidate):
                verdict = "same_place"
            if verdict == "not_best" and self._watching() and self._onCurrentBox(candidate):
                verdict = "same_place"  # the tracker already has it: nothing to jump to
            record: dict[str, object] = {
                "view": int(candidate.viewId),
                "similarity": None if similarity is None else round(float(similarity), 4),
                "score": round(score, 4),
                "verdict": verdict,
                **_place(candidate.bfov),
            }
            if judged:
                record["judge"] = (
                    None if candidate.judgeScore is None else round(float(candidate.judgeScore), 6)
                )
            records.append(record)
            if verdict == "not_best" and (best is None or _preferred(candidate, best, judged)):
                best, bestRecord = candidate, record
        if self._confirm is not None and not judged and candidates:
            best, bestRecord = self._confirmHeld(candidates, records, frameIndex, best, bestRecord)
        oracle = self._oracle
        if oracle is not None and candidates:
            if best is None:
                taken = oracle.takeMissed(frameIndex, list(candidates))
                if taken is not None:
                    best = taken
                    bestRecord = records[next(i for i, c in enumerate(candidates) if c is taken)]
                    bestRecord["oracle"] = "D"
            if best is not None and oracle.vetoJump(frameIndex, best, trackerBox):
                assert bestRecord is not None
                bestRecord["verdict"], bestRecord["oracle"] = "vetoed", "E"
                return None, records
        if bestRecord is not None:
            bestRecord["verdict"] = "accepted"
        return best, records

    def _watchBookkeeping(self, before: TrackMode, after: TrackMode) -> None:
        """E064: start, count down and end the watch window; keep the pre-doubt place."""
        trustedNow = after is TrackMode.TRACKING
        self._trustedStreak = self._trustedStreak + 1 if trustedNow else 0
        if self._postReleaseOptions is None:
            return
        if before is TrackMode.TRACKING and not trustedNow:
            if self._postRelease > 0 and self._releaseAnchor is not None:
                # Doubted again inside the window: search from the place trusted before
                # the first doubt, not from where the track was released.
                self._lastGoodBfov, self._lastGoodFrame = self._releaseAnchor
                self._doubtAnchor = self._releaseAnchor
            else:
                assert self._lastGoodBfov is not None
                self._doubtAnchor = (self._lastGoodBfov, self._lastGoodFrame)
            self._postRelease = 0
        elif before is not TrackMode.TRACKING and trustedNow and self._doubtAnchor is not None:
            self._postRelease = self._postReleaseOptions[0]
            self._releaseAnchor = self._doubtAnchor
        elif trustedNow and self._postRelease > 0:
            self._postRelease -= 1

    def _confirmHeld(self, candidates, records, frameIndex, best, bestRecord):
        """Two-frame confirmation (E061): check the held candidates, then hold anew."""
        assert self._confirm is not None
        low, window, tolerance = self._confirm
        held = [h for h in self._held if 0 < frameIndex - h[0] <= window]
        if best is None and held:
            confirmed = []
            for index, candidate in enumerate(candidates):
                if _observationScore(candidate) < low:
                    continue
                size = _angularSize(candidate.bfov)
                for frame, place, heldSize in held:
                    turned = place
                    if self._camera is not None and frameIndex - 1 > frame:
                        turned = _turned(place, self._camera.rotation(frame, frameIndex - 1))
                    c = candidate.bfov.center
                    cosine = c.x * turned.x + c.y * turned.y + c.z * turned.z
                    if acos(max(-1.0, min(1.0, cosine))) < tolerance * max(size, heldSize):
                        confirmed.append(index)
                        break
            if confirmed:
                pick = max(confirmed, key=lambda i: _observationScore(candidates[i]))
                best, bestRecord = candidates[pick], records[pick]
                bestRecord["confirmed"] = 1
        # Only the next search with candidates counts: hold this search's candidates.
        self._held = (
            []
            if best is not None
            else [
                (frameIndex, c.bfov.center, _angularSize(c.bfov))
                for c in candidates
                if low <= _observationScore(c) < self._backendTuning.reacquireScore
            ]
        )
        return best, bestRecord

    def _onCurrentBox(self, candidate: ProjectedObservation) -> bool:
        """The candidate is the box the tracker already has: same place, similar size."""
        current = self._currentBfov
        if current is None:
            return False
        cosine = (
            current.center.x * candidate.bfov.center.x
            + current.center.y * candidate.bfov.center.y
            + current.center.z * candidate.bfov.center.z
        )
        larger = max(_angularSize(current), _angularSize(candidate.bfov))
        smaller = min(_angularSize(current), _angularSize(candidate.bfov))
        return acos(max(-1.0, min(1.0, cosine))) < 0.5 * larger and larger < 1.5 * smaller

    def _reacquire(self, pending: _PendingFrame, candidate: ProjectedObservation) -> TrackResult:
        """Restart the track from a scan candidate."""
        plan, frame = pending.plan, pending.frame
        score = _observationScore(candidate)
        self._lastGoodBfov = candidate.bfov
        self._lastGoodFrame = int(frame.frameIndex)
        self._held = []
        self._postRelease = 0
        self._doubtAnchor = None
        self._trustedStreak = 0
        self._stateMachine.reset()
        self._restartMotion(candidate.bfov, frame)
        self._currentBfov = candidate.bfov
        self._currentBox = candidate.bbox
        self._trajectory.clear()
        self._trajectory.extend([candidate.bfov] * TRAJECTORY_LENGTH)
        self._lastFrameReacquired = True
        self._reacquiredFrames.append(int(frame.frameIndex))
        self._mode = TrackMode.TRACKING
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
        self._lastFrameIndex = int(frame.frameIndex)
        self._lastFrame = frame
        self._pending = None
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
            if currentArea >= 0.10 * frameArea:
                outputBox = self._currentBox
                outputBfov = self._currentBfov
                holdingWeak = True
        self._mode = decision.nextMode
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

    def _recordMeasurement(self, pending: _PendingFrame, bfov: BFoV, confidence: float) -> None:
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


def _turned(point: SphericalPoint, rotation) -> SphericalPoint:
    x, y, z = (float(value) for value in rotation @ (point.x, point.y, point.z))
    norm = sqrt(x * x + y * y + z * z)
    return makeSphericalPoint(atan2(x, z), asin(max(-1.0, min(1.0, y / norm))))


def _scanPlace(centre: SphericalPoint | None) -> dict[str, object]:
    """Where the enlarged search view of the frame was centred, for the state trace."""
    if centre is None:
        return {"scanYawDeg": "", "scanPitchDeg": ""}
    return {
        "scanYawDeg": round(degrees(atan2(centre.x, centre.z)), 3),
        "scanPitchDeg": round(degrees(asin(max(-1.0, min(1.0, centre.y)))), 3),
    }


def _preferred(candidate: ProjectedObservation, best: ProjectedObservation, judged: bool) -> bool:
    """Whether an accepted candidate beats the best one so far."""
    if judged:
        return float(candidate.judgeScore or 0.0) > float(best.judgeScore or 0.0)
    similarity = candidate.appearanceSimilarity
    bestSimilarity = best.appearanceSimilarity
    return (similarity if similarity is not None else -1.0) > (
        bestSimilarity if bestSimilarity is not None else -1.0
    )


def _observationScore(observation: ProjectedObservation) -> float:
    value = (
        observation.singleScore if observation.singleScore is not None else observation.fusedScore
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
