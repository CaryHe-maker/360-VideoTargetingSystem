"""Runtime driver orchestration."""

from __future__ import annotations

import math
import queue
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

import numpy as np

from track360.backends import (
    ARTrackBackend,
    ARTrackSession,
    TrackerBackendImpl,
    createArtrackSession,
)
from track360.controller import (
    TrackControllerImpl,
    backendScoreProbability,
    scoreViewCenterMotion,
    withScoreProbability,
)
from track360.core.config import AppConfig, ModelConfig
from track360.core.errors import DecodeError, GeometryError, ProtocolError
from track360.core.protocols import FrameSource as FrameSourceProtocol
from track360.core.protocols import ResultSink as ResultSinkProtocol
from track360.core.protocols import SphericalGeometry, TrackerBackend
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FramePacket,
    LocalObservation,
    LocalView,
    MotionState3D,
    ProjectedObservation,
    SearchPlan,
    ViewSpec,
)
from track360.geometry import SphericalGeometryImpl
from track360.io.result_sink import FileResultSink
from track360.runtime.gpu_lock import isDeviceFailure
from track360.runtime.reproducibility import seedEverything

if TYPE_CHECKING:
    from track360.backends.artrack_model import ARTrackSession
    from track360.evaluation.profiler import RuntimeProfiler
    from track360.visualization.recorder import VisualizationRecorder
    from track360.visualization.result import ResultVisualizationRecorder
    from track360.visualization.time_counter import TimeCounter


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    geometry: SphericalGeometry
    controller: TrackControllerImpl
    backend: TrackerBackend
    sink: ResultSinkProtocol
    verifier: Any | None = None
    recorder: VisualizationRecorder | None = None
    # Experimental (E051): a learned judge of search candidates and the session whose
    # diagnostics it reads.
    judge: Any | None = None
    judgeSession: Any | None = None
    # The camera's rotation (``backendTuning.cameraMotion``, E053).
    camera: Any | None = None


class _PrefetchReader:
    """Decode frames ahead of inference while preserving source order."""

    _END = object()

    def __init__(self, source: FrameSourceProtocol) -> None:
        self._source = source
        self._queue: queue.Queue[object] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._decodeNs: dict[int, int] = {}
        self._readyNs: dict[int, int] = {}
        self._lastProfile: dict[str, int | bool] = {}

    @property
    def lastProfile(self) -> dict[str, int | bool]:
        return dict(self._lastProfile)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("prefetch reader already started")
        self._thread = threading.Thread(target=self._worker, name="track360-decode", daemon=True)
        self._thread.start()

    def read(self) -> FramePacket | None:
        waitStartedNs = perf_counter_ns()
        item = self._queue.get()
        waitNs = perf_counter_ns() - waitStartedNs
        if isinstance(item, BaseException):
            raise item
        if item is self._END:
            return None
        if not isinstance(item, FramePacket):
            raise RuntimeError("prefetch reader returned an invalid item")
        frameIndex = int(item.frameIndex)
        self._lastProfile = {
            "pipelinePrefetchEnabled": True,
            "pipelineDecodeNs": int(self._decodeNs.pop(frameIndex, 0)),
            "pipelineQueueWaitNs": int(waitNs),
            "pipelineReadyLeadNs": int(
                max(0, perf_counter_ns() - self._readyNs.pop(frameIndex, 0))
            ),
            "pipelineFrameIndex": frameIndex,
        }
        return item

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _worker(self) -> None:
        try:
            while not self._stop.is_set():
                decodeStartedNs = perf_counter_ns()
                frame = self._source.read()
                decodeNs = perf_counter_ns() - decodeStartedNs
                if frame is not None:
                    frameIndex = int(frame.frameIndex)
                    self._decodeNs[frameIndex] = int(decodeNs)
                    self._readyNs[frameIndex] = perf_counter_ns()
                if not self._enqueue(self._END if frame is None else frame):
                    return
                if frame is None:
                    return
        except BaseException as error:
            self._error = error
            self._enqueue(error)

    def _enqueue(self, item: object) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False


def buildRuntime(
    config: AppConfig,
    *,
    artrackSessionFactory: Callable[[ModelConfig], ARTrackSession] | None = None,
    geometryFactory: Callable[[int], SphericalGeometry] | None = None,
    profile: bool = False,
    candidateJudge: Any | None = None,
) -> RuntimeBundle:
    tuning = config.backendTuning
    seedEverything(config.reproducibility)
    if geometryFactory is not None:
        geometry = geometryFactory(config.geometry.boundarySamplesPerEdge)
    else:
        geometry = SphericalGeometryImpl(
            boundarySamplesPerEdge=config.geometry.boundarySamplesPerEdge,
        )
    rgbSession = (
        artrackSessionFactory(config.model)
        if artrackSessionFactory is not None
        else createArtrackSession(config)
    )
    backend = TrackerBackendImpl(ARTrackBackend(rgbSession))
    controller = TrackControllerImpl(geometry, config)
    sink = FileResultSink()
    recorder = None
    if config.visualization.enabled:
        from track360.visualization.recorder import VisualizationRecorder

        recorder = VisualizationRecorder(config.visualization)
    verifier = None
    if tuning.lossHandling:
        from track360.backends.appearance import AppearanceVerifier

        verifier = AppearanceVerifier("dinov2", Path(config.model.weights).parent / "hub")
    camera = None
    if tuning.cameraMotion:
        from track360.runtime.camera_motion import CameraMotion

        camera = CameraMotion()
        controller.useCameraMotion(camera)
    if candidateJudge is not None:
        if verifier is None:
            raise ProtocolError("a candidate judge needs loss handling")
        controller.useCandidateJudge(candidateJudge.threshold)
    return RuntimeBundle(
        geometry=geometry,
        controller=controller,
        backend=backend,
        sink=sink,
        recorder=recorder,
        verifier=verifier,
        judge=candidateJudge,
        judgeSession=rgbSession if candidateJudge is not None else None,
        camera=camera,
    )


def runTracking(
    *,
    source: FrameSourceProtocol,
    initialBox: BBoxXYWH | None = None,
    initialBfov: BFoV | None = None,
    geometry: SphericalGeometry,
    controller: TrackControllerImpl,
    backend: TrackerBackend,
    sink: ResultSinkProtocol,
    recorder: VisualizationRecorder | None = None,
    resultRecorder: ResultVisualizationRecorder | None = None,
    processingTimer: TimeCounter | None = None,
    profiler: RuntimeProfiler | None = None,
    probe: Any | None = None,
    verifier: Any | None = None,
    trace: list[dict[str, object]] | None = None,
    judge: Any | None = None,
    judgeSession: Any | None = None,
    oracle: Any | None = None,
    camera: Any | None = None,
    globalSearcher: Any | None = None,
) -> int:
    """Run the sequential tracking pipeline and publish one result per frame.

    The frame-0 target is given as exactly one of ``initialBox`` (ERP pixels) or
    ``initialBfov``.
    """
    try:
        initializationStartedNs = _profileNow(profiler)
        _startProcessing(processingTimer)
        try:
            decodeStartedNs = _profileNow(profiler)
            frame0 = _requireFrame(source.read())
            if camera is not None:
                camera.submit(int(frame0.frameIndex), frame0.rgb)
            if globalSearcher is not None and initialBfov is not None:
                globalSearcher.begin(frame0.rgb, initialBfov)
            _startProfileFrame(profiler, int(frame0.frameIndex), decodeStartedNs)
            with _profile(profiler, "controller"):
                initPlan = controller.buildInitialization(
                    frame0, initialBox, initialBfov=initialBfov
                )
            with _profile(profiler, "crop"):
                templateView = geometry.cropViews(frame0, [initPlan.templateView])[0]
            _recordGeometryProfile(profiler, geometry)
            backend.initialize(templateView, initPlan.templateBox)
            if probe is not None:
                probe.begin(templateView, initPlan.templateBox)
            if verifier is not None:
                verifier.setTemplate(templateView, initPlan.templateBox)
            initialResult = controller.commitInitialization(initPlan)
            if camera is not None:
                camera.mark(int(frame0.frameIndex), True)
        finally:
            _stopProcessing(processingTimer)
        _finishProfileFrame(profiler, initializationStartedNs, batchSizes=[1], forwardCount=0)
        sink.write(initialResult)
        if resultRecorder is not None:
            resultRecorder.record(frame0, initialResult, stateScore=None)
        resultCount = 1
        if recorder is not None:
            recorder.recordLocalRgb(frame0, [templateView])

        pipelineReader = _PrefetchReader(source)
        pipelineReader.start()
        try:
            while True:
                iterationStartedNs = (
                    perf_counter_ns() if profiler is not None and profiler.enabled else None
                )
                _startProcessing(processingTimer)
                frame = None
                result = None
                try:
                    decodeStartedNs = _profileNow(profiler)
                    frame = pipelineReader.read()
                    if frame is not None:
                        _startProfileFrame(profiler, int(frame.frameIndex), decodeStartedNs)
                        if camera is not None:
                            camera.submit(int(frame.frameIndex), frame.rgb)
                        _recordPipelineProfile(profiler, pipelineReader, controller)
                        try:
                            with _profile(profiler, "controller"):
                                plan = controller.beginFrame(frame)
                        except Exception as error:
                            result = _fallbackFrameResult(
                                controller,
                                frame,
                                error,
                            )
                            _finishProfileFrame(
                                profiler,
                                iterationStartedNs,
                                batchSizes=[],
                                forwardCount=0,
                                frameFallback=True,
                            )
                            sink.write(result)
                            if resultRecorder is not None:
                                resultRecorder.record(frame, result, stateScore=None)
                            releaseFrame = getattr(geometry, "releaseFrame", None)
                            if callable(releaseFrame):
                                releaseFrame()
                            resultCount += 1
                            continue
                        forwardCount = 0
                        memory = ""
                        probed = None
                        visualization: (
                            tuple[LocalView, LocalObservation, ProjectedObservation | None] | None
                        ) = None
                        try:
                            with _profile(profiler, "crop"):
                                view = geometry.cropViews(frame, [plan.view])[0]
                            _recordGeometryProfile(profiler, geometry)
                            forwardCount = 1
                            savedState = (
                                backend.saveState()  # type: ignore[attr-defined]
                                if verifier is not None
                                else None
                            )
                            with _profile(profiler, "backend"):
                                rawObservation = backend.infer((view,))[0]
                            probed = (view, rawObservation)
                            _recordBackendProfile(profiler, backend)
                            with _profile(profiler, "calibration"):
                                observation = withScoreProbability((rawObservation,))[0]
                            with _profile(profiler, "projection"):
                                projected = _projectValidObservation(
                                    frame=frame,
                                    view=view,
                                    observation=observation,
                                    predictedMotion=plan.predictedMotion,
                                    geometry=geometry,
                                )
                            # PostTrainV2.4 keeps recorder inputs host-only at this boundary.
                            # Recorders only consume the compatibility RGB and view
                            # metadata.  Never retain CUDA tensors after this frame.
                            if recorder is not None:
                                visualization = (
                                    LocalView(spec=view.spec, rgb=view.rgb),
                                    observation,
                                    projected,
                                )
                            candidates: tuple[ProjectedObservation, ...] = ()
                            if verifier is not None:
                                if projected is not None:
                                    projected = replace(
                                        projected,
                                        appearanceSimilarity=verifier.similarity(
                                            view, observation.bbox
                                        ),
                                    )
                                if globalSearcher is not None and controller.globalSearchPending:
                                    candidates, scanForwards = _globalCandidates(
                                        frame=frame,
                                        controller=controller,
                                        geometry=geometry,
                                        backend=backend,
                                        verifier=verifier,
                                        searcher=globalSearcher,
                                    )
                                    forwardCount += scanForwards
                                    controller.settleScan(scanForwards)
                                elif plan.scanViews:
                                    candidates, scanForwards = _scanCandidates(
                                        frame=frame,
                                        plan=plan,
                                        geometry=geometry,
                                        backend=backend,
                                        verifier=verifier,
                                        refinementView=(
                                            controller.refinementView
                                            if plan.scanRefine
                                            else None
                                        ),
                                    )
                                    forwardCount += scanForwards
                                    controller.settleScan(scanForwards)
                                    if oracle is not None and plan.scanRefine:
                                        # Evaluation only (E052): a box near the target.
                                        scanView = plan.scanViews[0]
                                        extra = oracle.extraCandidate(
                                            int(frame.frameIndex),
                                            scanView.bfov.center,
                                            0.5 * scanView.bfov.horizontalFovRad,
                                            candidates,
                                            geometry,
                                        )
                                        if extra is not None:
                                            candidates = (*candidates, extra)
                                    if judge is not None and candidates:
                                        # Not charged to the scan budget: the search
                                        # stays as it was, only the decision changes.
                                        candidates, judgeForwards = _judgeCandidates(
                                            frame=frame,
                                            candidates=candidates,
                                            controller=controller,
                                            geometry=geometry,
                                            backend=backend,
                                            judge=judge,
                                            session=judgeSession,
                                        )
                                        forwardCount += judgeForwards
                            with _profile(profiler, "controller"):
                                result = controller.consume(plan, projected, candidates)
                            if camera is not None:
                                index = int(frame.frameIndex)
                                camera.mark(index, controller.lastGoodFrame == index)
                            if verifier is not None:
                                # A doubted frame must not shape the tracker's appearance
                                # memory; a jump to a candidate starts it afresh.
                                if controller.lastFrameReacquired:
                                    backend.resetState()  # type: ignore[attr-defined]
                                    memory = "reset"
                                elif controller.lastFrameSuspect and savedState is not None:
                                    backend.restoreState(savedState)  # type: ignore[attr-defined]
                                    memory = "frozen"
                            if trace is not None:
                                trace.append(
                                    {
                                        **controller.lastFrameTrace,
                                        "forwards": forwardCount,
                                        "memory": memory,
                                    }
                                )
                        except Exception as error:
                            # A failed frame (including an OOM converted by the backend)
                            # must not keep its CUDA view alive until the next frame.
                            visualization = None
                            result = _fallbackFrameResult(
                                controller,
                                frame,
                                error,
                            )
                        if probe is not None and probed is not None:
                            # Observation only: a failing probe must not touch the run.
                            try:
                                probe.observe(int(frame.frameIndex), *probed)
                            except Exception as error:
                                print(
                                    f"[runtime] probe failed: frame={int(frame.frameIndex)}, "
                                    f"reason={error}",
                                    file=sys.stderr,
                                )
                        _recordPipelineProfile(profiler, pipelineReader, controller)
                finally:
                    _stopProcessing(processingTimer)
                if frame is None:
                    break
                if result is None:
                    raise RuntimeError("tracking frame produced no result")
                _finishProfileFrame(
                    profiler,
                    iterationStartedNs,
                    batchSizes=[1] * forwardCount,
                    forwardCount=forwardCount,
                )
                if recorder is not None and visualization is not None:
                    hostView, observation, projected = visualization
                    recorder.recordLocalRgb(frame, (hostView,))
                    recorder.recordBackendBoxes(frame, (hostView,), (observation,))
                    recorder.recordGeometryBoxes(
                        frame, () if projected is None else (projected,)
                    )
                # Do not let this frame's CUDA-backed view survive into the next
                # iteration through Python locals.
                view = observation = projected = visualization = None
                sink.write(result)
                if resultRecorder is not None:
                    stateObservation = controller.lastStateObservation
                    stateScore = (
                        stateObservation.stateScore
                        if stateObservation is not None
                        and stateObservation.frameIndex == frame.frameIndex
                        else None
                    )
                    resultRecorder.record(
                        frame,
                        result,
                        stateScore=stateScore,
                    )
                releaseFrame = getattr(geometry, "releaseFrame", None)
                if callable(releaseFrame):
                    releaseFrame()
                resultCount += 1
                del result
        finally:
            pipelineReader.close()
            releaseFrame = getattr(geometry, "releaseFrame", None)
            if callable(releaseFrame):
                releaseFrame()
        return resultCount
    except Exception:
        if hasattr(sink, "close"):
            try:
                sink.close()  # type: ignore[call-arg]
            except Exception:
                pass
        raise


def finalizeSink(sink: ResultSinkProtocol, expectedFrameCount: int) -> None:
    sink.finalize(expectedFrameCount)


def openSink(sink: ResultSinkProtocol, destination: str) -> None:
    sink.open(destination)


def closeBackend(backend: TrackerBackend) -> None:
    backend.close()


def _fallbackFrameResult(
    controller: TrackControllerImpl,
    frame: FramePacket,
    error: BaseException,
):
    """Convert any expected runtime failure into one invalid frame result.

    A failure of the GPU itself is not converted: every later frame would fail the
    same way and the sequence would end looking complete (evaluation log, E047).
    """
    if isDeviceFailure(error):
        raise error
    print(
        "[runtime] frame failed; emitted zero-confidence fallback: "
        f"sequence={frame.sequenceId}, frame={int(frame.frameIndex)}, reason={error}",
        file=sys.stderr,
    )
    return controller.commitFallback(
        frame,
        reason=type(error).__name__,
    )


def closeRuntime(runtime: RuntimeBundle) -> None:
    """Release Geometry CUDA state before the backend empties the CUDA cache."""
    if runtime.camera is not None:
        runtime.camera.close()
    try:
        closeGeometry = getattr(runtime.geometry, "close", None)
        if callable(closeGeometry):
            closeGeometry()
    finally:
        closeBackend(runtime.backend)


def _projectObservation(
    *,
    frame: FramePacket,
    view: LocalView,
    observation: LocalObservation,
    predictedMotion: MotionState3D | None,
    geometry: SphericalGeometry,
) -> ProjectedObservation:
    projection = geometry.projectLocalBoxBoundary(
        observation.bbox,
        view.spec,
        frame.rgb.shape[1],
        frame.rgb.shape[0],
    )
    motion = scoreViewCenterMotion(view.spec.bfov.center, predictedMotion)
    appearanceProbability = (
        observation.appearanceProbability
        if observation.appearanceProbability is not None
        else backendScoreProbability(observation.fusedScore)
    )
    # ARTrack already scores its box against the template and the search crop.
    singleScore = float(np.clip(appearanceProbability, 0.0, 1.0))
    scaleScore = _scaleScore(observation.bbox, view, predictedMotion)
    normalizedRadius, edgeMargin = _projectionQuality(observation.bbox, view)
    return ProjectedObservation(
        viewId=view.spec.viewId,
        bfov=projection.bfov,
        bbox=projection.bbox,
        modelScore=observation.modelScore,
        appearanceScore=observation.appearanceScore,
        motionScore=motion.effectiveProbability,
        scaleScore=scaleScore,
        fusedScore=singleScore,
        localBox=observation.bbox,
        backendFusedScore=observation.fusedScore,
        appearanceProbability=appearanceProbability,
        rawMotionScore=motion.rawScore,
        motionProbability=motion.probability,
        motionReliability=motion.reliability,
        singleScore=singleScore,
        erpBoundary=projection.erpBoundary,
        envelopeInflation=projection.envelopeInflation,
        normalizedRadius=normalizedRadius,
        edgeMargin=edgeMargin,
    )


def _scanCandidates(
    *,
    frame: FramePacket,
    plan: SearchPlan,
    geometry: SphericalGeometry,
    backend: TrackerBackend,
    verifier: Any,
    refinementView: Callable[..., ViewSpec] | None = None,
) -> tuple[tuple[ProjectedObservation, ...], int]:
    """Boxes found in the plan's search views with their similarity to the template,
    and the forward passes spent.

    With ``refinementView`` a search view only points at a place: a view of the normal
    size is taken around each box found, and the boxes of those views are returned.
    """
    views = {view.spec.viewId: view for view in geometry.cropViews(frame, plan.scanViews)}
    found = backend.inferDetached(tuple(views.values()))  # type: ignore[attr-defined]
    forwards = len(views)
    if refinementView is not None:
        specs = []
        for local in found:
            try:
                pointed = geometry.projectLocalBoxBoundary(
                    local.bbox,
                    views[local.viewId].spec,
                    frame.rgb.shape[1],
                    frame.rgb.shape[0],
                )
            except GeometryError:
                continue
            specs.append(refinementView(pointed.bfov.center, len(specs)))
        views = (
            {view.spec.viewId: view for view in geometry.cropViews(frame, specs)}
            if specs
            else {}
        )
        found = backend.inferDetached(tuple(views.values())) if views else ()  # type: ignore[attr-defined]
        forwards += len(views)
    candidates = []
    for local in withScoreProbability(found):
        view = views[local.viewId]
        projected = _projectValidObservation(
            frame=frame,
            view=view,
            observation=local,
            predictedMotion=None,
            geometry=geometry,
        )
        if projected is not None:
            candidates.append(
                replace(
                    projected, appearanceSimilarity=verifier.similarity(view, local.bbox)
                )
            )
    return tuple(candidates), forwards


def _globalCandidates(
    *,
    frame: FramePacket,
    controller: TrackControllerImpl,
    geometry: SphericalGeometry,
    backend: TrackerBackend,
    verifier: Any,
    searcher: Any,
) -> tuple[tuple[ProjectedObservation, ...], int]:
    """Normal views at the best places a whole-sphere detector found (E054)."""
    trusted = controller.trustedBfov
    if trusted is None:
        return (), 0
    size = math.sqrt(trusted.horizontalFovRad * trusted.verticalFovRad)
    places = searcher.search(frame.rgb, controller._globalCount, max(size, math.radians(1.0)))
    specs = [controller.refinementView(place, index) for index, place in enumerate(places)]
    if not specs:
        return (), 0
    views = {view.spec.viewId: view for view in geometry.cropViews(frame, specs)}
    found = backend.inferDetached(tuple(views.values()))  # type: ignore[attr-defined]
    candidates = []
    for local in withScoreProbability(found):
        view = views[local.viewId]
        projected = _projectValidObservation(
            frame=frame,
            view=view,
            observation=local,
            predictedMotion=None,
            geometry=geometry,
        )
        if projected is not None:
            candidates.append(
                replace(projected, appearanceSimilarity=verifier.similarity(view, local.bbox))
            )
    return tuple(candidates), len(views)


def _judgeCandidates(
    *,
    frame: FramePacket,
    candidates: tuple[ProjectedObservation, ...],
    controller: TrackControllerImpl,
    geometry: SphericalGeometry,
    backend: TrackerBackend,
    judge: Any,
    session: Any,
) -> tuple[tuple[ProjectedObservation, ...], int]:
    """Score each candidate with the learned judge, and the forward passes spent.

    Each candidate gets one stateless pass on a normal-size view centred on it, as in
    the judge's training data (``tools/backend_signals.py trace``).  A candidate below
    the judge's search-score prefilter, or whose pass finds nothing, is not judged and
    cannot be taken.
    """
    from track360.runtime.candidate_judge import judgeFeatures

    judge.select(str(frame.sequenceId))
    scan = controller.currentScan
    judged = []
    forwards = 0
    for candidate in candidates:
        raw = candidate.singleScore if candidate.singleScore is not None else candidate.fusedScore
        search = float(min(1.0, max(0.0, raw)))
        if search < judge.prefilter:
            judged.append(candidate)
            continue
        size = math.sqrt(candidate.bfov.horizontalFovRad * candidate.bfov.verticalFovRad)
        spec = controller.recentredView(candidate.bfov.center, size)
        view = geometry.cropViews(frame, [spec])[0]
        session.enableDiagnostics(True)
        try:
            again = backend.inferDetached((view,))  # type: ignore[attr-defined]
            diagnostics = session.lastDiagnostics
        finally:
            session.enableDiagnostics(False)
        forwards += 1
        if not again or not diagnostics:
            judged.append(candidate)
            continue
        features = judgeFeatures(
            searchScore=search,
            recentredScore=again[0].fusedScore,
            similarity=candidate.appearanceSimilarity,
            scan=scan,
            diagnostics=diagnostics,
        )
        judged.append(replace(candidate, judgeScore=float(judge.score(features)[0])))
    return tuple(judged), forwards


def _projectValidObservation(
    *,
    frame: FramePacket,
    view: LocalView,
    observation: LocalObservation,
    predictedMotion: MotionState3D | None,
    geometry: SphericalGeometry,
) -> ProjectedObservation | None:
    """Project the local box onto the sphere; ``None`` when it has no valid projection."""
    try:
        return _projectObservation(
            frame=frame,
            view=view,
            observation=observation,
            predictedMotion=predictedMotion,
            geometry=geometry,
        )
    except GeometryError as error:
        print(
            "[runtime] skipped invalid spherical projection: "
            f"sequence={frame.sequenceId}, frame={int(frame.frameIndex)}, "
            f"view={view.spec.viewId}, reason={error}",
            file=sys.stderr,
        )
        return None


def _scaleScore(
    box: BBoxXYWH,
    view: LocalView,
    predictedMotion: MotionState3D | None = None,
) -> float:
    """Score scale against the predicted angular target, not the whole view.

    The old implementation compared object area with the 256x256 crop area,
    assigning inherently low scores to small targets.  That made correct
    detections lose the fusion gate exactly when the object shrank.
    """
    expectedWidth = expectedHeight = None
    if (
        predictedMotion is not None
        and predictedMotion.horizontalSizeRad > 0.0
        and predictedMotion.verticalSizeRad > 0.0
    ):
        expectedWidth = view.spec.outputWidthPx * math.tan(
            predictedMotion.horizontalSizeRad * 0.5
        ) / max(math.tan(view.spec.bfov.horizontalFovRad * 0.5), 1e-6)
        expectedHeight = view.spec.outputHeightPx * math.tan(
            predictedMotion.verticalSizeRad * 0.5
        ) / max(math.tan(view.spec.bfov.verticalFovRad * 0.5), 1e-6)
    if expectedWidth is None or expectedHeight is None:
        # Conservative neutral fallback during frame-zero motion bootstrap.
        expectedWidth = max(8.0, view.spec.outputWidthPx * 0.18)
        expectedHeight = max(8.0, view.spec.outputHeightPx * 0.18)
    widthResidual = abs(math.log(max(box.widthPx, 1e-3) / max(expectedWidth, 1e-3)))
    heightResidual = abs(math.log(max(box.heightPx, 1e-3) / max(expectedHeight, 1e-3)))
    return float(np.clip(math.exp(-0.5 * (widthResidual + heightResidual)), 0.0, 1.0))


def _projectionQuality(box: BBoxXYWH, view: LocalView) -> tuple[float, float]:
    width = float(view.spec.outputWidthPx)
    height = float(view.spec.outputHeightPx)
    centerX = box.xPx + box.widthPx / 2.0
    centerY = box.yPx + box.heightPx / 2.0
    halfDiagonal = max(math.hypot(width / 2.0, height / 2.0), 1e-9)
    normalizedRadius = math.hypot(centerX - width / 2.0, centerY - height / 2.0) / halfDiagonal
    edgeMargin = min(
        box.xPx,
        box.yPx,
        width - (box.xPx + box.widthPx),
        height - (box.yPx + box.heightPx),
    ) / max(width, height, 1.0)
    return max(0.0, normalizedRadius), max(0.0, edgeMargin)


def _requireFrame(frame: FramePacket | None) -> FramePacket:
    if frame is None:
        raise DecodeError("input source is empty")
    return frame


def _startProcessing(timer: TimeCounter | None) -> None:
    if timer is not None:
        timer.startProcessing()


def _stopProcessing(timer: TimeCounter | None) -> None:
    if timer is not None:
        timer.stopProcessing()


def _profile(profiler: RuntimeProfiler | None, name: str):
    if profiler is None or not profiler.enabled:
        from contextlib import nullcontext

        return nullcontext()
    return profiler.track(name)


def _profileNow(profiler: RuntimeProfiler | None) -> int | None:
    return perf_counter_ns() if profiler is not None and profiler.enabled else None


def _startProfileFrame(
    profiler: RuntimeProfiler | None,
    frameIndex: int,
    decodeStartedNs: int | None,
) -> None:
    if profiler is None or not profiler.enabled:
        return
    profiler.startFrame(frameIndex)
    if decodeStartedNs is not None:
        profiler.record("decode", perf_counter_ns() - decodeStartedNs)


def _finishProfileFrame(
    profiler: RuntimeProfiler | None,
    totalStartedNs: int | None,
    **metadata: object,
) -> None:
    if profiler is None or not profiler.enabled:
        return
    profiler.annotateFrame(**metadata)
    if totalStartedNs is not None:
        profiler.record("total", perf_counter_ns() - totalStartedNs)
    profiler.finishFrame()


def _recordBackendProfile(
    profiler: RuntimeProfiler | None,
    backend: TrackerBackend,
) -> None:
    if profiler is None or not profiler.enabled:
        return
    values = getattr(backend, "lastProfile", {})
    if not isinstance(values, dict):
        return
    for name in ("preprocess", "hostToDevice", "cudaForward"):
        value = values.get(name)
        if isinstance(value, (int, float)):
            profiler.record(name, int(value))
    profiler.appendFrameMetadata(
        "backendBatches",
        {
            name: value
            for name, value in values.items()
            if isinstance(value, (bool, int, float, str))
        },
    )


def _recordGeometryProfile(
    profiler: RuntimeProfiler | None,
    geometry: SphericalGeometry,
) -> None:
    if profiler is None or not profiler.enabled:
        return
    values = getattr(geometry, "lastProfile", {})
    if not isinstance(values, dict) or not values:
        return
    for name in ("frameToDevice", "gpuCrop", "gpuGeometryTotal"):
        value = values.get(name)
        if isinstance(value, (int, float)):
            profiler.record(name, int(value))
    profiler.appendFrameMetadata(
        "geometryBatches",
        {
            name: value
            for name, value in values.items()
            if isinstance(value, (bool, int, float, str))
        },
    )


def _recordPipelineProfile(
    profiler: RuntimeProfiler | None,
    pipelineReader: _PrefetchReader,
    controller: TrackControllerImpl,
) -> None:
    if profiler is None or not profiler.enabled:
        return
    readerValues = pipelineReader.lastProfile
    for source in (readerValues, controller.lastPipelineProfile):
        for name, value in source.items():
            if name.endswith("Ns") and isinstance(value, (int, float)):
                profiler.record(name, int(value))
        profiler.appendFrameMetadata("pipelineBatches", source)


__all__ = [
    "RuntimeBundle",
    "buildRuntime",
    "closeRuntime",
    "closeBackend",
    "finalizeSink",
    "openSink",
    "runTracking",
]
