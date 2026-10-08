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
    UNCALIBRATED_STAGE3_SCORE_CALIBRATION,
    ScoreCalibration,
    TrackControllerImpl,
    calibrateBackendFusedScore,
    calibrateLocalAppearanceProbabilities,
    composeSingleScore,
    loadScoreCalibration,
    scoreViewCenterMotion,
)
from track360.core.config import AppConfig, ModelConfig
from track360.core.errors import ConfigError, DecodeError, GeometryError
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
)
from track360.geometry import GpuGeometryImpl, SphericalGeometryImpl
from track360.io.result_sink import FileResultSink
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
    scoreCalibration: ScoreCalibration
    useMotionScore: bool
    verifier: Any | None = None
    recorder: VisualizationRecorder | None = None


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
    allowUncalibratedScoring: bool = False,
    profile: bool = False,
) -> RuntimeBundle:
    tuning = config.backendTuning
    seedEverything(config.reproducibility)
    if tuning.alignedSearch and config.geometry.resampler == "cuda":
        raise ConfigError(
            "backendTuning.alignedSearch is not implemented for geometry.resampler: cuda"
        )
    if geometryFactory is not None:
        geometry = geometryFactory(config.geometry.boundarySamplesPerEdge)
    elif config.geometry.resampler == "cuda":
        geometry = GpuGeometryImpl(
            boundarySamplesPerEdge=config.geometry.boundarySamplesPerEdge,
            profileEnabled=profile,
        )
    else:
        geometry = SphericalGeometryImpl(
            boundarySamplesPerEdge=config.geometry.boundarySamplesPerEdge,
            useRemap=config.geometry.resampler == "opencv",
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
    if config.scoring.calibrationArtifact is not None:
        scoreCalibration = loadScoreCalibration(
            config.scoring.calibrationArtifact,
            checkpointPath=config.model.weights,
            candidateMinScore=config.tracking.candidateMinScore,
            requireCheckpointHashMatch=config.scoring.requireCheckpointHashMatch,
        )
    elif (
        allowUncalibratedScoring
        or config.model.variant.lower().replace("-", "_") == "artrackv2_b_256"
    ):
        scoreCalibration = UNCALIBRATED_STAGE3_SCORE_CALIBRATION
    else:
        raise ValueError("production runtime requires a score calibration artifact")
    verifier = None
    if tuning.lossHandling:
        from track360.backends.appearance import AppearanceVerifier

        verifier = AppearanceVerifier(
            tuning.verifierModel, Path(config.model.weights).parent / "hub"
        )
    return RuntimeBundle(
        geometry=geometry,
        controller=controller,
        backend=backend,
        sink=sink,
        scoreCalibration=scoreCalibration,
        recorder=recorder,
        useMotionScore=tuning.useMotionScore,
        verifier=verifier,
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
    scoreCalibration: ScoreCalibration,
    useMotionScore: bool,
    probe: Any | None = None,
    verifier: Any | None = None,
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
                        _recordPipelineProfile(profiler, pipelineReader, controller)
                        try:
                            with _profile(profiler, "controller"):
                                plan = controller.beginFrame(frame)
                        except Exception as error:
                            result = _fallbackFrameResult(
                                controller,
                                frame,
                                error,
                                backendRevision=getattr(backend, "templateRevision", None),
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
                                rawObservation = backend.infer((view,), plan.templateCommand)[0]
                            probed = (view, rawObservation)
                            if recorder is not None and hasattr(recorder, "setActiveTemplateFrame"):
                                recorder.setActiveTemplateFrame(  # type: ignore[attr-defined]
                                    int(frame.frameIndex),
                                    getattr(backend, "activeTemplateFrameIndex", 0),
                                )
                            _recordBackendProfile(profiler, backend)
                            with _profile(profiler, "calibration"):
                                observation = calibrateLocalAppearanceProbabilities(
                                    (rawObservation,),
                                    scoreCalibration,
                                )[0]
                            with _profile(profiler, "projection"):
                                projected = _projectValidObservation(
                                    frame=frame,
                                    view=view,
                                    observation=observation,
                                    predictedMotion=plan.predictedMotion,
                                    geometry=geometry,
                                    scoreCalibration=scoreCalibration,
                                    useMotionScore=useMotionScore,
                                )
                            # PostTrainV2.4 keeps recorder inputs host-only at this boundary.
                            # Recorders only consume the compatibility RGB and view
                            # metadata.  Never retain CUDA tensors after this frame.
                            if recorder is not None:
                                visualization = (
                                    LocalView(spec=view.spec, rgb=view.rgb, deviceRgb=None),
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
                                if plan.scanViews:
                                    forwardCount += len(plan.scanViews)
                                    candidates = _scanCandidates(
                                        frame=frame,
                                        plan=plan,
                                        geometry=geometry,
                                        backend=backend,
                                        verifier=verifier,
                                        scoreCalibration=scoreCalibration,
                                        useMotionScore=useMotionScore,
                                    )
                            with _profile(profiler, "controller"):
                                result = controller.consume(plan, projected, candidates)
                            if verifier is not None:
                                # A doubted frame must not shape the tracker's appearance
                                # memory; a jump to a candidate starts it afresh.
                                if controller.lastFrameReacquired:
                                    backend.resetState()  # type: ignore[attr-defined]
                                elif controller.lastFrameSuspect and savedState is not None:
                                    backend.restoreState(savedState)  # type: ignore[attr-defined]
                        except Exception as error:
                            # A failed frame (including an OOM converted by the backend)
                            # must not keep its CUDA view alive until the next frame.
                            visualization = None
                            result = _fallbackFrameResult(
                                controller,
                                frame,
                                error,
                                backendRevision=getattr(backend, "templateRevision", None),
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
    *,
    backendRevision: int | None = None,
):
    """Convert any expected runtime failure into one invalid frame result."""
    print(
        "[runtime] frame failed; emitted zero-confidence fallback: "
        f"sequence={frame.sequenceId}, frame={int(frame.frameIndex)}, reason={error}",
        file=sys.stderr,
    )
    return controller.commitFallback(
        frame,
        backendRevision=backendRevision,
        reason=type(error).__name__,
    )


def closeRuntime(runtime: RuntimeBundle) -> None:
    """Release Geometry CUDA state before the backend empties the CUDA cache."""
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
    scoreCalibration: ScoreCalibration,
    useMotionScore: bool,
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
        else calibrateBackendFusedScore(observation.fusedScore, scoreCalibration)
    )
    if not useMotionScore:
        # ARTrack already scores localization against its template/search crop.
        # The legacy spherical motion prior can reject a correct appearance hit.
        singleScore = float(np.clip(appearanceProbability, 0.0, 1.0))
    else:
        singleScore = composeSingleScore(
            appearanceProbability,
            motion.effectiveProbability,
            scoreCalibration,
        )
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
    scoreCalibration: ScoreCalibration,
    useMotionScore: bool,
) -> tuple[ProjectedObservation, ...]:
    """Boxes found in the plan's scan views, each with its similarity to the template."""
    views = {view.spec.viewId: view for view in geometry.cropViews(frame, plan.scanViews)}
    found = backend.inferDetached(tuple(views.values()))  # type: ignore[attr-defined]
    candidates = []
    for local in calibrateLocalAppearanceProbabilities(found, scoreCalibration):
        view = views[local.viewId]
        projected = _projectValidObservation(
            frame=frame,
            view=view,
            observation=local,
            predictedMotion=None,
            geometry=geometry,
            scoreCalibration=scoreCalibration,
            useMotionScore=useMotionScore,
        )
        if projected is not None:
            candidates.append(
                replace(
                    projected, appearanceSimilarity=verifier.similarity(view, local.bbox)
                )
            )
    return tuple(candidates)


def _projectValidObservation(
    *,
    frame: FramePacket,
    view: LocalView,
    observation: LocalObservation,
    predictedMotion: MotionState3D | None,
    geometry: SphericalGeometry,
    scoreCalibration: ScoreCalibration,
    useMotionScore: bool,
) -> ProjectedObservation | None:
    """Project the local box onto the sphere; ``None`` when it has no valid projection."""
    try:
        return _projectObservation(
            frame=frame,
            view=view,
            observation=observation,
            predictedMotion=predictedMotion,
            geometry=geometry,
            scoreCalibration=scoreCalibration,
            useMotionScore=useMotionScore,
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
