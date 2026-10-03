"""RGB-only tracker backend facade for ARTrackV2."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter_ns

from track360.backends.artrack_model import ARTrackBackend
from track360.backends.observation import buildRgbObservation
from track360.backends.template_cache import TemplateCache
from track360.core.errors import ProtocolError
from track360.core.protocols import TrackerBackend as TrackerBackendProtocol
from track360.core.types import (
    BBoxXYWH,
    LocalObservation,
    LocalView,
    RoutedInferenceTask,
    RoutedLocalObservation,
    TaskKey,
    TemplateCommand,
)


class TrackerBackendImpl(TrackerBackendProtocol):
    """Own one RGB ARTrackV2 session, its template cache, and observations."""

    def __init__(self, artrackBackend: ARTrackBackend) -> None:
        self._artrackBackend = artrackBackend
        self._templates = TemplateCache()
        self._previousViews: dict[int, LocalView] = {}
        self._previousViewsFrameIndex: int | None = None
        self._initialized = False
        self._closed = False

    @property
    def templateRevision(self) -> int:
        return self._templates.revision

    @property
    def lastProfile(self) -> dict[str, int | float | bool | str]:
        return self._artrackBackend.lastProfile

    @property
    def activeTemplateFrameIndex(self) -> int:
        return self._templates.activeTemplateFrameIndex

    def initialize(self, template: LocalView, templateBox: BBoxXYWH) -> None:
        if self._closed:
            raise ProtocolError("tracker backend is closed")
        if self._initialized:
            raise ProtocolError("tracker backend is already initialized")
        self._templates.initialize(self._artrackBackend, template, templateBox)
        self._initialized = True
        self._previousViews = {template.spec.viewId: _copyView(template)}
        self._previousViewsFrameIndex = 0

    def infer(
        self,
        views: Sequence[LocalView],
        command: TemplateCommand,
    ) -> Sequence[LocalObservation]:
        _validateViewSequence(views)
        observations = self._inferViews(views, command)
        self._rememberViews(views, int(command.frameIndex))
        return observations

    def inferTasks(
        self,
        tasks: Sequence[RoutedInferenceTask],
        command: TemplateCommand,
    ) -> Sequence[RoutedLocalObservation]:
        """Infer one mixed batch and bind every output to its immutable task identity."""
        taskKeys = tuple(task.key for task in tasks)
        if not tasks:
            return ()
        if len(taskKeys) != len(set(taskKeys)):
            raise ProtocolError("routed inference tasks must have unique TaskKeys")
        if any(task.key.sequenceId != taskKeys[0].sequenceId for task in tasks[1:]):
            raise ProtocolError("one routed batch cannot cross sequence boundaries")
        commandFrame = int(command.frameIndex)
        taskFrames = {int(task.key.frameIndex) for task in tasks}
        if any(frame not in {commandFrame, commandFrame + 1} for frame in taskFrames):
            raise ProtocolError("routed tasks may only contain the command frame and its successor")
        views = tuple(task.view for task in tasks)
        _validateViewSequence(views)
        observations = self._inferViews(views, command)
        formalViews = tuple(task.view for task in tasks if int(task.key.frameIndex) == commandFrame)
        self._rememberViews(formalViews, commandFrame)
        return tuple(
            RoutedLocalObservation(task.key, observation)
            for task, observation in zip(tasks, observations, strict=True)
        )

    @staticmethod
    def routeTasks(
        outputs: Sequence[RoutedLocalObservation],
        expectedKeys: Sequence[TaskKey],
    ) -> tuple[RoutedLocalObservation, ...]:
        """Restore deterministic TaskKey order and reject duplicates or missing slots."""
        expected = tuple(expectedKeys)
        if len(expected) != len(set(expected)):
            raise ProtocolError("expected routed inference keys must be unique")
        byKey = {item.key: item for item in outputs}
        if len(byKey) != len(outputs) or set(byKey) != set(expected):
            raise ProtocolError("routed inference output keys do not match requested TaskKeys")
        return tuple(byKey[key] for key in expected)

    def _inferViews(
        self,
        views: Sequence[LocalView],
        command: TemplateCommand,
    ) -> tuple[LocalObservation, ...]:
        if self._closed:
            raise ProtocolError("tracker backend is closed")
        if not self._initialized:
            raise ProtocolError("tracker backend has not been initialized")
        self._templates.apply(self._artrackBackend, command, self._previousViews)
        snapshot = self._templates.snapshot()
        # Preserve the cache ordering so ARTrackV2 can use the anchor together
        # with recent/stable online templates.  Passing only the anchor makes
        # every UPDATE_RECENT/UPDATE_STABLE command a no-op for the model.
        templateFeatures = snapshot.features
        self._activeTemplateFrameIndex = (
            int(snapshot.stable.frameIndex)
            if snapshot.stable is not None
            else int(snapshot.recent.frameIndex)
            if snapshot.recent is not None
            else int(snapshot.anchor.frameIndex)
        )
        inferenceStartedNs = perf_counter_ns()
        deviceViews = tuple(getattr(view, "deviceRgb", None) for view in views)
        if all(item is not None for item in deviceViews):
            predictions = self._artrackBackend.inferDeviceBatch(
                tuple(deviceViews),
                tuple((view.spec.outputWidthPx, view.spec.outputHeightPx) for view in views),
                templateFeatures,
                tuple(
                    (view.spec.bfov.horizontalFovRad, view.spec.bfov.verticalFovRad)
                    for view in views
                ),
            )
        else:
            predictions = self._artrackBackend.inferBatch(
                tuple(view.rgb for view in views),
                templateFeatures,
                tuple(
                    (view.spec.bfov.horizontalFovRad, view.spec.bfov.verticalFovRad)
                    for view in views
                ),
            )
        sharedInferenceNs = (perf_counter_ns() - inferenceStartedNs) // len(views) if views else 0
        return tuple(
            buildRgbObservation(view, prediction, sharedInferenceNs)
            for view, prediction in zip(views, predictions, strict=True)
        )

    def _rememberViews(self, views: Sequence[LocalView], frameIndex: int) -> None:
        if not views:
            return
        currentViews = {view.spec.viewId: _copyView(view) for view in views}
        if self._previousViewsFrameIndex == frameIndex:
            self._previousViews.update(currentViews)
        else:
            self._previousViews = currentViews
        self._previousViewsFrameIndex = frameIndex

    def close(self) -> None:
        if self._closed:
            return
        self._artrackBackend.close()
        self._templates.clear()
        self._previousViews.clear()
        self._previousViewsFrameIndex = None
        self._closed = True


def _validateViewSequence(views: Sequence[LocalView]) -> None:
    viewIds = [view.spec.viewId for view in views]
    if len(viewIds) != len(set(viewIds)):
        raise ProtocolError("tracker infer views must have unique viewIds")


def _copyView(view: LocalView) -> LocalView:
    rgb = view.rgb.copy()
    rgb.setflags(write=False)
    return LocalView(
        spec=view.spec,
        rgb=rgb,
        # Keep the normalized crop alive for a possible online-template
        # command.  GPU Geometry's host RGB is only a placeholder, so dropping
        # this tensor would make a later update encode black pixels.
        deviceRgb=view.deviceRgb,
    )


TrackerBackend = TrackerBackendImpl

__all__ = ["TrackerBackend", "TrackerBackendImpl"]
