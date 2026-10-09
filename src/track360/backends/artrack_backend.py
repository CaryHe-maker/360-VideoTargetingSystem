"""RGB-only tracker backend facade for ARTrackV2."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter_ns

from track360.backends.artrack_model import ARTrackBackend
from track360.backends.observation import buildRgbObservation
from track360.backends.template_cache import TemplateCache
from track360.core.errors import ModelError, ProtocolError
from track360.core.protocols import TrackerBackend as TrackerBackendProtocol
from track360.core.types import (
    BBoxXYWH,
    LocalObservation,
    LocalView,
    TemplateCommand,
)


class TrackerBackendImpl(TrackerBackendProtocol):
    """Own one RGB ARTrackV2 session, its template cache, and observations."""

    def __init__(self, artrackBackend: ARTrackBackend) -> None:
        self._artrackBackend = artrackBackend
        self._usesTrajectory = artrackBackend.trajectoryLength > 0
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
        # Stateless passes use their own encoding of the template, made on first use.
        self._detachedSource = (_copyView(template), templateBox)
        self._detachedTemplate = None
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
        priorBoxes = tuple(view.spec.priorBox for view in views)
        if views and all(box is not None for box in priorBoxes):
            if any(item is not None for item in deviceViews):
                raise ProtocolError(
                    "search priors are not supported with CUDA-resampled views"
                )
            predictions = self._artrackBackend.inferBatch(
                tuple(view.rgb for view in views),
                templateFeatures,
                tuple(
                    (view.spec.bfov.horizontalFovRad, view.spec.bfov.verticalFovRad)
                    for view in views
                ),
                priorBoxes=priorBoxes,
                trajectories=(
                    tuple(view.spec.trajectory for view in views)
                    if self._usesTrajectory
                    else None
                ),
            )
        elif all(item is not None for item in deviceViews):
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

    def inferDetached(self, views: Sequence[LocalView]) -> tuple[LocalObservation, ...]:
        """Locate the target in views without using or changing the tracker's state.

        The pass sees the frame-0 template only: its appearance feature starts from
        the template.  A view's own trajectory is used when it has one; otherwise the
        prior box stands in for it.  Used to look for a lost target.
        Each view needs a prior box; a view whose box falls outside it yields no
        observation.
        """
        if self._closed or not self._initialized:
            raise ProtocolError("tracker backend is not ready")
        if any(view.spec.priorBox is None for view in views):
            raise ProtocolError("detached inference needs a prior box in every view")
        if self._detachedTemplate is None:
            self._detachedTemplate = self._artrackBackend.encodeTemplateView(
                *self._detachedSource
            )
        memory = getattr(self._detachedTemplate, "memory", None)
        observations = []
        for view in views:
            if memory is not None:
                memory.clear()
            started = perf_counter_ns()
            prediction = self._artrackBackend.inferBatch(
                (view.rgb,),
                (self._detachedTemplate,),
                ((view.spec.bfov.horizontalFovRad, view.spec.bfov.verticalFovRad),),
                priorBoxes=(view.spec.priorBox,),
                trajectories=(view.spec.trajectory,) if view.spec.trajectory else None,
            )[0]
            try:
                observations.append(
                    buildRgbObservation(view, prediction, perf_counter_ns() - started)
                )
            except ModelError:
                # The box left the view: this view holds no candidate.
                continue
        if memory is not None:
            memory.clear()
        return tuple(observations)

    def saveState(self) -> dict[str, object]:
        """The tracker's per-sequence state, to undo the effect of a frame."""
        memory = getattr(self._templates.snapshot().anchor.features, "memory", None)
        return {} if memory is None else dict(memory)

    def restoreState(self, state: dict[str, object]) -> None:
        memory = getattr(self._templates.snapshot().anchor.features, "memory", None)
        if memory is not None:
            memory.clear()
            memory.update(state)

    def resetState(self) -> None:
        """Forget what the tracker accumulated; it starts from the template again."""
        self.restoreState({})

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
