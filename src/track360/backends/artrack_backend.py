"""RGB-only tracker backend facade for ARTrackV2."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter_ns

from track360.backends.artrack_model import ARTrackBackend
from track360.backends.observation import buildRgbObservation
from track360.core.errors import ModelError, ProtocolError
from track360.core.protocols import TrackerBackend as TrackerBackendProtocol
from track360.core.types import (
    BBoxXYWH,
    LocalObservation,
    LocalView,
)


class TrackerBackendImpl(TrackerBackendProtocol):
    """Own one RGB ARTrackV2 session, its frame-0 template, and observations."""

    def __init__(self, artrackBackend: ARTrackBackend) -> None:
        self._artrackBackend = artrackBackend
        self._usesTrajectory = artrackBackend.trajectoryLength > 0
        self._template: object | None = None
        self._initialized = False
        self._closed = False

    @property
    def lastProfile(self) -> dict[str, int | float | bool | str]:
        return self._artrackBackend.lastProfile

    def initialize(self, template: LocalView, templateBox: BBoxXYWH) -> None:
        if self._closed:
            raise ProtocolError("tracker backend is closed")
        if self._initialized:
            raise ProtocolError("tracker backend is already initialized")
        self._template = self._artrackBackend.encodeTemplateView(template, templateBox)
        # Stateless passes use their own encoding of the template, made on first use.
        self._detachedSource = (_copyView(template), templateBox)
        self._detachedTemplate = None
        self._initialized = True

    def infer(self, views: Sequence[LocalView]) -> Sequence[LocalObservation]:
        _validateViewSequence(views)
        return self._inferViews(views)

    def _inferViews(self, views: Sequence[LocalView]) -> tuple[LocalObservation, ...]:
        if self._closed:
            raise ProtocolError("tracker backend is closed")
        if not self._initialized:
            raise ProtocolError("tracker backend has not been initialized")
        templateFeatures = (self._template,)
        inferenceStartedNs = perf_counter_ns()
        priorBoxes = tuple(view.spec.priorBox for view in views)
        fovs = tuple(
            (view.spec.bfov.horizontalFovRad, view.spec.bfov.verticalFovRad) for view in views
        )
        if views and all(box is not None for box in priorBoxes):
            predictions = self._artrackBackend.inferBatch(
                tuple(view.rgb for view in views),
                templateFeatures,
                fovs,
                priorBoxes=priorBoxes,
                trajectories=(
                    tuple(view.spec.trajectory for view in views)
                    if self._usesTrajectory
                    else None
                ),
            )
        else:
            predictions = self._artrackBackend.inferBatch(
                tuple(view.rgb for view in views), templateFeatures, fovs
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
        memory = getattr(self._template, "memory", None)
        return {} if memory is None else dict(memory)

    def restoreState(self, state: dict[str, object]) -> None:
        memory = getattr(self._template, "memory", None)
        if memory is not None:
            memory.clear()
            memory.update(state)

    def resetState(self) -> None:
        """Forget what the tracker accumulated; it starts from the template again."""
        self.restoreState({})

    def close(self) -> None:
        if self._closed:
            return
        self._artrackBackend.close()
        self._template = None
        self._closed = True


def _validateViewSequence(views: Sequence[LocalView]) -> None:
    viewIds = [view.spec.viewId for view in views]
    if len(viewIds) != len(set(viewIds)):
        raise ProtocolError("tracker infer views must have unique viewIds")


def _copyView(view: LocalView) -> LocalView:
    rgb = view.rgb.copy()
    rgb.setflags(write=False)
    return LocalView(spec=view.spec, rgb=rgb)


TrackerBackend = TrackerBackendImpl

__all__ = ["TrackerBackend", "TrackerBackendImpl"]
