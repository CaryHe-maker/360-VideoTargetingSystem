"""Stage-oriented PNG recorder for manual tracking diagnostics."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from track360.core.config import VisualizationConfig
from track360.core.errors import ProtocolError
from track360.core.types import FramePacket, LocalObservation, LocalView, ProjectedObservation
from track360.visualization.image import drawBoxRgb
from track360.visualization.png import writeRgbPng

LOCAL_RGB_STAGE = "local_rgb"
BACKEND_BOX_STAGE = "backend_box"
GEOMETRY_BOX_STAGE = "geometry_box"


@dataclass(frozen=True, slots=True)
class VisualizationRecorder:
    """Write selected intermediate results without changing tracking decisions."""

    config: VisualizationConfig

    def recordLocalRgb(
        self,
        frame: FramePacket,
        views: Sequence[LocalView],
    ) -> tuple[Path, ...]:
        """Write the raw RGB crop for each local view."""
        if not self._active(LOCAL_RGB_STAGE):
            return ()
        _requireUniqueViewIds(views)
        return tuple(
            writeRgbPng(self._artifactPath(frame, LOCAL_RGB_STAGE, view.spec.viewId), view.rgb)
            for view in views
        )

    def recordBackendBoxes(
        self,
        frame: FramePacket,
        views: Sequence[LocalView],
        observations: Sequence[LocalObservation],
    ) -> tuple[Path, ...]:
        """Write each backend-local target box over its source local RGB view."""
        if not self._active(BACKEND_BOX_STAGE):
            return ()
        viewById = _indexViews(views)
        _requireUniqueObservationIds(observations)
        artifacts: list[Path] = []
        for observation in observations:
            view = _requireView(viewById, observation.viewId)
            annotatedRgb = drawBoxRgb(
                view.rgb,
                observation.bbox,
                label=(
                    f"fuseScore={observation.fusedScore:.3f}/"
                    f"{(observation.appearanceProbability or 0.0):.3f}"
                ),
            )
            artifacts.append(
                writeRgbPng(
                    self._artifactPath(frame, BACKEND_BOX_STAGE, observation.viewId),
                    annotatedRgb,
                )
            )
        return tuple(artifacts)

    def recordGeometryBoxes(
        self,
        frame: FramePacket,
        observations: Sequence[ProjectedObservation],
    ) -> tuple[Path, ...]:
        """Write each geometry-projected target box over the original ERP RGB frame."""
        if not self._active(GEOMETRY_BOX_STAGE):
            return ()
        _requireUniqueObservationIds(observations)
        return tuple(
            writeRgbPng(
                self._artifactPath(frame, GEOMETRY_BOX_STAGE, observation.viewId),
                drawBoxRgb(
                    frame.rgb,
                    observation.bbox,
                    wrapHorizontal=True,
                    label=(
                        f"score={_singleScore(observation):.3f}/"
                        f"{observation.motionScore:.3f}/"
                        f"{_appearanceProbability(observation):.3f}/"
                        f"{observation.envelopeInflation:.2f}"
                    ),
                ),
            )
            for observation in observations
        )

    def _active(self, stage: str) -> bool:
        return self.config.enabled and stage in self.config.stages

    def _artifactPath(self, frame: FramePacket, stage: str, viewId: int) -> Path:
        sequenceName = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(frame.sequenceId)).strip("._")
        if not sequenceName:
            sequenceName = "sequence"
        return (
            self.config.outputRoot
            / sequenceName
            / f"frame_{int(frame.frameIndex):06d}"
            / stage
            / f"view_{viewId:04d}.png"
        )


def _singleScore(observation: ProjectedObservation) -> float:
    return (
        observation.fusedScore
        if observation.singleScore is None
        else observation.singleScore
    )


def _appearanceProbability(observation: ProjectedObservation) -> float:
    return (
        observation.appearanceScore
        if observation.appearanceProbability is None
        else observation.appearanceProbability
    )


def _indexViews(views: Sequence[LocalView]) -> dict[int, LocalView]:
    _requireUniqueViewIds(views)
    return {view.spec.viewId: view for view in views}


def _requireUniqueViewIds(views: Sequence[LocalView]) -> None:
    viewIds = [view.spec.viewId for view in views]
    if len(viewIds) != len(set(viewIds)):
        raise ProtocolError("visualization local viewIds must be unique")


def _requireUniqueObservationIds(
    observations: Sequence[LocalObservation] | Sequence[ProjectedObservation],
) -> None:
    viewIds = [observation.viewId for observation in observations]
    if len(viewIds) != len(set(viewIds)):
        raise ProtocolError("visualization observation viewIds must be unique")


def _requireView(viewById: Mapping[int, LocalView], viewId: int) -> LocalView:
    try:
        return viewById[viewId]
    except KeyError as error:
        message = f"visualization observation has no local view: viewId={viewId}"
        raise ProtocolError(message) from error


__all__ = ["VisualizationRecorder"]
