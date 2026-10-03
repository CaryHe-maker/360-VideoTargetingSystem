"""Structural interfaces between project modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from track360.core.types import (
    AirSim360Record,
    BBoxXYWH,
    BFoV,
    FramePacket,
    InitializationPlan,
    LocalBoxProjection,
    LocalObservation,
    LocalView,
    MotionState3D,
    ProjectedObservation,
    RoutedInferenceTask,
    RoutedLocalObservation,
    SearchPlan,
    SphericalPoint,
    TemplateCommand,
    TrackResult,
    ViewSpec,
)


@dataclass(frozen=True, slots=True)
class MoreViewsRequired:
    """A bounded same-frame escalation request from the controller."""

    plan: SearchPlan


@dataclass(frozen=True, slots=True)
class FrameCommitted:
    """The single final result committed for a frame transaction."""

    result: TrackResult


@runtime_checkable
class SphericalGeometry(Protocol):
    """Convert between ERP pixels, spherical BFoVs, and local views."""

    def bboxToBfov(
        self,
        bbox: BBoxXYWH,
        frameWidthPx: int,
        frameHeightPx: int,
    ) -> BFoV: ...

    def cropViews(
        self,
        frame: FramePacket,
        specs: Sequence[ViewSpec],
    ) -> Sequence[LocalView]: ...

    def localBoxToBfov(
        self,
        localBox: BBoxXYWH,
        spec: ViewSpec,
    ) -> BFoV: ...

    def projectLocalBoxBoundary(
        self,
        localBox: BBoxXYWH,
        spec: ViewSpec,
        frameWidthPx: int,
        frameHeightPx: int,
    ) -> LocalBoxProjection: ...

    def bfovToBbox(
        self,
        bfov: BFoV,
        frameWidthPx: int,
        frameHeightPx: int,
    ) -> BBoxXYWH: ...


@runtime_checkable
class MotionEstimator(Protocol):
    """Maintain and predict the controller-owned spherical motion state."""

    def initialize(
        self,
        point: SphericalPoint,
        timestampNs: int,
    ) -> MotionState3D: ...

    def predict(self, timestampNs: int) -> MotionState3D: ...

    def update(
        self,
        point: SphericalPoint,
        timestampNs: int,
        observationConfidence: float,
    ) -> MotionState3D: ...


@runtime_checkable
class TrackerBackend(Protocol):
    """Own the local RGB tracker and all device-side resources."""

    def initialize(
        self,
        template: LocalView,
        templateBox: BBoxXYWH,
    ) -> None: ...

    def infer(
        self,
        views: Sequence[LocalView],
        command: TemplateCommand,
    ) -> Sequence[LocalObservation]: ...

    def inferTasks(
        self,
        tasks: Sequence[RoutedInferenceTask],
        command: TemplateCommand,
    ) -> Sequence[RoutedLocalObservation]: ...

    def close(self) -> None: ...


@runtime_checkable
class TrackController(Protocol):
    """Plan searches and atomically commit ordered tracking state."""

    def buildInitialization(
        self,
        frame: FramePacket,
        initialBox: BBoxXYWH | None = None,
        *,
        initialBfov: BFoV | None = None,
    ) -> InitializationPlan: ...

    def commitInitialization(
        self,
        plan: InitializationPlan,
    ) -> TrackResult: ...

    def plan(self, frame: FramePacket) -> SearchPlan: ...

    def update(
        self,
        plan: SearchPlan,
        observations: Sequence[ProjectedObservation],
    ) -> TrackResult: ...

    def consume(
        self,
        plan: SearchPlan,
        observations: Sequence[ProjectedObservation],
    ) -> MoreViewsRequired | FrameCommitted: ...


@runtime_checkable
class FrameSource(Protocol):
    """Read a strictly ordered sequence of aligned frame packets."""

    def open(self, uri: str) -> None: ...

    def read(self) -> FramePacket | None: ...

    def close(self) -> None: ...


@runtime_checkable
class ResultSink(Protocol):
    """Write one ordered result per frame and publish it atomically."""

    def open(self, destination: str) -> None: ...

    def write(self, result: TrackResult) -> None: ...

    def finalize(self, expectedFrameCount: int) -> None: ...


@runtime_checkable
class AirSim360DataSource(Protocol):
    """Read an AirSim360 sequence through the common frame contract."""

    def open(self, root: str, sequenceId: str | None = None) -> None: ...

    def read(self) -> FramePacket | None: ...

    def close(self) -> None: ...


@runtime_checkable
class PseudoTrackBuilder(Protocol):
    """Derive temporary training boxes from AirSim360 instance masks."""

    def buildInitialBox(
        self,
        frame: FramePacket,
        targetInstanceId: int,
    ) -> BBoxXYWH: ...

    def buildPseudoGroundTruth(
        self,
        frame: FramePacket,
        targetInstanceId: int,
    ) -> tuple[BBoxXYWH, bool]: ...


__all__ = [
    "AirSim360DataSource",
    "AirSim360Record",
    "FrameSource",
    "MotionEstimator",
    "FrameCommitted",
    "MoreViewsRequired",
    "PseudoTrackBuilder",
    "ResultSink",
    "SphericalGeometry",
    "TrackController",
    "TrackerBackend",
]
