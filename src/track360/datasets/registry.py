"""Extensible dataset entry points.

The tracker consumes :class:`FramePacket`, not a particular on-disk layout.
New formats can register a factory here without changing the tracking driver.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from track360.core.errors import DecodeError
from track360.core.types import FramePacket
from track360.datasets.airsim360_source import AirSim360DataSource
from track360.datasets.vot360 import Vot360DataSource


class DatasetSource(Protocol):
    frameCount: int

    def open(self, root: str, sequenceId: str | None = None) -> None: ...
    def read(self) -> FramePacket | None: ...
    def close(self) -> None: ...


SourceFactory = Callable[[], DatasetSource]
_FACTORIES: dict[str, SourceFactory] = {
    "airsim360": AirSim360DataSource,
    "360vot": Vot360DataSource,
}


def registerDatasetFormat(name: str, factory: SourceFactory) -> None:
    """Register a format name for application and training consumers."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("dataset format name must be non-empty")
    _FACTORIES[normalized] = factory


def openDataset(root: str, *, format: str = "auto", sequenceId: str | None = None) -> DatasetSource:
    """Create and open a source, keeping format selection outside the tracker."""
    normalized = format.strip().lower()
    if normalized == "auto":
        normalized = "airsim360"
    try:
        factory = _FACTORIES[normalized]
    except KeyError as error:
        raise DecodeError(f"unknown dataset format '{format}'") from error
    source = factory()
    source.open(root, sequenceId)
    return source


def registeredDatasetFormats() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


__all__ = ["DatasetSource", "openDataset", "registerDatasetFormat", "registeredDatasetFormats"]
