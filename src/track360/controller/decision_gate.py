"""Single-frame aggregate consumed by the template policy."""

from __future__ import annotations

from dataclasses import dataclass

from track360.core.types import BBoxXYWH, BFoV


@dataclass(frozen=True, slots=True)
class FrameAggregate:
    """The best geometrically consistent cluster in one frame."""

    bfov: BFoV
    bbox: BBoxXYWH
    confidence: float
    decisionScore: float
    sourceViewIds: tuple[int, ...]
    representativeViewId: int
    localBox: BBoxXYWH | None
    supported: bool
    clusterCount: int = 1
    agreementScore: float = 1.0


__all__ = ["FrameAggregate"]
