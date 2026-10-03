"""Circular horizontal intervals used by ERP bounding boxes."""

from __future__ import annotations

from math import isfinite

import numpy as np

from track360.core.errors import GeometryError
from track360.core.types import BBoxXYWH


def wrapPixelX(xPx: float, frameWidthPx: int) -> float:
    """Normalize a horizontal ERP coordinate to [0, frameWidthPx)."""
    _requireWidth(frameWidthPx)
    if not isfinite(xPx):
        raise GeometryError("xPx must be finite")
    return xPx % frameWidthPx


def minimalCircularInterval(xCoordinatesPx: np.ndarray, frameWidthPx: int) -> tuple[float, float]:
    """Return the shortest circular interval containing horizontal samples."""
    _requireWidth(frameWidthPx)
    coordinates = np.asarray(xCoordinatesPx, dtype=np.float64).reshape(-1)
    if coordinates.size == 0 or not np.isfinite(coordinates).all():
        raise GeometryError("xCoordinatesPx must be a non-empty finite array")
    normalized = np.sort(np.mod(coordinates, frameWidthPx))
    if normalized.size == 1:
        return float(normalized[0]), 0.0
    gaps = np.diff(normalized)
    wrapGap = normalized[0] + frameWidthPx - normalized[-1]
    allGaps = np.concatenate((gaps, np.asarray((wrapGap,), dtype=np.float64)))
    largestGapIndex = int(np.argmax(allGaps))
    startIndex = (largestGapIndex + 1) % normalized.size
    startPx = float(normalized[startIndex])
    widthPx = float(frameWidthPx - allGaps[largestGapIndex])
    return startPx, max(widthPx, 0.0)


def splitSeamBox(bbox: BBoxXYWH, frameWidthPx: int) -> tuple[BBoxXYWH, ...]:
    """Split a circular ERP box into one or two ordinary image boxes."""
    _requireWidth(frameWidthPx)
    if bbox.widthPx > frameWidthPx:
        raise GeometryError(
            f"seam bbox width cannot exceed frame width: {bbox.widthPx} > {frameWidthPx}"
        )
    xPx = wrapPixelX(bbox.xPx, frameWidthPx)
    if xPx + bbox.widthPx <= frameWidthPx:
        return (BBoxXYWH(xPx, bbox.yPx, bbox.widthPx, bbox.heightPx),)
    firstWidthPx = frameWidthPx - xPx
    secondWidthPx = bbox.widthPx - firstWidthPx
    return (
        BBoxXYWH(xPx, bbox.yPx, firstWidthPx, bbox.heightPx),
        BBoxXYWH(0.0, bbox.yPx, secondWidthPx, bbox.heightPx),
    )


def containsCircularX(xPx: float, bbox: BBoxXYWH, frameWidthPx: int) -> bool:
    """Test whether a horizontal point lies inside a circular ERP interval."""
    _requireWidth(frameWidthPx)
    if bbox.widthPx > frameWidthPx:
        raise GeometryError("bbox width cannot exceed frame width")
    relativePx = (wrapPixelX(xPx, frameWidthPx) - wrapPixelX(bbox.xPx, frameWidthPx)) % frameWidthPx
    return relativePx <= bbox.widthPx + 1e-9


def _requireWidth(frameWidthPx: int) -> None:
    if isinstance(frameWidthPx, bool) or frameWidthPx <= 0:
        raise GeometryError(f"frameWidthPx must be a positive integer, actual={frameWidthPx}")
