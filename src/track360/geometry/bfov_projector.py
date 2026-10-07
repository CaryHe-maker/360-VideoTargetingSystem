"""BFoV projection helpers for RGB ERP crops."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from track360.core.errors import GeometryError
from track360.core.types import FramePacket, LocalView, ViewSpec
from track360.geometry.projection_math import (
    unitVectorsToErpPixels,
    viewPixelsToUnitVectors,
)


@dataclass(frozen=True, slots=True)
class BfovProjector:
    """Project ERP frames into local views."""

    boundarySamplesPerEdge: int = 65

    def __post_init__(self) -> None:
        _requireBoundarySamplesPerEdge(self.boundarySamplesPerEdge)

    def cropView(self, frame: FramePacket, spec: ViewSpec) -> LocalView:
        """Crop one local RGB view."""
        _requireFrame(frame)
        _requireViewSpec(spec)
        localX, localY = _localPixelGrid(spec.outputWidthPx, spec.outputHeightPx)
        vectors = viewPixelsToUnitVectors(localX, localY, spec)
        sampleX, sampleY = unitVectorsToErpPixels(
            vectors,
            frame.rgb.shape[1],
            frame.rgb.shape[0],
            pixelCenters=True,
        )
        rgb = _sampleRgb(frame.rgb, sampleX, sampleY)
        return LocalView(spec=spec, rgb=rgb)

    def cropViews(
        self,
        frame: FramePacket,
        specs: Sequence[ViewSpec],
    ) -> list[LocalView]:
        """Crop multiple views while preserving the input order."""
        return [self.cropView(frame, spec) for spec in specs]


def _localPixelGrid(
    viewWidthPx: int,
    viewHeightPx: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    _requirePositiveSize("viewWidthPx", viewWidthPx)
    _requirePositiveSize("viewHeightPx", viewHeightPx)
    xCoords = np.arange(viewWidthPx, dtype=np.float64) + 0.5
    yCoords = np.arange(viewHeightPx, dtype=np.float64) + 0.5
    return np.meshgrid(xCoords, yCoords)


def _sampleRgb(
    image: NDArray[np.uint8],
    sampleX: NDArray[np.float64],
    sampleY: NDArray[np.float64],
) -> NDArray[np.uint8]:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise GeometryError("rgb image must have shape [H, W, 3] and dtype uint8")
    weights = _prepareBilinearWeights(sampleX, sampleY, image.shape[1], image.shape[0])
    x0, x1, y0, y1, wx, wy = weights
    topLeft = image[y0, x0].astype(np.float64)
    topRight = image[y0, x1].astype(np.float64)
    bottomLeft = image[y1, x0].astype(np.float64)
    bottomRight = image[y1, x1].astype(np.float64)
    top = topLeft * (1.0 - wx)[..., np.newaxis] + topRight * wx[..., np.newaxis]
    bottom = bottomLeft * (1.0 - wx)[..., np.newaxis] + bottomRight * wx[..., np.newaxis]
    sampled = top * (1.0 - wy)[..., np.newaxis] + bottom * wy[..., np.newaxis]
    return np.clip(np.rint(sampled), 0.0, 255.0).astype(np.uint8)


def _prepareBilinearWeights(
    sampleX: NDArray[np.float64],
    sampleY: NDArray[np.float64],
    widthPx: int,
    heightPx: int,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    if sampleX.shape != sampleY.shape:
        raise GeometryError(
            f"sample coordinate shapes must match: {sampleX.shape} != {sampleY.shape}"
        )
    if not np.isfinite(sampleX).all() or not np.isfinite(sampleY).all():
        raise GeometryError("sample coordinates must be finite")
    if widthPx <= 0 or heightPx <= 0:
        raise GeometryError("image dimensions must be positive")
    x = np.mod(sampleX, widthPx)
    y = np.clip(sampleY, 0.0, float(heightPx - 1))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = (x0 + 1) % widthPx
    y1 = np.minimum(y0 + 1, heightPx - 1)
    return x0, x1, y0, y1, x - x0, y - y0


def _requireBoundarySamplesPerEdge(samples: int) -> None:
    if isinstance(samples, bool) or samples < 2:
        raise GeometryError(
            f"boundarySamplesPerEdge must be an integer >= 2, actual={samples}"
        )


def _requireFrame(frame: FramePacket) -> None:
    if frame.rgb.ndim != 3 or frame.rgb.shape[2] != 3:
        raise GeometryError("frame.rgb must have shape [H, W, 3]")


def _requireViewSpec(spec: ViewSpec) -> None:
    if spec.outputWidthPx <= 0 or spec.outputHeightPx <= 0:
        raise GeometryError("view dimensions must be positive")


def _requirePositiveSize(name: str, value: int) -> None:
    if isinstance(value, bool) or value <= 0:
        raise GeometryError(f"{name} must be a positive integer, actual={value}")
