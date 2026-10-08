"""BFoV projection helpers for RGB ERP crops."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from track360.core.errors import GeometryError
from track360.core.types import FramePacket, LocalView, ViewProjection, ViewSpec
from track360.geometry.projection_math import (
    unitVectorsToErpPixels,
    viewAxes,
    viewPixelsToUnitVectors,
)


@dataclass(frozen=True, slots=True)
class BfovProjector:
    """Project ERP frames into local views."""

    boundarySamplesPerEdge: int = 65
    # Sample with single-precision coordinates and OpenCV's remap instead of the
    # double-precision reference path.  Same sampling positions; values differ by
    # at most one intensity level.
    useRemap: bool = False

    def __post_init__(self) -> None:
        _requireBoundarySamplesPerEdge(self.boundarySamplesPerEdge)

    def cropView(self, frame: FramePacket, spec: ViewSpec) -> LocalView:
        """Crop one local RGB view."""
        _requireFrame(frame)
        _requireViewSpec(spec)
        if self.useRemap:
            return LocalView(spec=spec, rgb=_remapView(frame.rgb, spec))
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


def _remapView(image: NDArray[np.uint8], spec: ViewSpec) -> NDArray[np.uint8]:
    """Sample a view with ``cv2.remap``: the positions of the reference path.

    The view is separable in its own coordinates, so the directions are built from
    one row and one column of values.  ``BORDER_WRAP`` makes the ERP columns
    cyclic; rows never leave the frame because the row coordinate is clamped.
    """
    import cv2

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise GeometryError("rgb image must have shape [H, W, 3] and dtype uint8")
    frameHeightPx, frameWidthPx = image.shape[:2]
    widthPx, heightPx = spec.outputWidthPx, spec.outputHeightPx
    single = np.float32
    column = (np.arange(widthPx, dtype=single) + single(0.5)) / single(widthPx)
    row = (np.arange(heightPx, dtype=single) + single(0.5)) / single(heightPx)
    if spec.projection is ViewProjection.SPHERICAL:
        longitude = (column - single(0.5)) * single(spec.bfov.horizontalFovRad)
        latitude = (single(0.5) - row) * single(spec.bfov.verticalFovRad)
        cosLatitude = np.cos(latitude)[:, np.newaxis]
        along = cosLatitude * np.cos(longitude)[np.newaxis, :]
        across = cosLatitude * np.sin(longitude)[np.newaxis, :]
        upward = np.broadcast_to(np.sin(latitude)[:, np.newaxis], along.shape)
    else:
        horizontal = (single(2.0) * column - single(1.0)) * single(
            np.tan(spec.bfov.horizontalFovRad / 2.0)
        )
        vertical = (single(1.0) - single(2.0) * row) * single(
            np.tan(spec.bfov.verticalFovRad / 2.0)
        )
        shape = (heightPx, widthPx)
        along = np.ones(shape, dtype=single)
        across = np.broadcast_to(horizontal[np.newaxis, :], shape)
        upward = np.broadcast_to(vertical[:, np.newaxis], shape)
    forward, right, up = (
        axis.astype(single) for axis in viewAxes(spec.bfov.center, spec.bfov.rollRad)
    )
    x = along * forward[0] + across * right[0] + upward * up[0]
    y = along * forward[1] + across * right[1] + upward * up[1]
    z = along * forward[2] + across * right[2] + upward * up[2]
    yaw = np.arctan2(x, z)
    pitch = np.arctan2(y, np.hypot(x, z))
    mapX = np.mod(
        (yaw + single(np.pi)) * single(frameWidthPx / (2.0 * np.pi)) - single(0.5),
        single(frameWidthPx),
    )
    mapY = np.clip(
        (single(np.pi / 2.0) - pitch) * single(frameHeightPx / np.pi) - single(0.5),
        single(0.0),
        single(frameHeightPx - 1),
    )
    return cv2.remap(
        image,
        np.ascontiguousarray(mapX, dtype=single),
        np.ascontiguousarray(mapY, dtype=single),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


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
