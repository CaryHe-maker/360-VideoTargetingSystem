"""Concrete spherical geometry facade built on top of the projection helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import cos, pi, sin

import numpy as np
from numpy.typing import NDArray

from track360.core.errors import GeometryError
from track360.core.protocols import SphericalGeometry as SphericalGeometryProtocol
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FramePacket,
    LocalBoxProjection,
    LocalView,
    ViewProjection,
    ViewSpec,
)
from track360.geometry.bfov_projector import BfovProjector
from track360.geometry.projection_math import (
    cameraBasis,
    erpPixelToSphericalPoint,
    localPixelsToUnitVectors,
    makeSphericalPoint,
    unitVectorsToErpPixels,
    unitVectorToYawPitch,
    viewPixelsToUnitVectors,
)
from track360.geometry.seam import minimalCircularInterval, wrapPixelX


@dataclass(slots=True)
class SphericalGeometryImpl(SphericalGeometryProtocol):
    """Default geometry implementation for ERP crops and BFoV envelopes."""

    boundarySamplesPerEdge: int = 65
    _projector: BfovProjector = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _requireBoundarySamplesPerEdge(self.boundarySamplesPerEdge)
        self._projector = BfovProjector(self.boundarySamplesPerEdge)

    def bboxToBfov(self, bbox: BBoxXYWH, frameWidthPx: int, frameHeightPx: int) -> BFoV:
        _requireFrameDimensions(frameWidthPx, frameHeightPx)
        _requireErpBox(bbox, frameWidthPx, frameHeightPx)
        sampleX, sampleY = _sampleErpBoxBoundary(
            bbox.xPx,
            bbox.yPx,
            bbox.widthPx,
            bbox.heightPx,
            self.boundarySamplesPerEdge,
        )
        vectors = _erpSamplesToVectors(sampleX, sampleY, frameWidthPx, frameHeightPx)
        fitted = _fitBfovFromVectors(vectors)
        seamCrossing = bbox.xPx + bbox.widthPx > frameWidthPx
        if seamCrossing and (
            bbox.widthPx >= 0.45 * frameWidthPx
            or bbox.heightPx >= 0.70 * frameHeightPx
        ):
            # Boundary-vector means become unstable when a large seam-crossing
            # ERP box spans both sides of the camera or approaches both poles.
            # Its annotated ERP center remains exact and must drive tracking.
            center = erpPixelToSphericalPoint(
                (bbox.xPx + 0.5 * bbox.widthPx) % frameWidthPx,
                min(float(frameHeightPx), max(0.0, bbox.yPx + 0.5 * bbox.heightPx)),
                frameWidthPx,
                frameHeightPx,
            )
            return BFoV(
                center=center,
                horizontalFovRad=fitted.horizontalFovRad,
                verticalFovRad=fitted.verticalFovRad,
            )
        return fitted

    def cropViews(self, frame: FramePacket, specs: Sequence[ViewSpec]) -> list[LocalView]:
        return self._projector.cropViews(frame, specs)

    def localBoxToBfov(self, localBox: BBoxXYWH, spec: ViewSpec) -> BFoV:
        _requireViewSpec(spec)
        _requireLocalBox(localBox, spec.outputWidthPx, spec.outputHeightPx)
        if spec.projection is ViewProjection.SPHERICAL:
            return _sphericalBoxBfov(localBox, spec)
        sampleX, sampleY = _sampleErpBoxBoundary(
            localBox.xPx,
            localBox.yPx,
            localBox.widthPx,
            localBox.heightPx,
            self.boundarySamplesPerEdge,
        )
        return _fitBfovFromVectors(viewPixelsToUnitVectors(sampleX, sampleY, spec))

    def projectLocalBoxBoundary(
        self,
        localBox: BBoxXYWH,
        spec: ViewSpec,
        frameWidthPx: int,
        frameHeightPx: int,
    ) -> LocalBoxProjection:
        """Project a local boundary once and fit both output envelopes from it."""
        _requireViewSpec(spec)
        _requireLocalBox(localBox, spec.outputWidthPx, spec.outputHeightPx)
        _requireFrameDimensions(frameWidthPx, frameHeightPx)
        sampleX, sampleY = _sampleErpBoxBoundary(
            localBox.xPx,
            localBox.yPx,
            localBox.widthPx,
            localBox.heightPx,
            self.boundarySamplesPerEdge,
        )
        vectors = viewPixelsToUnitVectors(sampleX, sampleY, spec)
        erpX, erpY = unitVectorsToErpPixels(vectors, frameWidthPx, frameHeightPx)
        if spec.projection is ViewProjection.SPHERICAL:
            bfov = _sphericalBoxBfov(localBox, spec)
            # A box this large can contain a pole or wrap the whole frame, which its
            # boundary alone does not show: the ERP envelope also covers the interior.
            insideX, insideY = unitVectorsToErpPixels(
                viewPixelsToUnitVectors(*_sampleBoxInterior(localBox), spec),
                frameWidthPx,
                frameHeightPx,
            )
            envelopeX = np.concatenate((erpX, insideX))
            envelopeY = np.concatenate((erpY, insideY))
        else:
            bfov = _fitBfovFromVectors(vectors)
            envelopeX, envelopeY = erpX, erpY
        xPx, widthPx = minimalCircularInterval(envelopeX, frameWidthPx)
        yMin = float(np.min(envelopeY))
        yMax = float(np.max(envelopeY))
        bbox = BBoxXYWH(
            xPx=xPx,
            yPx=yMin,
            widthPx=widthPx,
            heightPx=max(yMax - yMin, float(np.finfo(np.float64).eps)),
        )
        # The envelope of the fitted BFoV is a perspective construction; a box from a
        # spherical view has no such second envelope.
        indirectBbox = (
            bbox
            if spec.projection is ViewProjection.SPHERICAL
            else self.bfovToBbox(bfov, frameWidthPx, frameHeightPx)
        )
        directArea = bbox.widthPx * bbox.heightPx
        indirectArea = indirectBbox.widthPx * indirectBbox.heightPx
        sphericalBoundary = tuple(
            makeSphericalPoint(
                float(np.arctan2(vector[0], vector[2])),
                float(np.arcsin(np.clip(vector[1], -1.0, 1.0))),
            )
            for vector in vectors
        )
        return LocalBoxProjection(
            bfov=bfov,
            bbox=bbox,
            sphericalBoundary=sphericalBoundary,
            erpBoundary=tuple(
                (float(xValue), float(yValue))
                for xValue, yValue in zip(erpX, erpY, strict=True)
            ),
            indirectBbox=indirectBbox,
            envelopeInflation=float(indirectArea / max(directArea, 1e-12)),
        )

    def bfovToBbox(self, bfov: BFoV, frameWidthPx: int, frameHeightPx: int) -> BBoxXYWH:
        _requireFrameDimensions(frameWidthPx, frameHeightPx)
        sampleX, sampleY = _sampleCanonicalRectBoundary(self.boundarySamplesPerEdge)
        vectors = localPixelsToUnitVectors(sampleX, sampleY, bfov, 1, 1)
        erpX, erpY = unitVectorsToErpPixels(vectors, frameWidthPx, frameHeightPx)
        xPx, widthPx = minimalCircularInterval(erpX, frameWidthPx)
        yMin = float(np.min(erpY))
        yMax = float(np.max(erpY))
        bbox = BBoxXYWH(xPx=xPx, yPx=yMin, widthPx=widthPx, heightPx=max(yMax - yMin, 0.0))
        _requireErpBox(bbox, frameWidthPx, frameHeightPx)
        return bbox


_INTERIOR_SAMPLES_PER_AXIS = 33
_MAX_HORIZONTAL_SPAN_RAD = 2.0 * pi - 1e-6
_MAX_VERTICAL_SPAN_RAD = pi - 1e-6


def _sphericalBoxBfov(localBox: BBoxXYWH, spec: ViewSpec) -> BFoV:
    """BFoV of a box in a spherical view: its center direction and angular spans.

    A spherical view is linear in angle, so the box maps to the spans directly, except
    that meridians converge away from the view's equator: there a box covers more
    longitude than it is wide, by the cosine of its latitude.  This is the extended
    BFoV of 360VOT, used there for regions too large for a tangent plane.
    """
    centerY = localBox.yPx + 0.5 * localBox.heightPx
    center = viewPixelsToUnitVectors(
        np.asarray([localBox.xPx + 0.5 * localBox.widthPx], dtype=np.float64),
        np.asarray([centerY], dtype=np.float64),
        spec,
    )[0]
    latitude = (0.5 - centerY / spec.outputHeightPx) * spec.bfov.verticalFovRad
    horizontal = (
        localBox.widthPx / spec.outputWidthPx * spec.bfov.horizontalFovRad * cos(latitude)
    )
    vertical = localBox.heightPx / spec.outputHeightPx * spec.bfov.verticalFovRad
    minimum = float(np.finfo(np.float64).eps)
    return BFoV(
        center=makeSphericalPoint(*unitVectorToYawPitch(tuple(center))),
        horizontalFovRad=min(_MAX_HORIZONTAL_SPAN_RAD, max(minimum, horizontal)),
        verticalFovRad=min(_MAX_VERTICAL_SPAN_RAD, max(minimum, vertical)),
    )


def _sampleBoxInterior(
    localBox: BBoxXYWH,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    edge = np.linspace(0.0, 1.0, _INTERIOR_SAMPLES_PER_AXIS, dtype=np.float64)
    gridX, gridY = np.meshgrid(
        localBox.xPx + edge * localBox.widthPx, localBox.yPx + edge * localBox.heightPx
    )
    return gridX.reshape(-1), gridY.reshape(-1)


def fitBfovToVectors(vectors: NDArray[np.float64]) -> BFoV:
    """Fit the tightest unrotated BFoV around a set of unit directions."""
    return _fitBfovFromVectors(np.asarray(vectors, dtype=np.float64))


def _fitBfovFromVectors(vectors: NDArray[np.float64]) -> BFoV:
    if vectors.ndim != 2 or vectors.shape[1] != 3 or not np.isfinite(vectors).all():
        raise GeometryError("boundary samples must be a finite array with final dimension 3")
    meanVector = np.mean(vectors, axis=0)
    meanNorm = float(np.linalg.norm(meanVector))
    if meanNorm == 0.0:
        meanVector = vectors[0]
        meanNorm = float(np.linalg.norm(meanVector))
    if meanNorm == 0.0:
        raise GeometryError("boundary samples must contain at least one non-zero vector")
    center = makeSphericalPoint(*unitVectorToYawPitch(tuple(meanVector)))
    horizontalAngles = np.empty(0, dtype=np.float64)
    verticalAngles = np.empty(0, dtype=np.float64)
    for _ in range(12):
        forward, right, up = cameraBasis(
            BFoV(center=center, horizontalFovRad=1.0, verticalFovRad=1.0)
        )
        forwardDots = vectors @ forward
        horizontalAngles = np.arctan2(vectors @ right, forwardDots)
        verticalAngles = np.arctan2(vectors @ up, forwardDots)
        horizontalOffset, _ = _minimalCircularAngleInterval(horizontalAngles)
        verticalOffset, _ = _linearAngleInterval(verticalAngles)
        if max(abs(horizontalOffset), abs(verticalOffset)) < 1e-10:
            break
        horizontalForward = (
            cos(horizontalOffset) * forward + sin(horizontalOffset) * right
        )
        shifted = cos(verticalOffset) * horizontalForward + sin(verticalOffset) * up
        shifted /= np.linalg.norm(shifted)
        center = makeSphericalPoint(*unitVectorToYawPitch(tuple(shifted)))

    forward, right, up = cameraBasis(
        BFoV(center=center, horizontalFovRad=1.0, verticalFovRad=1.0)
    )
    forwardDots = vectors @ forward
    horizontalAngles = np.arctan2(vectors @ right, forwardDots)
    verticalAngles = np.arctan2(vectors @ up, forwardDots)
    horizontalOffset, horizontalFovRad = _minimalCircularAngleInterval(horizontalAngles)
    verticalOffset, verticalFovRad = _linearAngleInterval(verticalAngles)
    if max(abs(horizontalOffset), abs(verticalOffset)) >= 1e-8:
        horizontalForward = (
            cos(horizontalOffset) * forward + sin(horizontalOffset) * right
        )
        shifted = cos(verticalOffset) * horizontalForward + sin(verticalOffset) * up
        shifted /= np.linalg.norm(shifted)
        center = makeSphericalPoint(*unitVectorToYawPitch(tuple(shifted)))
        forward, right, up = cameraBasis(
            BFoV(center=center, horizontalFovRad=1.0, verticalFovRad=1.0)
        )
        forwardDots = vectors @ forward
        horizontalAngles = np.arctan2(vectors @ right, forwardDots)
        verticalAngles = np.arctan2(vectors @ up, forwardDots)
        _, horizontalFovRad = _minimalCircularAngleInterval(horizontalAngles)
        _, verticalFovRad = _linearAngleInterval(verticalAngles)
    # A perspective boundary can straddle the camera's rear hemisphere.  Its minimal
    # circular angular interval may then approach 2*pi (for example 6.1356 rad),
    # which cannot be represented by the production BFoV contract.  This is not bad
    # input: it is a degenerate candidate envelope.  Saturate it to the widest valid
    # perspective BFoV instead of aborting the entire sequence.
    horizontalFovRad = _safePerspectiveFov(horizontalFovRad)
    verticalFovRad = _safePerspectiveFov(verticalFovRad)
    return BFoV(
        center=center,
        horizontalFovRad=horizontalFovRad,
        verticalFovRad=verticalFovRad,
        rollRad=0.0,
    )


def _minimalCircularAngleInterval(
    angles: NDArray[np.float64],
) -> tuple[float, float]:
    values = np.asarray(angles, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise GeometryError("angles must be a non-empty finite array")
    normalized = np.sort(np.mod(values, 2.0 * pi))
    if normalized.size == 1:
        return _wrapAngle(float(normalized[0])), 0.0
    gaps = np.diff(normalized)
    wrapGap = normalized[0] + 2.0 * pi - normalized[-1]
    allGaps = np.concatenate((gaps, np.asarray((wrapGap,), dtype=np.float64)))
    largestGapIndex = int(np.argmax(allGaps))
    start = float(normalized[(largestGapIndex + 1) % normalized.size])
    span = max(0.0, float(2.0 * pi - allGaps[largestGapIndex]))
    midpoint = _wrapAngle(start + span / 2.0)
    return midpoint, span


def _safePerspectiveFov(value: float) -> float:
    if not np.isfinite(value):
        raise GeometryError(f"BFoV span must be finite: {value}")
    minimum = float(np.finfo(np.float64).eps)
    maximum = float(np.nextafter(np.pi, 0.0))
    return min(maximum, max(minimum, float(value)))


def _linearAngleInterval(angles: NDArray[np.float64]) -> tuple[float, float]:
    values = np.asarray(angles, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise GeometryError("angles must be a non-empty finite array")
    lower = float(np.min(values))
    upper = float(np.max(values))
    return (lower + upper) / 2.0, max(0.0, upper - lower)


def _wrapAngle(angle: float) -> float:
    return float((angle + pi) % (2.0 * pi) - pi)


def _sampleErpBoxBoundary(
    xPx: float,
    yPx: float,
    widthPx: float,
    heightPx: float,
    samplesPerEdge: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    _requireBoundarySamplesPerEdge(samplesPerEdge)
    edge = np.linspace(0.0, 1.0, samplesPerEdge, dtype=np.float64)
    topX = xPx + edge * widthPx
    rightX = np.full_like(edge, xPx + widthPx)
    bottomX = xPx + (1.0 - edge) * widthPx
    leftX = np.full_like(edge, xPx)
    topY = np.full_like(edge, yPx)
    rightY = yPx + edge * heightPx
    bottomY = np.full_like(edge, yPx + heightPx)
    leftY = yPx + (1.0 - edge) * heightPx
    sampleX = np.concatenate((topX, rightX, bottomX, leftX))
    sampleY = np.concatenate((topY, rightY, bottomY, leftY))
    return sampleX, sampleY


def _sampleCanonicalRectBoundary(
    samplesPerEdge: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    edge = np.linspace(0.0, 1.0, samplesPerEdge, dtype=np.float64)
    topX = edge
    rightX = np.ones_like(edge)
    bottomX = 1.0 - edge
    leftX = np.zeros_like(edge)
    topY = np.zeros_like(edge)
    rightY = edge
    bottomY = np.ones_like(edge)
    leftY = 1.0 - edge
    return (
        np.concatenate((topX, rightX, bottomX, leftX)),
        np.concatenate((topY, rightY, bottomY, leftY)),
    )


def _erpSamplesToVectors(
    sampleX: NDArray[np.float64],
    sampleY: NDArray[np.float64],
    frameWidthPx: int,
    frameHeightPx: int,
) -> NDArray[np.float64]:
    if sampleX.shape != sampleY.shape:
        raise GeometryError(
            f"sample coordinate shapes must match: {sampleX.shape} != {sampleY.shape}"
        )
    points = [
        erpPixelToSphericalPoint(
            float(wrapPixelX(float(xPx), frameWidthPx)),
            float(yPx),
            frameWidthPx,
            frameHeightPx,
        )
        for xPx, yPx in zip(sampleX, sampleY, strict=True)
    ]
    return np.asarray([(point.x, point.y, point.z) for point in points], dtype=np.float64)


def _requireBoundarySamplesPerEdge(samples: int) -> None:
    if isinstance(samples, bool) or samples < 2:
        raise GeometryError(
            f"boundarySamplesPerEdge must be an integer >= 2, actual={samples}"
        )


def _requireFrameDimensions(frameWidthPx: int, frameHeightPx: int) -> None:
    if isinstance(frameWidthPx, bool) or isinstance(frameHeightPx, bool):
        raise GeometryError("frame dimensions must be integers")
    if frameWidthPx <= 0 or frameHeightPx <= 0:
        raise GeometryError(
            f"frame dimensions must be positive, actual=({frameWidthPx}, {frameHeightPx})"
        )


def _requireErpBox(bbox: BBoxXYWH, frameWidthPx: int, frameHeightPx: int) -> None:
    if bbox.widthPx <= 0.0 or bbox.heightPx <= 0.0:
        raise GeometryError("bbox dimensions must be positive")
    if bbox.widthPx > frameWidthPx or bbox.heightPx > frameHeightPx:
        raise GeometryError("bbox dimensions cannot exceed the frame size")
    if not np.isfinite([bbox.xPx, bbox.yPx, bbox.widthPx, bbox.heightPx]).all():
        raise GeometryError("bbox coordinates must be finite")
    if bbox.yPx < 0.0 or bbox.yPx + bbox.heightPx > frameHeightPx:
        raise GeometryError("bbox must stay within the vertical frame range")


def _requireLocalBox(localBox: BBoxXYWH, viewWidthPx: int, viewHeightPx: int) -> None:
    if localBox.widthPx <= 0.0 or localBox.heightPx <= 0.0:
        raise GeometryError("local box dimensions must be positive")
    if localBox.widthPx > viewWidthPx or localBox.heightPx > viewHeightPx:
        raise GeometryError("local box dimensions cannot exceed the view size")
    if not np.isfinite([localBox.xPx, localBox.yPx, localBox.widthPx, localBox.heightPx]).all():
        raise GeometryError("local box coordinates must be finite")
    if localBox.xPx < 0.0 or localBox.yPx < 0.0:
        raise GeometryError("local box must start within the view bounds")
    if (
        localBox.xPx + localBox.widthPx > viewWidthPx
        or localBox.yPx + localBox.heightPx > viewHeightPx
    ):
        raise GeometryError("local box must stay within the local view bounds")


def _requireViewSpec(spec: ViewSpec) -> None:
    if spec.outputWidthPx <= 0 or spec.outputHeightPx <= 0:
        raise GeometryError("view dimensions must be positive")


__all__ = ["SphericalGeometryImpl", "fitBfovToVectors"]
