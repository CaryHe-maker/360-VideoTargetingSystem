"""Pure spherical and perspective projection mathematics.

The camera coordinate system uses +x to the right, +y upward, and +z
forward. ERP coordinates are continuous pixel-edge coordinates: x=0 is yaw
-pi, x=width is yaw +pi, y=0 is pitch +pi/2, and y=height is pitch -pi/2.
"""

from __future__ import annotations

from math import asin, atan, atan2, cos, isfinite, pi, sin, sqrt, tan

import numpy as np
from numpy.typing import NDArray

from track360.core.errors import GeometryError
from track360.core.types import BFoV, SphericalPoint


#to solve the condition when the target crossing the bound in Yaw
def wrapYaw(yawRad: float) -> float:
    """Normalize yaw to the half-open interval [-pi, pi)."""
    _requireFinite("yawRad", yawRad)
    wrapped = (yawRad + pi) % (2.0 * pi) - pi
    if wrapped >= pi:
        wrapped -= 2.0 * pi
    return wrapped


#to solve the condition when the target crossing the bound in Pitch
def clampPitch(pitchRad: float) -> float:
    """Clamp a finite pitch to the closed interval [-pi/2, pi/2]."""
    _requireFinite("pitchRad", pitchRad)
    return min(max(pitchRad, -pi / 2.0), pi / 2.0)


def yawPitchToUnitVector(yawRad: float, pitchRad: float) -> tuple[float, float, float]:
    """Convert yaw and pitch in radians to a right-handed unit direction."""
    _requireFinite("yaw/pitch", yawRad, pitchRad)
    if not -pi / 2.0 <= pitchRad <= pi / 2.0:
        raise GeometryError(f"pitchRad must be in [-pi/2, pi/2], actual={pitchRad}")
    cosPitch = cos(pitchRad)
    return (
        cosPitch * sin(yawRad),
        sin(pitchRad),
        cosPitch * cos(yawRad),
    )


def unitVectorToYawPitch(vector: tuple[float, float, float]) -> tuple[float, float]:
    """Convert a non-zero direction to normalized yaw and pitch radians."""
    if len(vector) != 3:
        raise GeometryError("unit vector must contain exactly three components")
    x, y, z = (float(component) for component in vector)
    _requireFinite("unit vector", x, y, z)
    norm = sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        raise GeometryError("unit vector must be non-zero")
    x /= norm
    y /= norm
    z /= norm
    return wrapYaw(atan2(x, z)), asin(min(max(y, -1.0), 1.0))


def makeSphericalPoint(yawRad: float, pitchRad: float) -> SphericalPoint:
    """Build a protocol-valid spherical point from yaw and pitch."""
    normalizedYawRad = wrapYaw(yawRad)
    normalizedPitchRad = clampPitch(pitchRad)
    x, y, z = yawPitchToUnitVector(normalizedYawRad, normalizedPitchRad)
    return SphericalPoint(
        x=x,
        y=y,
        z=z,
        yawRad=normalizedYawRad,
        pitchRad=normalizedPitchRad,
    )


def erpPixelToSphericalPoint(
    xPx: float,
    yPx: float,
    frameWidthPx: int,
    frameHeightPx: int,
) -> SphericalPoint:
    """Map a continuous ERP pixel-edge coordinate to a spherical point."""
    _requireFrameDimensions(frameWidthPx, frameHeightPx)
    _requireFinite("ERP coordinate", xPx, yPx)
    if not 0.0 <= xPx <= frameWidthPx or not 0.0 <= yPx <= frameHeightPx:
        raise GeometryError(
            "ERP coordinate outside frame: "
            f"coordinate=({xPx}, {yPx}), frame=({frameWidthPx}, {frameHeightPx})"
        )
    yawRad = 2.0 * pi * (xPx % frameWidthPx) / frameWidthPx - pi
    pitchRad = pi / 2.0 - pi * yPx / frameHeightPx
    return makeSphericalPoint(yawRad, pitchRad)


def sphericalPointToErpPixel(
    point: SphericalPoint,
    frameWidthPx: int,
    frameHeightPx: int,
) -> tuple[float, float]:
    """Map a spherical point to a continuous ERP pixel-edge coordinate."""
    _requireFrameDimensions(frameWidthPx, frameHeightPx)
    xPx = (point.yawRad + pi) * frameWidthPx / (2.0 * pi)
    yPx = (pi / 2.0 - point.pitchRad) * frameHeightPx / pi
    return xPx % frameWidthPx, min(max(yPx, 0.0), float(frameHeightPx))


def fovToFocalLengthPx(fovRad: float, imageExtentPx: float) -> float:
    """Convert a perspective field of view to focal length in pixels."""
    _requirePerspectiveFov("fovRad", fovRad)
    _requirePositiveFinite("imageExtentPx", imageExtentPx)
    return imageExtentPx / (2.0 * tan(fovRad / 2.0))


def focalLengthPxToFov(focalLengthPx: float, imageExtentPx: float) -> float:
    """Convert focal length and image extent to a perspective field of view."""
    _requirePositiveFinite("focalLengthPx", focalLengthPx)
    _requirePositiveFinite("imageExtentPx", imageExtentPx)
    return 2.0 * atan(imageExtentPx / (2.0 * focalLengthPx))


def pixelOffsetToAngleRad(offsetPx: float, imageExtentPx: float, fovRad: float) -> float:
    """Convert a signed image-plane offset from the optical axis to an angle."""
    _requireFinite("offsetPx", offsetPx)
    focalLengthPx = fovToFocalLengthPx(fovRad, imageExtentPx)
    return atan(offsetPx / focalLengthPx)


def angleToPixelOffsetPx(angleRad: float, imageExtentPx: float, fovRad: float) -> float:
    """Convert an optical-axis angle to a signed image-plane pixel offset."""
    _requireFinite("angleRad", angleRad)
    if not -pi / 2.0 < angleRad < pi / 2.0:
        raise GeometryError(f"angleRad must be in (-pi/2, pi/2), actual={angleRad}")
    return fovToFocalLengthPx(fovRad, imageExtentPx) * tan(angleRad)


def cameraBasis(bfov: BFoV) -> tuple[NDArray[np.float64], ...]:
    """Return forward, right, and up unit axes for a BFoV camera."""
    _requirePerspectiveFov("horizontalFovRad", bfov.horizontalFovRad)
    _requirePerspectiveFov("verticalFovRad", bfov.verticalFovRad)
    yawRad = bfov.center.yawRad
    pitchRad = bfov.center.pitchRad
    forward = np.asarray((bfov.center.x, bfov.center.y, bfov.center.z), dtype=np.float64)
    baseRight = np.asarray((cos(yawRad), 0.0, -sin(yawRad)), dtype=np.float64)
    baseUp = np.asarray(
        (-sin(pitchRad) * sin(yawRad), cos(pitchRad), -sin(pitchRad) * cos(yawRad)),
        dtype=np.float64,
    )
    cosRoll = cos(bfov.rollRad)
    sinRoll = sin(bfov.rollRad)
    right = cosRoll * baseRight + sinRoll * baseUp
    up = -sinRoll * baseRight + cosRoll * baseUp
    return forward, right, up


def localPixelsToUnitVectors(
    xPx: NDArray[np.float64],
    yPx: NDArray[np.float64],
    bfov: BFoV,
    viewWidthPx: int,
    viewHeightPx: int,
) -> NDArray[np.float64]:
    """Project local continuous pixel coordinates to unit directions."""
    _requireFrameDimensions(viewWidthPx, viewHeightPx)
    if xPx.shape != yPx.shape:
        raise GeometryError(f"local coordinate shapes must match: {xPx.shape} != {yPx.shape}")
    if not np.isfinite(xPx).all() or not np.isfinite(yPx).all():
        raise GeometryError("local coordinates must contain only finite values")
    forward, right, up = cameraBasis(bfov)
    horizontalScale = tan(bfov.horizontalFovRad / 2.0)
    verticalScale = tan(bfov.verticalFovRad / 2.0)
    horizontal = (2.0 * xPx / viewWidthPx - 1.0) * horizontalScale
    vertical = (1.0 - 2.0 * yPx / viewHeightPx) * verticalScale
    vectors = (
        forward
        + horizontal[..., np.newaxis] * right
        + vertical[..., np.newaxis] * up
    )
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / norms


def unitVectorsToErpPixels(
    vectors: NDArray[np.float64],
    frameWidthPx: int,
    frameHeightPx: int,
    *,
    pixelCenters: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project unit directions to ERP edge coordinates or sample-center indices."""
    _requireFrameDimensions(frameWidthPx, frameHeightPx)
    if vectors.shape[-1:] != (3,) or not np.isfinite(vectors).all():
        raise GeometryError("vectors must be a finite array with final dimension 3")
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms == 0.0):
        raise GeometryError("vectors must be non-zero")
    normalized = vectors / norms
    yawRad = np.arctan2(normalized[..., 0], normalized[..., 2])
    pitchRad = np.arcsin(np.clip(normalized[..., 1], -1.0, 1.0))
    xPx = np.mod((yawRad + pi) * frameWidthPx / (2.0 * pi), frameWidthPx)
    yPx = np.clip((pi / 2.0 - pitchRad) * frameHeightPx / pi, 0.0, frameHeightPx)
    if pixelCenters:
        xPx = np.mod(xPx - 0.5, frameWidthPx)
        yPx = np.clip(yPx - 0.5, 0.0, frameHeightPx - 1.0)
    return xPx, yPx


def _requirePerspectiveFov(name: str, fovRad: float) -> None:
    _requireFinite(name, fovRad)
    if not 0.0 < fovRad < pi:
        raise GeometryError(f"{name} must be in (0, pi) for perspective projection")


def _requireFrameDimensions(frameWidthPx: int, frameHeightPx: int) -> None:
    if frameWidthPx <= 0 or frameHeightPx <= 0:
        raise GeometryError(
            f"frame dimensions must be positive, actual=({frameWidthPx}, {frameHeightPx})"
        )


def _requirePositiveFinite(name: str, value: float) -> None:
    _requireFinite(name, value)
    if value <= 0.0:
        raise GeometryError(f"{name} must be positive, actual={value}")


def _requireFinite(name: str, *values: float) -> None:
    if not all(isfinite(value) for value in values):
        raise GeometryError(f"{name} must contain only finite values")
