"""Geometry package public surface."""

from track360.geometry.bfov_projector import BfovProjector
from track360.geometry.projection_math import (
    angleToPixelOffsetPx,
    cameraBasis,
    clampPitch,
    erpPixelToSphericalPoint,
    focalLengthPxToFov,
    fovToFocalLengthPx,
    localPixelsToUnitVectors,
    makeSphericalPoint,
    pixelOffsetToAngleRad,
    sphericalPointToErpPixel,
    unitVectorsToErpPixels,
    unitVectorToYawPitch,
    wrapYaw,
)
from track360.geometry.seam import (
    containsCircularX,
    minimalCircularInterval,
    splitSeamBox,
    wrapPixelX,
)
from track360.geometry.spherical_geometry import SphericalGeometryImpl

__all__ = [
    "BfovProjector",
    "SphericalGeometryImpl",
    "angleToPixelOffsetPx",
    "cameraBasis",
    "clampPitch",
    "containsCircularX",
    "erpPixelToSphericalPoint",
    "focalLengthPxToFov",
    "fovToFocalLengthPx",
    "localPixelsToUnitVectors",
    "makeSphericalPoint",
    "minimalCircularInterval",
    "pixelOffsetToAngleRad",
    "sphericalPointToErpPixel",
    "splitSeamBox",
    "unitVectorToYawPitch",
    "unitVectorsToErpPixels",
    "wrapPixelX",
    "wrapYaw",
]
