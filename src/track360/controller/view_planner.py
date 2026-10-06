"""Plan the single perspective search view of a frame."""

from __future__ import annotations

from math import pi

from track360.core.config import BackendTuningConfig, GeometryConfig, TrackingConfig
from track360.core.types import BBoxXYWH, BFoV, SphericalPoint, ViewSpec

# The search view spans this many times the predicted target extent on each axis.
SEARCH_FOV_SCALE = 3.0
SEARCH_VIEW_ID = 0


class ViewPlanner:
    """Center one view on the predicted target direction and size it from the target."""

    def __init__(
        self,
        geometryConfig: GeometryConfig,
        trackingConfig: TrackingConfig,
        backendTuning: BackendTuningConfig | None = None,
    ) -> None:
        self._geometry = geometryConfig
        self._tracking = trackingConfig
        self._tuning = backendTuning or BackendTuningConfig()

    def searchView(
        self,
        center: SphericalPoint,
        horizontalSizeRad: float,
        verticalSizeRad: float,
    ) -> ViewSpec:
        """Return the search view for a target of the given predicted angular size."""
        horizontalFov = clampFov(SEARCH_FOV_SCALE * horizontalSizeRad, self._geometry)
        verticalFov = clampFov(SEARCH_FOV_SCALE * verticalSizeRad, self._geometry)
        if self._tuning.viewHorizontalFovCapRad is not None:
            horizontalFov = min(horizontalFov, self._tuning.viewHorizontalFovCapRad)
        if self._tuning.viewVerticalFovCapRad is not None:
            verticalFov = min(verticalFov, self._tuning.viewVerticalFovCapRad)
        return ViewSpec(
            viewId=SEARCH_VIEW_ID,
            bfov=BFoV(center=center, horizontalFovRad=horizontalFov, verticalFovRad=verticalFov),
            outputWidthPx=self._geometry.viewWidthPx,
            outputHeightPx=self._geometry.viewHeightPx,
        )

    def contextBfov(
        self,
        center: SphericalPoint,
        frameWidthPx: int,
        frameHeightPx: int,
        anchorBox: BBoxXYWH,
        currentBox: BBoxXYWH,
        uncertaintyRad: float = 0.0,
    ) -> BFoV:
        """Build the motion fallback envelope output when a frame has no candidate."""
        widthPx = self._tracking.contextScale * max(anchorBox.widthPx, currentBox.widthPx)
        heightPx = self._tracking.contextScale * max(anchorBox.heightPx, currentBox.heightPx)
        widthPx *= 1.0 + self._tracking.contextMarginRatio
        heightPx *= 1.0 + self._tracking.contextMarginRatio
        horizontalFov = 2.0 * pi * widthPx / frameWidthPx + 2.0 * uncertaintyRad
        verticalFov = pi * heightPx / frameHeightPx + 2.0 * uncertaintyRad
        return BFoV(
            center=center,
            horizontalFovRad=clampFov(horizontalFov, self._geometry),
            verticalFovRad=clampFov(verticalFov, self._geometry),
        )


def clampFov(value: float, geometry: GeometryConfig) -> float:
    return min(geometry.maxFovRad, max(geometry.minFovRad, value))


__all__ = ["SEARCH_FOV_SCALE", "SEARCH_VIEW_ID", "ViewPlanner", "clampFov"]
