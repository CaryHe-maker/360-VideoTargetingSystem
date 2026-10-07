"""Plan the template view and the single perspective search view of a frame."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import atan, pi, sqrt, tan

from track360.core.config import BackendTuningConfig, GeometryConfig, TrackingConfig
from track360.core.types import BBoxXYWH, BFoV, SphericalPoint, ViewSpec
from track360.geometry.projection_math import cameraBasis

# The search view spans this many times the predicted target extent on each axis.
SEARCH_FOV_SCALE = 3.0
# With ``alignedSearch`` the view is this many times the mean target size: the search
# factor the tracker was trained with, which is also the crop the backend takes.
ALIGNED_SEARCH_FACTOR = 4.0
# Targets larger than this are sized as if they were this large when the backend's
# search crop is laid out.  A perspective view cannot hold such targets anyway.
PRIOR_MAX_SIZE_RAD = 150.0 * pi / 180.0
# Directions closer than this to the image plane's horizon project to its far edge.
_MIN_DEPTH = 0.05
SEARCH_VIEW_ID = 0
# Previous target boxes handed to the backend with every view (ARTrackV2 reads seven).
TRAJECTORY_LENGTH = 7


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
        history: Sequence[BFoV] = (),
    ) -> ViewSpec:
        """Return the search view for a target of the given predicted angular size.

        ``history`` holds the target's previous BFoVs, oldest first; they are laid
        out in the view's pixels as the trajectory of sequence-level backends.
        """
        view = self._searchView(center, horizontalSizeRad, verticalSizeRad)
        if not history:
            return view
        return replace(
            view,
            trajectory=tuple(
                localBoxOfBfov(view, bfov) for bfov in history[-TRAJECTORY_LENGTH:]
            ),
        )

    def _searchView(
        self,
        center: SphericalPoint,
        horizontalSizeRad: float,
        verticalSizeRad: float,
    ) -> ViewSpec:
        if self._tuning.alignedSearch:
            # A square view with isotropic pixels whose side is
            # ``ALIGNED_SEARCH_FACTOR`` times the geometric-mean target size, as in
            # the tracker's training crops.
            fov = _squareFov(horizontalSizeRad, verticalSizeRad, ALIGNED_SEARCH_FACTOR)
            fov = min(self._geometry.maxFovRad, max(self._tuning.alignedMinFovRad, fov))
            for cap in (self._tuning.viewHorizontalFovCapRad, self._tuning.viewVerticalFovCapRad):
                if cap is not None:
                    fov = min(fov, cap)
            # The backend crops its search region around this box.  With an unclamped
            # FOV that crop is the whole view; when the FOV limit cut the view short
            # (large targets) the crop extends past the view and is padded, so the
            # target still fills the usual share of the search region.
            # The tangent extent of a target grows without bound towards 180 degrees;
            # PRIOR_MAX_SIZE_RAD keeps the padded crop finite.
            scale = (1.0 - 1e-9) / tan(fov / 2.0)
            widthPx = self._geometry.viewWidthPx * scale * _halfTangent(
                min(horizontalSizeRad, PRIOR_MAX_SIZE_RAD)
            )
            heightPx = self._geometry.viewHeightPx * scale * _halfTangent(
                min(verticalSizeRad, PRIOR_MAX_SIZE_RAD)
            )
            return ViewSpec(
                viewId=SEARCH_VIEW_ID,
                bfov=BFoV(center=center, horizontalFovRad=fov, verticalFovRad=fov),
                outputWidthPx=self._geometry.viewWidthPx,
                outputHeightPx=self._geometry.viewHeightPx,
                priorBox=BBoxXYWH(
                    xPx=(self._geometry.viewWidthPx - widthPx) / 2.0,
                    yPx=(self._geometry.viewHeightPx - heightPx) / 2.0,
                    widthPx=widthPx,
                    heightPx=heightPx,
                ),
            )
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

    def templateBfov(self, target: BFoV) -> BFoV:
        """Return the view the frame-0 template is cropped from."""
        scale = self._tuning.templateFovScale
        if self._tuning.alignedSearch:
            fov = _squareFov(target.horizontalFovRad, target.verticalFovRad, scale)
            fov = min(self._geometry.maxFovRad, max(self._tuning.alignedMinFovRad, fov))
            return BFoV(center=target.center, horizontalFovRad=fov, verticalFovRad=fov)
        return BFoV(
            center=target.center,
            horizontalFovRad=clampFov(scale * target.horizontalFovRad, self._geometry),
            verticalFovRad=clampFov(scale * target.verticalFovRad, self._geometry),
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


def localBoxOfBfov(view: ViewSpec, bfov: BFoV) -> BBoxXYWH:
    """Where a BFoV falls in a perspective view, as a box in the view's pixels.

    The box is centered on the projection of the BFoV center and sized as the BFoV
    would be at the view center, which is accurate for the small offsets between
    consecutive frames.  A direction at or behind the image plane's horizon is placed
    far outside the view on the side it lies; callers clamp it.
    """
    forward, right, up = cameraBasis(view.bfov)
    direction = (bfov.center.x, bfov.center.y, bfov.center.z)
    depth = float(sum(a * b for a, b in zip(direction, forward, strict=True)))
    across = float(sum(a * b for a, b in zip(direction, right, strict=True)))
    upward = float(sum(a * b for a, b in zip(direction, up, strict=True)))
    if depth < _MIN_DEPTH:
        # Keep only the side the direction lies on; straight behind counts as to the right.
        lateral = sqrt(across * across + upward * upward)
        across, upward = (across / lateral, upward / lateral) if lateral > 1e-9 else (1.0, 0.0)
        depth = _MIN_DEPTH
    halfViewX = tan(view.bfov.horizontalFovRad / 2.0)
    halfViewY = tan(view.bfov.verticalFovRad / 2.0)
    centerX = view.outputWidthPx / 2.0 * (1.0 + across / depth / halfViewX)
    centerY = view.outputHeightPx / 2.0 * (1.0 - upward / depth / halfViewY)
    widthPx = view.outputWidthPx * _halfTangent(
        min(bfov.horizontalFovRad, PRIOR_MAX_SIZE_RAD)
    ) / halfViewX
    heightPx = view.outputHeightPx * _halfTangent(
        min(bfov.verticalFovRad, PRIOR_MAX_SIZE_RAD)
    ) / halfViewY
    return BBoxXYWH(
        xPx=centerX - widthPx / 2.0,
        yPx=centerY - heightPx / 2.0,
        widthPx=widthPx,
        heightPx=heightPx,
    )


def _squareFov(horizontalSizeRad: float, verticalSizeRad: float, factor: float) -> float:
    """FOV of a square view ``factor`` times the target's mean size on the image plane."""
    halfWidth = _halfTangent(horizontalSizeRad)
    halfHeight = _halfTangent(verticalSizeRad)
    return 2.0 * atan(factor * sqrt(halfWidth * halfHeight))


def _halfTangent(sizeRad: float) -> float:
    """Half extent on the image plane of an angular size, finite up to 180 degrees."""
    return tan(min(sizeRad, pi - 1e-6) / 2.0)


def clampFov(value: float, geometry: GeometryConfig) -> float:
    return min(geometry.maxFovRad, max(geometry.minFovRad, value))


__all__ = [
    "SEARCH_FOV_SCALE",
    "SEARCH_VIEW_ID",
    "TRAJECTORY_LENGTH",
    "ViewPlanner",
    "clampFov",
    "localBoxOfBfov",
]
