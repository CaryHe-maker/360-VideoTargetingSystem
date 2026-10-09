"""Plan the template view and the single perspective search view of a frame."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import asin, atan, atan2, ceil, cos, floor, pi, sin, sqrt, tan

from track360.core.config import BackendTuningConfig, GeometryConfig, TrackingConfig
from track360.core.types import BBoxXYWH, BFoV, SphericalPoint, ViewProjection, ViewSpec
from track360.geometry.projection_math import (
    cameraBasis,
    fovToFocalLengthPx,
    makeSphericalPoint,
    unitVectorToYawPitch,
    viewAxes,
)

# The search view is this many times the mean target size: the search factor the
# tracker was trained with, which is also the crop the backend takes.
ALIGNED_SEARCH_FACTOR = 4.0
# Targets larger than this are sized as if they were this large when the backend's
# search crop is laid out.  A perspective view cannot hold such targets anyway.
PRIOR_MAX_SIZE_RAD = 150.0 * pi / 180.0
# A spherical view is at least this many times the target's extent on each axis, so
# an elongated target is not cut by a view sized from its mean extent.
SPHERICAL_TARGET_MARGIN = 1.25
# Spans of a spherical view stay just short of the whole sphere.
_MAX_HORIZONTAL_SPAN_RAD = 2.0 * pi - 1e-6
_MAX_VERTICAL_SPAN_RAD = pi - 1e-6
# Directions closer than this to the image plane's horizon project to its far edge.
_MIN_DEPTH = 0.05
SEARCH_VIEW_ID = 0
# Scan views get the ids after the search view's.
SCAN_VIEW_ID_BASE = 1
# Neighbouring scan views are this fraction of a view apart, so they overlap by half.
SCAN_STEP_RATIO = 0.5
# Views of the normal size that follow an enlarged scan view get the ids from here.
REFINE_VIEW_ID_BASE = 64
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
        self._scanOffsets: dict[int, list[tuple[float, float, float]]] = {}

    def scanViews(
        self, lastSeen: BFoV, cursor: int, count: int
    ) -> tuple[tuple[ViewSpec, ...], int]:
        """Views to look for a lost target in, and the cursor for the next frame.

        The sphere is covered with views the size of the target's normal search
        view, half a view apart, ordered by distance from where the target was
        last seen.  Each call hands out the next ``count`` of them; the cursor
        wraps around once the whole sphere has been visited.
        """
        if count <= 0:
            return (), cursor
        horizontal = min(lastSeen.horizontalFovRad, _MAX_HORIZONTAL_SPAN_RAD)
        vertical = min(lastSeen.verticalFovRad, _MAX_VERTICAL_SPAN_RAD)
        span = ALIGNED_SEARCH_FACTOR * sqrt(horizontal * vertical)
        span = min(pi, max(self._tuning.alignedMinFovRad, span))
        offsets = self._offsetsFor(SCAN_STEP_RATIO * span)
        forward, right, up = viewAxes(lastSeen.center)
        views = []
        for index in range(min(count, len(offsets))):
            along, across, upward = offsets[(cursor + index) % len(offsets)]
            direction = along * forward + across * right + upward * up
            center = makeSphericalPoint(*unitVectorToYawPitch(tuple(direction)))
            views.append(
                replace(
                    self._searchView(center, horizontal, vertical),
                    viewId=SCAN_VIEW_ID_BASE + index,
                )
            )
        return tuple(views), (cursor + len(views)) % len(offsets)

    def probeView(
        self,
        center: SphericalPoint,
        horizontalSizeRad: float,
        verticalSizeRad: float,
        scale: float,
        viewId: int,
    ) -> ViewSpec:
        """A view ``scale`` times the normal search view of such a target, around ``center``.

        Laid out as the search view of a target ``scale`` times as large, so the
        backend's search crop is the whole view and the target appears that much
        smaller in it.
        """
        view = self._searchView(center, scale * horizontalSizeRad, scale * verticalSizeRad)
        return replace(view, viewId=viewId)

    def _offsetsFor(self, stepRad: float) -> list[tuple[float, float, float]]:
        """Directions covering the sphere ``stepRad`` apart, nearest to straight ahead first."""
        key = max(1, round(stepRad * 1000.0))
        if key not in self._scanOffsets:
            step = key / 1000.0
            rows = max(1, ceil(pi / step))
            offsets = []
            for row in range(rows):
                latitude = -pi / 2.0 + (row + 0.5) * pi / rows
                columns = max(1, ceil(2.0 * pi * cos(latitude) / step))
                for column in range(columns):
                    longitude = -pi + (column + 0.5) * 2.0 * pi / columns
                    offsets.append(
                        (
                            cos(latitude) * cos(longitude),
                            cos(latitude) * sin(longitude),
                            sin(latitude),
                        )
                    )
            offsets.sort(key=lambda offset: -offset[0])
            self._scanOffsets[key] = offsets
        return self._scanOffsets[key]

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
        if self._usesSphericalView(horizontalSizeRad, verticalSizeRad):
            return self._sphericalView(
                center,
                horizontalSizeRad,
                verticalSizeRad,
                ALIGNED_SEARCH_FACTOR,
                self._geometry.viewWidthPx,
            )
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

    def templateView(self, target: BFoV) -> ViewSpec:
        """The view the frame-0 template is cropped from; its prior box is the target."""
        if self._usesSphericalView(target.horizontalFovRad, target.verticalFovRad):
            return self._sphericalView(
                target.center,
                target.horizontalFovRad,
                target.verticalFovRad,
                self._tuning.templateFovScale,
                self._geometry.viewWidthPx,
            )
        bfov = self.templateBfov(target)
        widthPx, heightPx = self._geometry.viewWidthPx, self._geometry.viewHeightPx
        boxWidth = 2.0 * fovToFocalLengthPx(bfov.horizontalFovRad, widthPx) * tan(
            target.horizontalFovRad / 2.0
        )
        boxHeight = 2.0 * fovToFocalLengthPx(bfov.verticalFovRad, heightPx) * tan(
            target.verticalFovRad / 2.0
        )
        boxWidth = max(2.0, min(float(widthPx), boxWidth))
        boxHeight = max(2.0, min(float(heightPx), boxHeight))
        return ViewSpec(
            viewId=SEARCH_VIEW_ID,
            bfov=bfov,
            outputWidthPx=widthPx,
            outputHeightPx=heightPx,
            priorBox=BBoxXYWH(
                xPx=(widthPx - boxWidth) / 2.0,
                yPx=(heightPx - boxHeight) / 2.0,
                widthPx=boxWidth,
                heightPx=boxHeight,
            ),
        )

    def _usesSphericalView(self, horizontalSizeRad: float, verticalSizeRad: float) -> bool:
        """Whether the search region of this target is too wide for a tangent plane."""
        if not self._tuning.sphericalSearch:
            return False
        span = ALIGNED_SEARCH_FACTOR * sqrt(horizontalSizeRad * verticalSizeRad)
        return span >= self._tuning.sphericalSearchFovRad

    def _sphericalView(
        self,
        center: SphericalPoint,
        horizontalSizeRad: float,
        verticalSizeRad: float,
        factor: float,
        sidePx: int,
    ) -> ViewSpec:
        """A view linear in angle in which ``factor`` times the mean target size is ``sidePx``.

        Each axis spans ``factor`` times the mean size, widened for elongated
        targets and cut at the whole sphere, so the view is not always square; its
        pixels stay isotropic.  The backend's crop around the prior box pads what a
        cut view leaves out.
        """
        span = factor * sqrt(horizontalSizeRad * verticalSizeRad)
        pixelsPerRad = sidePx / span

        def axis(sizeRad: float, limit: float) -> tuple[int, float]:
            wanted = max(span, SPHERICAL_TARGET_MARGIN * sizeRad)
            if wanted == span and span <= limit:
                return sidePx, span
            pixels = max(2, floor(min(wanted, limit) * pixelsPerRad))
            return pixels, pixels / pixelsPerRad

        widthPx, horizontalSpan = axis(horizontalSizeRad, _MAX_HORIZONTAL_SPAN_RAD)
        heightPx, verticalSpan = axis(verticalSizeRad, _MAX_VERTICAL_SPAN_RAD)
        boxWidth = min(float(widthPx), horizontalSizeRad * pixelsPerRad)
        boxHeight = min(float(heightPx), verticalSizeRad * pixelsPerRad)
        return ViewSpec(
            viewId=SEARCH_VIEW_ID,
            bfov=BFoV(center, horizontalSpan, verticalSpan),
            outputWidthPx=widthPx,
            outputHeightPx=heightPx,
            priorBox=BBoxXYWH(
                xPx=(widthPx - boxWidth) / 2.0,
                yPx=(heightPx - boxHeight) / 2.0,
                widthPx=boxWidth,
                heightPx=boxHeight,
            ),
            projection=ViewProjection.SPHERICAL,
        )

    def templateBfov(self, target: BFoV) -> BFoV:
        """Return the view the frame-0 template is cropped from."""
        scale = self._tuning.templateFovScale
        fov = _squareFov(target.horizontalFovRad, target.verticalFovRad, scale)
        fov = min(self._geometry.maxFovRad, max(self._tuning.alignedMinFovRad, fov))
        return BFoV(center=target.center, horizontalFovRad=fov, verticalFovRad=fov)

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
    """Where a BFoV falls in a view, as a box in the view's pixels.

    The box is centered on the projection of the BFoV center and sized as the BFoV
    would be at the view center, which is accurate for the small offsets between
    consecutive frames.  In a perspective view a direction at or behind the image
    plane's horizon is placed far outside the view on the side it lies; callers
    clamp it.
    """
    if view.projection is ViewProjection.SPHERICAL:
        forward, right, up = viewAxes(view.bfov.center, view.bfov.rollRad)
        direction = (bfov.center.x, bfov.center.y, bfov.center.z)
        depth = float(sum(a * b for a, b in zip(direction, forward, strict=True)))
        across = float(sum(a * b for a, b in zip(direction, right, strict=True)))
        upward = float(sum(a * b for a, b in zip(direction, up, strict=True)))
        horizontalScale = view.outputWidthPx / view.bfov.horizontalFovRad
        verticalScale = view.outputHeightPx / view.bfov.verticalFovRad
        latitude = asin(max(-1.0, min(1.0, upward)))
        # Meridians converge away from the view's equator: the same width covers more
        # longitude there.
        widthPx = bfov.horizontalFovRad * horizontalScale / max(cos(latitude), _MIN_DEPTH)
        heightPx = bfov.verticalFovRad * verticalScale
        return BBoxXYWH(
            xPx=view.outputWidthPx / 2.0 + atan2(across, depth) * horizontalScale - widthPx / 2.0,
            yPx=view.outputHeightPx / 2.0 - latitude * verticalScale - heightPx / 2.0,
            widthPx=widthPx,
            heightPx=heightPx,
        )
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
    "ALIGNED_SEARCH_FACTOR",
    "REFINE_VIEW_ID_BASE",
    "SCAN_VIEW_ID_BASE",
    "SEARCH_VIEW_ID",
    "TRAJECTORY_LENGTH",
    "ViewPlanner",
    "clampFov",
    "localBoxOfBfov",
]
