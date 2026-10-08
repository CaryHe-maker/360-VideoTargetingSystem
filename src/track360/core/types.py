"""Immutable data contracts shared by all project modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from math import isfinite, sqrt
from typing import NewType

import numpy as np
from numpy.typing import NDArray

from track360.core.errors import ProtocolError

FrameIndex = NewType("FrameIndex", int)
SequenceId = NewType("SequenceId", str)

UNIT_VECTOR_TOLERANCE = 1e-6


def _requireFinite(name: str, *values: float) -> None:
    if not all(isfinite(value) for value in values):
        raise ProtocolError(f"{name} must contain only finite values")


def _requireProbability(name: str, value: float) -> None:
    _requireFinite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ProtocolError(f"{name} must be in [0, 1], actual={value}")


def _requireCovariance2x2(
    name: str,
    value: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    if len(value) != 2 or any(len(row) != 2 for row in value):
        raise ProtocolError(f"{name} must be a 2x2 matrix")
    a, b = value[0]
    c, d = value[1]
    _requireFinite(name, a, b, c, d)
    if abs(b - c) > 1e-9 or a < 0.0 or d < 0.0 or a * d - b * c < -1e-12:
        raise ProtocolError(f"{name} must be symmetric positive semidefinite")


@dataclass(frozen=True, slots=True)
class BBoxXYWH:
    """An ERP or local-view bounding box in continuous pixel coordinates."""

    xPx: float
    yPx: float
    widthPx: float
    heightPx: float

    def __post_init__(self) -> None:
        _requireFinite("bbox", self.xPx, self.yPx, self.widthPx, self.heightPx)
        if self.widthPx <= 0.0 or self.heightPx <= 0.0:
            raise ProtocolError(
                f"bbox dimensions must be positive, actual=({self.widthPx}, {self.heightPx})"
            )


@dataclass(frozen=True, slots=True)
class SphericalPoint:
    """A unit direction and its yaw/pitch representation in radians."""

    x: float
    y: float
    z: float
    yawRad: float
    pitchRad: float

    def __post_init__(self) -> None:
        _requireFinite("spherical point", self.x, self.y, self.z, self.yawRad, self.pitchRad)
        norm = sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        if abs(norm - 1.0) > UNIT_VECTOR_TOLERANCE:
            raise ProtocolError(f"spherical point must be a unit vector, norm={norm}")
        if not -np.pi <= self.yawRad < np.pi:
            raise ProtocolError(f"yawRad must be in [-pi, pi), actual={self.yawRad}")
        if not -np.pi / 2.0 <= self.pitchRad <= np.pi / 2.0:
            raise ProtocolError(f"pitchRad must be in [-pi/2, pi/2], actual={self.pitchRad}")


@dataclass(frozen=True, slots=True)
class BFoV:
    """A bounded field of view on the unit sphere."""

    center: SphericalPoint
    horizontalFovRad: float
    verticalFovRad: float
    rollRad: float = 0.0

    def __post_init__(self) -> None:
        _requireFinite("BFoV", self.horizontalFovRad, self.verticalFovRad, self.rollRad)
        if not 0.0 < self.horizontalFovRad < 2.0 * np.pi:
            raise ProtocolError("horizontalFovRad must be in (0, 2*pi)")
        if not 0.0 < self.verticalFovRad < np.pi:
            raise ProtocolError("verticalFovRad must be in (0, pi)")


@dataclass(frozen=True, slots=True)
class SegmentationPlane:
    """Optional semantic and instance maps aligned with an ERP frame."""

    semantic: NDArray[np.int32] | None
    instance: NDArray[np.int32] | None
    classNames: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shapes: set[tuple[int, ...]] = set()
        for name, plane in (("semantic", self.semantic), ("instance", self.instance)):
            if plane is None:
                continue
            if not isinstance(plane, np.ndarray) or plane.dtype != np.int32:
                raise ProtocolError(f"{name} segmentation must be an int32 NumPy array")
            if plane.ndim != 2:
                raise ProtocolError(f"{name} segmentation must have shape [H, W]")
            shapes.add(plane.shape)
        if len(shapes) > 1:
            raise ProtocolError("semantic and instance segmentation shapes must match")
        if any(
            not isinstance(key, int) or not isinstance(value, str)
            for key, value in self.classNames.items()
        ):
            raise ProtocolError("classNames must map integer IDs to strings")


@dataclass(frozen=True, slots=True)
class FramePacket:
    """One RGB ERP frame with optional aligned segmentation."""

    sequenceId: SequenceId
    frameIndex: FrameIndex
    timestampNs: int
    rgb: NDArray[np.uint8]
    segmentation: SegmentationPlane | None = None

    def __post_init__(self) -> None:
        if not str(self.sequenceId):
            raise ProtocolError("sequenceId must be non-empty")
        if int(self.frameIndex) < 0:
            raise ProtocolError(f"frameIndex must be non-negative, actual={self.frameIndex}")
        if self.timestampNs < 0:
            raise ProtocolError(f"timestampNs must be non-negative, actual={self.timestampNs}")
        if not isinstance(self.rgb, np.ndarray) or self.rgb.dtype != np.uint8:
            raise ProtocolError("rgb must be a uint8 NumPy array")
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ProtocolError(f"rgb must have shape [H, W, 3], actual={self.rgb.shape}")
        if self.rgb.shape[0] == 0 or self.rgb.shape[1] == 0:
            raise ProtocolError("rgb frame dimensions must be positive")
        frameShape = self.rgb.shape[:2]
        if self.segmentation is not None:
            for plane in (self.segmentation.semantic, self.segmentation.instance):
                if plane is not None and plane.shape != frameShape:
                    raise ProtocolError(
                        f"segmentation must align with rgb, actual={plane.shape} and {frameShape}"
                    )


@dataclass(frozen=True, slots=True)
class MotionState3D:
    """Predicted spherical direction, extent, uncertainty, and confidence."""

    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    confidence: float
    horizontalSizeRad: float = 0.0
    verticalSizeRad: float = 0.0
    angularUncertaintyRad: float = 0.0
    scaleUncertainty: float = 0.0
    reliability: float = 0.0
    centerCovarianceRad2: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0025, 0.0),
        (0.0, 0.0025),
    )
    scaleCovarianceLog2: tuple[tuple[float, float], tuple[float, float]] = (
        (0.04, 0.0),
        (0.0, 0.04),
    )

    def __post_init__(self) -> None:
        if len(self.position) != 3 or len(self.velocity) != 3:
            raise ProtocolError("motion position and velocity must contain three components")
        _requireFinite(
            "motion state",
            *self.position,
            *self.velocity,
            self.confidence,
            self.horizontalSizeRad,
            self.verticalSizeRad,
            self.angularUncertaintyRad,
            self.scaleUncertainty,
            self.reliability,
        )
        _requireCovariance2x2("centerCovarianceRad2", self.centerCovarianceRad2)
        _requireCovariance2x2("scaleCovarianceLog2", self.scaleCovarianceLog2)
        if min(
            self.horizontalSizeRad,
            self.verticalSizeRad,
            self.angularUncertaintyRad,
            self.scaleUncertainty,
            self.centerCovarianceRad2[0][0],
            self.centerCovarianceRad2[1][1],
            self.scaleCovarianceLog2[0][0],
            self.scaleCovarianceLog2[1][1],
        ) < 0.0:
            raise ProtocolError("motion sizes and uncertainties must be non-negative")
        _requireProbability("motion confidence", self.confidence)
        _requireProbability("motion reliability", self.reliability)


class TrackStatus(Enum):
    TRACKING = auto()
    UNCERTAIN = auto()
    LOST = auto()


class ResultSource(Enum):
    """How the controller produced the committed frame result."""

    INITIAL = auto()
    OBSERVED_CONFIRMED = auto()
    OBSERVED_REACQUIRED = auto()
    OBSERVED_WEAK_BLEND = auto()
    MOTION_PREDICTED = auto()


@dataclass(frozen=True, slots=True)
class TrackResult:
    """The single committed output for one input frame."""

    sequenceId: SequenceId
    frameIndex: FrameIndex
    bbox: BBoxXYWH
    bfov: BFoV
    confidence: float
    status: TrackStatus
    valid: bool
    resultSource: ResultSource = ResultSource.OBSERVED_CONFIRMED

    def __post_init__(self) -> None:
        if not str(self.sequenceId):
            raise ProtocolError("sequenceId must be non-empty")
        if int(self.frameIndex) < 0:
            raise ProtocolError("frameIndex must be non-negative")
        _requireProbability("track confidence", self.confidence)


class ViewProjection(Enum):
    """How the pixels of a local view map to directions on the sphere."""

    # A pinhole camera: pixels are linear on the tangent plane.  Needs FOV < 180 deg.
    PERSPECTIVE = auto()
    # Pixels are linear in longitude and latitude of the sphere rotated so that the
    # view center is at (0, 0).  Reaches the whole sphere: up to 360 x 180 degrees.
    SPHERICAL = auto()


@dataclass(frozen=True, slots=True)
class ViewSpec:
    """The local view requested from an ERP frame.

    ``bfov`` gives the view center and its extent: the camera FOV of a perspective
    view, the longitude and latitude span of a spherical one.
    """

    viewId: int
    bfov: BFoV
    outputWidthPx: int
    outputHeightPx: int
    # Where the target is expected in this view, in view pixels.  When set, the
    # backend centers its search crop on it instead of deriving one from the template.
    priorBox: BBoxXYWH | None = None
    # The target's boxes in the previous frames, in this view's pixels, oldest first.
    # They may lie outside the view.  Read by sequence-level backends.
    trajectory: tuple[BBoxXYWH, ...] = ()
    projection: ViewProjection = ViewProjection.PERSPECTIVE

    def __post_init__(self) -> None:
        if self.viewId < 0:
            raise ProtocolError("viewId must be non-negative")
        if self.outputWidthPx <= 0 or self.outputHeightPx <= 0:
            raise ProtocolError("view output dimensions must be positive")


@dataclass(frozen=True, slots=True)
class LocalView:
    """A perspective RGB crop."""

    spec: ViewSpec
    rgb: NDArray[np.uint8]
    deviceRgb: object | None = None

    def __post_init__(self) -> None:
        expectedShape = (self.spec.outputHeightPx, self.spec.outputWidthPx)
        if not isinstance(self.rgb, np.ndarray) or self.rgb.dtype != np.uint8:
            raise ProtocolError("local rgb must be a uint8 NumPy array")
        if self.rgb.shape != (*expectedShape, 3):
            raise ProtocolError(
                f"local rgb shape must be {(*expectedShape, 3)}, actual={self.rgb.shape}"
            )
        if self.deviceRgb is not None:
            shape = getattr(self.deviceRgb, "shape", None)
            ndim = getattr(self.deviceRgb, "ndim", None)
            if shape is None or ndim != 3:
                raise ProtocolError("local deviceRgb must be a [C, H, W] tensor-like object")
            if tuple(shape) != (3, self.spec.outputHeightPx, self.spec.outputWidthPx):
                raise ProtocolError(
                    "local deviceRgb shape must be "
                    f"{(3, self.spec.outputHeightPx, self.spec.outputWidthPx)}, "
                    f"actual={tuple(shape)}"
                )


@dataclass(frozen=True, slots=True)
class LocalObservation:
    viewId: int
    bbox: BBoxXYWH
    modelScore: float
    appearanceScore: float
    fusedScore: float
    latencyNs: int
    appearanceProbability: float | None = None
    presenceLogit: float | None = None
    qualityLogit: float | None = None
    presenceProbability: float | None = None
    qualityProbability: float | None = None
    predictedIoU: float | None = None
    cornerScore: float | None = None

    def __post_init__(self) -> None:
        if self.viewId < 0 or self.latencyNs < 0:
            raise ProtocolError("viewId and latencyNs must be non-negative")
        for name, value in (
            ("modelScore", self.modelScore),
            ("appearanceScore", self.appearanceScore),
            ("fusedScore", self.fusedScore),
        ):
            _requireProbability(name, value)
        if self.appearanceProbability is not None:
            _requireProbability("appearanceProbability", self.appearanceProbability)
        for name, value in (
            ("presenceLogit", self.presenceLogit),
            ("qualityLogit", self.qualityLogit),
        ):
            if value is not None:
                _requireFinite(name, value)
        for name, value in (
            ("presenceProbability", self.presenceProbability),
            ("qualityProbability", self.qualityProbability),
            ("predictedIoU", self.predictedIoU),
            ("cornerScore", self.cornerScore),
        ):
            if value is not None:
                _requireProbability(name, value)


@dataclass(frozen=True, slots=True)
class LocalBoxProjection:
    """One boundary projection shared by BFoV fitting and direct ERP bbox fitting."""

    bfov: BFoV
    bbox: BBoxXYWH
    sphericalBoundary: tuple[SphericalPoint, ...]
    erpBoundary: tuple[tuple[float, float], ...]
    indirectBbox: BBoxXYWH
    envelopeInflation: float

    def __post_init__(self) -> None:
        if len(self.sphericalBoundary) < 8:
            raise ProtocolError("projected local boundary requires at least eight samples")
        if len(self.sphericalBoundary) != len(self.erpBoundary):
            raise ProtocolError("spherical and ERP boundaries must have equal length")
        for point in self.erpBoundary:
            if len(point) != 2:
                raise ProtocolError("each ERP boundary point must contain x and y")
            _requireFinite("ERP boundary", *point)
        _requireFinite("envelopeInflation", self.envelopeInflation)
        if self.envelopeInflation <= 0.0:
            raise ProtocolError("envelopeInflation must be positive")


class TemplateCommandKind(Enum):
    KEEP = auto()
    UPDATE_RECENT = auto()
    UPDATE_STABLE = auto()
    RESET_TO_ANCHOR = auto()


@dataclass(frozen=True, slots=True)
class TemplateCommand:
    kind: TemplateCommandKind
    frameIndex: FrameIndex
    viewId: int | None
    localBox: BBoxXYWH | None
    expectedRevision: int

    def __post_init__(self) -> None:
        if int(self.frameIndex) < 0 or self.expectedRevision < 0:
            raise ProtocolError("template frameIndex and revision must be non-negative")
        if self.viewId is not None and self.viewId < 0:
            raise ProtocolError("template viewId must be non-negative")
        hasSelection = self.viewId is not None and self.localBox is not None
        if self.kind in {TemplateCommandKind.UPDATE_RECENT, TemplateCommandKind.UPDATE_STABLE}:
            if not hasSelection:
                raise ProtocolError("template update commands require viewId and localBox")
        elif self.viewId is not None or self.localBox is not None:
            raise ProtocolError("KEEP and RESET_TO_ANCHOR must not select a local box")


@dataclass(frozen=True, slots=True)
class SearchPlan:
    """The one perspective view to search in a frame and the template command to apply."""

    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    view: ViewSpec
    templateCommand: TemplateCommand
    predictedMotion: MotionState3D | None
    # Extra views searched without tracker state while the target is considered lost.
    scanViews: tuple[ViewSpec, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.sequenceId) or int(self.frameIndex) < 0 or self.stateRevision < 0:
            raise ProtocolError("search plan identity and revision must be valid")
        if self.templateCommand.frameIndex != self.frameIndex:
            raise ProtocolError("template command and search plan frameIndex must match")


@dataclass(frozen=True, slots=True)
class InitializationPlan:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    stateRevision: int
    templateView: ViewSpec
    templateBox: BBoxXYWH

    def __post_init__(self) -> None:
        if int(self.frameIndex) != 0:
            raise ProtocolError("initialization plan must target frame 0")
        if self.stateRevision < 0 or not str(self.sequenceId):
            raise ProtocolError("initialization identity and revision must be valid")


@dataclass(frozen=True, slots=True)
class ProjectedObservation:
    viewId: int
    bfov: BFoV
    bbox: BBoxXYWH
    modelScore: float
    appearanceScore: float
    motionScore: float
    scaleScore: float
    fusedScore: float
    localBox: BBoxXYWH | None = None
    backendFusedScore: float | None = None
    appearanceProbability: float | None = None
    rawMotionScore: float | None = None
    motionProbability: float | None = None
    motionReliability: float = 0.0
    singleScore: float | None = None
    erpBoundary: tuple[tuple[float, float], ...] = ()
    envelopeInflation: float = 1.0
    normalizedRadius: float = 0.0
    edgeMargin: float = 0.0
    # Feature similarity of the box to the frame-0 template; None: not measured.
    appearanceSimilarity: float | None = None

    def __post_init__(self) -> None:
        if self.viewId < 0:
            raise ProtocolError("viewId must be non-negative")
        for name, value in (
            ("modelScore", self.modelScore),
            ("appearanceScore", self.appearanceScore),
            ("motionScore", self.motionScore),
            ("scaleScore", self.scaleScore),
            ("fusedScore", self.fusedScore),
            ("motionReliability", self.motionReliability),
        ):
            _requireProbability(name, value)
        for name, value in (
            ("backendFusedScore", self.backendFusedScore),
            ("appearanceProbability", self.appearanceProbability),
            ("rawMotionScore", self.rawMotionScore),
            ("motionProbability", self.motionProbability),
            ("singleScore", self.singleScore),
        ):
            if value is not None:
                _requireProbability(name, value)
        _requireFinite(
            "projection diagnostics",
            self.envelopeInflation,
            self.normalizedRadius,
            self.edgeMargin,
        )
        if self.envelopeInflation <= 0.0:
            raise ProtocolError("envelopeInflation must be positive")
        if self.normalizedRadius < 0.0 or self.edgeMargin < 0.0:
            raise ProtocolError("projection radius and edge margin must be non-negative")


@dataclass(frozen=True, slots=True)
class AirSim360Record:
    sequenceId: SequenceId
    frameIndex: FrameIndex
    rgbPath: str
    semanticPath: str | None
    instancePath: str | None

    def __post_init__(self) -> None:
        if not str(self.sequenceId) or int(self.frameIndex) < 0:
            raise ProtocolError("AirSim360 record identity must be valid")
        if not self.rgbPath.strip():
            raise ProtocolError("AirSim360 rgbPath must be non-empty")
