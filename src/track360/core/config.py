"""Strict, immutable configuration loading for runtime components."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from pathlib import Path
from typing import cast

from track360.core.errors import ConfigError

SUPPORTED_SCHEMA_VERSION = 2
VISUALIZATION_STAGES = frozenset({"local_rgb", "backend_box", "geometry_box"})


MODEL_PRECISIONS = ("fp32", "tf32")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """``precision``: fp32 (the reference) or tf32 (float32 with TensorFloat-32
    matrix products on the GPU: faster, results differ in the last digits).

    Half precision was tried and dropped: the forward pass is bound by the work
    of launching its many small operations, so it ran no faster than tf32 in the
    pipeline and moved the boxes more (evaluation log E039)."""

    backend: str
    variant: str
    weights: Path
    precision: str

    def __post_init__(self) -> None:
        if self.backend not in {"pytorch", "onnxruntime", "tensorrt"}:
            raise ConfigError(f"unsupported model.backend: {self.backend}")
        if self.variant.lower().replace("-", "_") != "artrackv2_b_256":
            raise ConfigError("model.variant must be artrackv2_b_256")
        if not self.variant.strip():
            raise ConfigError("model.variant must be non-empty")
        if self.precision not in MODEL_PRECISIONS:
            raise ConfigError(
                f"unsupported model.precision: {self.precision}; "
                f"expected one of {', '.join(MODEL_PRECISIONS)}"
            )


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    viewWidthPx: int
    viewHeightPx: int
    boundarySamplesPerEdge: int
    minFovRad: float
    maxFovRad: float

    def __post_init__(self) -> None:
        if self.viewWidthPx <= 0 or self.viewHeightPx <= 0:
            raise ConfigError("geometry view dimensions must be positive")
        if self.boundarySamplesPerEdge < 2:
            raise ConfigError("geometry.boundarySamplesPerEdge must be at least 2")
        if not 0.0 < self.minFovRad < self.maxFovRad < pi:
            raise ConfigError("geometry FOV must satisfy 0 < minFovRad < maxFovRad < pi")


@dataclass(frozen=True, slots=True)
class MotionConfig:
    minSamplesForVelocity: int = 2
    maxTangentSpanRad: float = 1.20
    huberDeltaRad: float = 0.15
    processNoiseRadPerSec: float = 0.04
    maxAngularSpeedRadPerSec: float = 2.0
    maxLogScaleRatePerSec: float = 1.0

    def __post_init__(self) -> None:
        if self.minSamplesForVelocity < 2:
            raise ConfigError("motion.minSamplesForVelocity must be at least 2")
        for name, value in (
            ("maxTangentSpanRad", self.maxTangentSpanRad),
            ("huberDeltaRad", self.huberDeltaRad),
            ("processNoiseRadPerSec", self.processNoiseRadPerSec),
            ("maxAngularSpeedRadPerSec", self.maxAngularSpeedRadPerSec),
            ("maxLogScaleRatePerSec", self.maxLogScaleRatePerSec),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ConfigError(f"motion.{name} must be positive and finite")


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    windowLength: int
    contextScale: float = 2.0
    contextMarginRatio: float = 0.15
    maxPredictionHorizon: int = 3

    def __post_init__(self) -> None:
        if self.windowLength < 2:
            raise ConfigError("tracking.windowLength must be at least 2")
        if not isfinite(self.contextScale) or self.contextScale < 2.0:
            raise ConfigError("tracking.contextScale must be at least 2")
        if not isfinite(self.contextMarginRatio) or self.contextMarginRatio < 0.0:
            raise ConfigError("tracking.contextMarginRatio must be non-negative")
        if self.maxPredictionHorizon <= 0:
            raise ConfigError("tracking.maxPredictionHorizon must be positive")


@dataclass(frozen=True, slots=True)
class BackendTuningConfig:
    """Controller and backend switches tuned for the ARTrackV2 backend.

    The defaults are the production operating point and must stay identical to
    ``configs/default.yaml``.  A ``None`` FOV cap means only the geometry FOV limit
    applies.
    """

    viewHorizontalFovCapRad: float | None = pi / 2.0
    viewVerticalFovCapRad: float | None = pi / 2.0
    alignedMinFovRad: float = pi / 90.0
    sphericalSearch: bool = True
    sphericalSearchFovRad: float = 2.0 * pi / 3.0
    templateFovScale: float = 2.5
    # State score: a weighted mean of the backend score, the appearance similarity
    # and the motion score; a frame below ``uncertainScore`` is not trusted.  It is
    # what the reported status follows when loss handling is off.
    stateBackendWeight: float = 0.40
    stateAppearanceWeight: float = 0.55
    stateMotionWeight: float = 0.05
    motionOffsetScale: float = 0.5
    motionSizeScale: float = 0.1
    uncertainScore: float = 0.42
    lostAfterFrames: int = 4
    # Loss handling: doubt a frame, search for the target, jump back to it.  The
    # judgement compares each score with the median of the frames trusted so far (a
    # frame is trusted while it is no more than ``relativeGate`` below it); the mean
    # of the two deviations below ``relativeEnterDeviation`` raises a doubt that
    # holds until ``releaseFrames`` calm frames.
    lossHandling: bool = False
    # none: only judge the state; jump: search and jump.
    lossActions: str = "jump"
    releaseFrames: int = 3
    relativeGate: float = 0.05
    relativeEnterDeviation: float = -0.53
    # Search views a sequence may spend: this many are earned per frame and at most
    # ``scanBudgetBurst`` are saved up.  0 per frame: no limit.
    scanBudgetPerFrame: float = 0.0
    scanBudgetBurst: float = 40.0
    # A candidate is taken from this score of a pass without the tracker's memory.
    reacquireScore: float = 0.70
    # The enlarged search view: this many times the normal one, and after how many
    # lost frames the larger one is used.
    zoomFirstScale: float = 2.0
    zoomLastScale: float = 4.0
    zoomLastAfterFrames: int = 20

    def __post_init__(self) -> None:
        for name in ("lossHandling", "sphericalSearch"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"backendTuning.{name} must be boolean")
        for name in ("viewHorizontalFovCapRad", "viewVerticalFovCapRad"):
            value = getattr(self, name)
            if value is not None and not 0.0 < value < pi:
                raise ConfigError(f"backendTuning.{name} must be in (0, pi)")
        if min(self.zoomFirstScale, self.zoomLastScale) < 1.0:
            raise ConfigError("backendTuning zoom scales must be at least 1")
        if self.zoomLastAfterFrames < 0:
            raise ConfigError("backendTuning.zoomLastAfterFrames must be non-negative")
        if not 0.0 <= self.relativeGate <= 1.0 or not -1.0 <= self.relativeEnterDeviation < 0:
            raise ConfigError(
                "backendTuning.relativeGate must be in [0, 1] and "
                "relativeEnterDeviation in [-1, 0)"
            )
        if self.lossActions not in ("none", "jump"):
            raise ConfigError("backendTuning.lossActions must be none or jump")
        if self.releaseFrames < 1:
            raise ConfigError("backendTuning.releaseFrames must be positive")
        for name in ("uncertainScore", "reacquireScore"):
            if not -1.0 <= getattr(self, name) <= 1.0:
                raise ConfigError(f"backendTuning.{name} must be in [-1, 1]")
        weights = (
            self.stateBackendWeight,
            self.stateAppearanceWeight,
            self.stateMotionWeight,
        )
        if min(weights) < 0.0 or self.stateBackendWeight + self.stateMotionWeight <= 0.0:
            raise ConfigError(
                "backendTuning state weights must be non-negative, and the backend and "
                "motion weights must not both be zero"
            )
        if self.motionOffsetScale <= 0.0 or self.motionSizeScale <= 0.0:
            raise ConfigError("backendTuning motion scales must be positive")
        if self.scanBudgetPerFrame < 0.0 or self.scanBudgetBurst < 0.0:
            raise ConfigError("backendTuning scan budget values must be non-negative")
        if self.lostAfterFrames < 1:
            raise ConfigError("backendTuning.lostAfterFrames must be positive")
        if not 0.0 < self.sphericalSearchFovRad < 2.0 * pi:
            raise ConfigError("backendTuning.sphericalSearchFovDeg must be in (0, 360)")
        if not 0.0 < self.alignedMinFovRad < pi:
            raise ConfigError("backendTuning.alignedMinFovDeg must be in (0, 180)")
        if not isfinite(self.templateFovScale) or self.templateFovScale < 1.0:
            raise ConfigError("backendTuning.templateFovScale must be at least 1")


@dataclass(frozen=True, slots=True)
class ReproducibilityConfig:
    seed: int = 0
    deterministic: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ConfigError("reproducibility.seed must be a non-negative integer")
        if not isinstance(self.deterministic, bool):
            raise ConfigError("reproducibility.deterministic must be boolean")


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    enabled: bool
    outputRoot: Path
    stages: frozenset[str]

    def __post_init__(self) -> None:
        unknown = self.stages - VISUALIZATION_STAGES
        if unknown:
            raise ConfigError(f"unsupported visualization stages: {sorted(unknown)}")
        if self.enabled and not self.stages:
            raise ConfigError("visualization.stages must not be empty when enabled")


@dataclass(frozen=True, slots=True)
class AppConfig:
    schemaVersion: int
    model: ModelConfig
    geometry: GeometryConfig
    motion: MotionConfig
    tracking: TrackingConfig
    backendTuning: BackendTuningConfig
    reproducibility: ReproducibilityConfig
    visualization: VisualizationConfig
    sourcePath: Path


def loadConfig(path: str | Path) -> AppConfig:
    """Load a YAML configuration and reject unknown or missing fields."""
    configPath = Path(path).expanduser().resolve()
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise ConfigError("PyYAML is required to load configuration files") from error

    try:
        with configPath.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except OSError as error:
        raise ConfigError(f"cannot read config file {configPath}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {configPath}: {error}") from error

    root = _requireMapping("config", raw)
    _requireKeys(
        "config",
        root,
        {
            "schemaVersion",
            "model",
            "geometry",
            "motion",
            "tracking",
            "backendTuning",
            "reproducibility",
            "visualization",
        },
    )

    schemaVersion = _requireInt("schemaVersion", root["schemaVersion"])
    if schemaVersion != SUPPORTED_SCHEMA_VERSION:
        raise ConfigError(
            f"unsupported schemaVersion: expected={SUPPORTED_SCHEMA_VERSION}, "
            f"actual={schemaVersion}"
        )

    modelRaw = _section(root, "model", {"backend", "variant", "weights", "precision"})
    geometryRaw = _section(
        root,
        "geometry",
        {"viewWidthPx", "viewHeightPx", "boundarySamplesPerEdge", "minFovDeg", "maxFovDeg"},
    )
    motionRaw = _section(
        root,
        "motion",
        {
            "minSamplesForVelocity",
            "maxTangentSpanRad",
            "huberDeltaRad",
            "processNoiseRadPerSec",
            "maxAngularSpeedRadPerSec",
            "maxLogScaleRatePerSec",
        },
    )
    trackingRaw = _section(
        root,
        "tracking",
        {"windowLength", "contextScale", "contextMarginRatio", "maxPredictionHorizon"},
    )
    tuningFloats = (
        "templateFovScale",
        "stateBackendWeight",
        "stateAppearanceWeight",
        "stateMotionWeight",
        "motionOffsetScale",
        "motionSizeScale",
        "uncertainScore",
        "relativeGate",
        "relativeEnterDeviation",
        "scanBudgetPerFrame",
        "scanBudgetBurst",
        "reacquireScore",
        "zoomFirstScale",
        "zoomLastScale",
    )
    tuningInts = ("lostAfterFrames", "releaseFrames", "zoomLastAfterFrames")
    tuningRaw = _section(
        root,
        "backendTuning",
        {
            "viewHorizontalFovCapDeg",
            "viewVerticalFovCapDeg",
            "alignedMinFovDeg",
            "sphericalSearch",
            "sphericalSearchFovDeg",
            "lossHandling",
            "lossActions",
            *tuningFloats,
            *tuningInts,
        },
    )
    reproducibilityRaw = _section(root, "reproducibility", {"seed", "deterministic"})
    visualizationRaw = _section(root, "visualization", {"enabled", "outputRoot", "stages"})

    weightsValue = _requireStr("model.weights", modelRaw["weights"])
    weightsPath = Path(weightsValue).expanduser()
    if not weightsPath.is_absolute():
        weightsPath = (configPath.parent / weightsPath).resolve()

    outputRootValue = _requireStr("visualization.outputRoot", visualizationRaw["outputRoot"])
    outputRoot = Path(outputRootValue).expanduser()
    if not outputRoot.is_absolute():
        outputRoot = (configPath.parent / outputRoot).resolve()

    return AppConfig(
        schemaVersion=schemaVersion,
        model=ModelConfig(
            backend=_requireStr("model.backend", modelRaw["backend"]),
            variant=_requireStr("model.variant", modelRaw["variant"]),
            weights=weightsPath,
            precision=_requireStr("model.precision", modelRaw["precision"]),
        ),
        geometry=GeometryConfig(
            viewWidthPx=_requireInt("geometry.viewWidthPx", geometryRaw["viewWidthPx"]),
            viewHeightPx=_requireInt("geometry.viewHeightPx", geometryRaw["viewHeightPx"]),
            boundarySamplesPerEdge=_requireInt(
                "geometry.boundarySamplesPerEdge", geometryRaw["boundarySamplesPerEdge"]
            ),
            minFovRad=_degreesToRadians(
                "geometry.minFovDeg", _requireFloat("geometry.minFovDeg", geometryRaw["minFovDeg"])
            ),
            maxFovRad=_degreesToRadians(
                "geometry.maxFovDeg", _requireFloat("geometry.maxFovDeg", geometryRaw["maxFovDeg"])
            ),
        ),
        motion=MotionConfig(
            minSamplesForVelocity=_requireInt(
                "motion.minSamplesForVelocity", motionRaw["minSamplesForVelocity"]
            ),
            maxTangentSpanRad=_requireFloat(
                "motion.maxTangentSpanRad", motionRaw["maxTangentSpanRad"]
            ),
            huberDeltaRad=_requireFloat("motion.huberDeltaRad", motionRaw["huberDeltaRad"]),
            processNoiseRadPerSec=_requireFloat(
                "motion.processNoiseRadPerSec", motionRaw["processNoiseRadPerSec"]
            ),
            maxAngularSpeedRadPerSec=_requireFloat(
                "motion.maxAngularSpeedRadPerSec", motionRaw["maxAngularSpeedRadPerSec"]
            ),
            maxLogScaleRatePerSec=_requireFloat(
                "motion.maxLogScaleRatePerSec", motionRaw["maxLogScaleRatePerSec"]
            ),
        ),
        tracking=TrackingConfig(
            windowLength=_requireInt("tracking.windowLength", trackingRaw["windowLength"]),
            contextScale=_requireFloat("tracking.contextScale", trackingRaw["contextScale"]),
            contextMarginRatio=_requireFloat(
                "tracking.contextMarginRatio", trackingRaw["contextMarginRatio"]
            ),
            maxPredictionHorizon=_requireInt(
                "tracking.maxPredictionHorizon", trackingRaw["maxPredictionHorizon"]
            ),
        ),
        backendTuning=BackendTuningConfig(
            viewHorizontalFovCapRad=_optionalDegreesToRadians(
                "backendTuning.viewHorizontalFovCapDeg",
                tuningRaw["viewHorizontalFovCapDeg"],
            ),
            viewVerticalFovCapRad=_optionalDegreesToRadians(
                "backendTuning.viewVerticalFovCapDeg",
                tuningRaw["viewVerticalFovCapDeg"],
            ),
            sphericalSearch=_requireBool(
                "backendTuning.sphericalSearch", tuningRaw["sphericalSearch"]
            ),
            sphericalSearchFovRad=_degreesToRadians(
                "backendTuning.sphericalSearchFovDeg",
                _requireFloat(
                    "backendTuning.sphericalSearchFovDeg", tuningRaw["sphericalSearchFovDeg"]
                ),
            ),
            alignedMinFovRad=_degreesToRadians(
                "backendTuning.alignedMinFovDeg",
                _requireFloat("backendTuning.alignedMinFovDeg", tuningRaw["alignedMinFovDeg"]),
            ),
            lossHandling=_requireBool(
                "backendTuning.lossHandling", tuningRaw["lossHandling"]
            ),
            lossActions=_requireStr("backendTuning.lossActions", tuningRaw["lossActions"]),
            **{
                name: _requireFloat(f"backendTuning.{name}", tuningRaw[name])
                for name in tuningFloats
            },
            **{
                name: _requireInt(f"backendTuning.{name}", tuningRaw[name])
                for name in tuningInts
            },
        ),
        reproducibility=ReproducibilityConfig(
            seed=_requireInt("reproducibility.seed", reproducibilityRaw["seed"]),
            deterministic=_requireBool(
                "reproducibility.deterministic", reproducibilityRaw["deterministic"]
            ),
        ),
        visualization=VisualizationConfig(
            enabled=_requireBool("visualization.enabled", visualizationRaw["enabled"]),
            outputRoot=outputRoot,
            stages=_requireStringSet("visualization.stages", visualizationRaw["stages"]),
        ),
        sourcePath=configPath,
    )


def _requireProbability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{name} must be in [0, 1], actual={value}")


def _requireMapping(name: str, value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{name} must be a mapping with string keys")
    return cast(dict[str, object], value)


def _requireKeys(name: str, value: dict[str, object], expected: set[str]) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ConfigError(f"{name} fields invalid: missing={missing}, unknown={unknown}")


def _section(
    root: dict[str, object],
    name: str,
    expected: set[str],
) -> dict[str, object]:
    section = _requireMapping(name, root[name])
    _requireKeys(name, section, expected)
    return section


def _requireStr(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _requireBool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return value


def _requireInt(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    return value


def _requireFloat(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ConfigError(f"{name} must be finite")
    return result


def _optionalDegreesToRadians(name: str, value: object) -> float | None:
    return None if value is None else _degreesToRadians(name, _requireFloat(name, value))


def _requireStringSet(name: str, value: object) -> frozenset[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ConfigError(f"{name} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ConfigError(f"{name} must not contain duplicates")
    return frozenset(value)


def _degreesToRadians(name: str, valueDeg: float) -> float:
    if not 0.0 < valueDeg < 180.0:
        raise ConfigError(f"{name} must be in (0, 180), actual={valueDeg}")
    return valueDeg * pi / 180.0
