"""Strict, immutable configuration loading for runtime components."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from pathlib import Path
from typing import cast

from track360.core.errors import ConfigError

SUPPORTED_SCHEMA_VERSION = 1
VISUALIZATION_STAGES = frozenset({"local_rgb", "backend_box", "geometry_box"})
GEOMETRY_RESAMPLERS = frozenset({"cpu", "cuda"})


@dataclass(frozen=True, slots=True)
class ModelConfig:
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
        if self.precision != "fp32":
            raise ConfigError(
                f"unsupported model.precision: {self.precision}; only fp32 is implemented"
            )


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    calibrationArtifact: Path | None
    requireCheckpointHashMatch: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.requireCheckpointHashMatch, bool):
            raise ConfigError("scoring.requireCheckpointHashMatch must be boolean")


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    viewWidthPx: int
    viewHeightPx: int
    boundarySamplesPerEdge: int
    minFovRad: float
    maxFovRad: float
    resampler: str = "cpu"

    def __post_init__(self) -> None:
        if self.viewWidthPx <= 0 or self.viewHeightPx <= 0:
            raise ConfigError("geometry view dimensions must be positive")
        if self.resampler not in GEOMETRY_RESAMPLERS:
            raise ConfigError(f"unsupported geometry.resampler: {self.resampler}")
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
    candidateMinScore: float
    stableFramesBeforeUpdate: int
    windowLength: int
    contextScale: float = 2.0
    contextMarginRatio: float = 0.15
    maxPredictionHorizon: int = 3

    def __post_init__(self) -> None:
        _requireProbability("tracking.candidateMinScore", self.candidateMinScore)
        if self.stableFramesBeforeUpdate <= 0:
            raise ConfigError("tracking.stableFramesBeforeUpdate must be positive")
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

    sequenceModel: bool = True
    acceptAnyCandidate: bool = True
    viewHorizontalFovCapRad: float | None = pi / 2.0
    viewVerticalFovCapRad: float | None = pi / 2.0
    fullViewSearch: bool = False
    alignedSearch: bool = True
    alignedMinFovRad: float = pi / 90.0
    sphericalSearch: bool = True
    sphericalSearchFovRad: float = 2.0 * pi / 3.0
    predictiveSearch: bool = True
    useMotionScore: bool = False
    templateFovScale: float = 2.5
    onlineTemplate: bool = True
    templateMinConfidence: float = 0.515
    holdWeakBox: bool = True

    def __post_init__(self) -> None:
        for name in (
            "sequenceModel",
            "acceptAnyCandidate",
            "fullViewSearch",
            "alignedSearch",
            "sphericalSearch",
            "predictiveSearch",
            "useMotionScore",
            "onlineTemplate",
            "holdWeakBox",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"backendTuning.{name} must be boolean")
        for name in ("viewHorizontalFovCapRad", "viewVerticalFovCapRad"):
            value = getattr(self, name)
            if value is not None and not 0.0 < value < pi:
                raise ConfigError(f"backendTuning.{name} must be in (0, pi)")
        if self.alignedSearch and self.fullViewSearch:
            raise ConfigError(
                "backendTuning.alignedSearch and fullViewSearch cannot both be enabled"
            )
        if not 0.0 < self.sphericalSearchFovRad < 2.0 * pi:
            raise ConfigError("backendTuning.sphericalSearchFovDeg must be in (0, 360)")
        if not 0.0 < self.alignedMinFovRad < pi:
            raise ConfigError("backendTuning.alignedMinFovDeg must be in (0, 180)")
        if not isfinite(self.templateFovScale) or self.templateFovScale < 1.0:
            raise ConfigError("backendTuning.templateFovScale must be at least 1")
        _requireProbability("backendTuning.templateMinConfidence", self.templateMinConfidence)


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
    scoring: ScoringConfig
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
            "scoring",
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
    scoringRaw = _section(
        root,
        "scoring",
        {"calibrationArtifact", "requireCheckpointHashMatch"},
    )
    geometryRaw = _section(
        root,
        "geometry",
        {
            "viewWidthPx",
            "viewHeightPx",
            "boundarySamplesPerEdge",
            "minFovDeg",
            "maxFovDeg",
            "resampler",
        },
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
        {
            "candidateMinScore",
            "stableFramesBeforeUpdate",
            "windowLength",
            "contextScale",
            "contextMarginRatio",
            "maxPredictionHorizon",
        },
    )
    tuningRaw = _section(
        root,
        "backendTuning",
        {
            "sequenceModel",
            "acceptAnyCandidate",
            "viewHorizontalFovCapDeg",
            "viewVerticalFovCapDeg",
            "fullViewSearch",
            "alignedSearch",
            "alignedMinFovDeg",
            "sphericalSearch",
            "sphericalSearchFovDeg",
            "predictiveSearch",
            "useMotionScore",
            "templateFovScale",
            "onlineTemplate",
            "templateMinConfidence",
            "holdWeakBox",
        },
    )
    reproducibilityRaw = _section(root, "reproducibility", {"seed", "deterministic"})
    visualizationRaw = _section(root, "visualization", {"enabled", "outputRoot", "stages"})

    weightsValue = _requireStr("model.weights", modelRaw["weights"])
    weightsPath = Path(weightsValue).expanduser()
    if not weightsPath.is_absolute():
        weightsPath = (configPath.parent / weightsPath).resolve()

    calibrationValue = scoringRaw["calibrationArtifact"]
    if calibrationValue is None:
        calibrationPath = None
    else:
        calibrationPath = Path(
            _requireStr("scoring.calibrationArtifact", calibrationValue)
        ).expanduser()
        if not calibrationPath.is_absolute():
            calibrationPath = (configPath.parent / calibrationPath).resolve()

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
        scoring=ScoringConfig(
            calibrationArtifact=calibrationPath,
            requireCheckpointHashMatch=_requireBool(
                "scoring.requireCheckpointHashMatch",
                scoringRaw["requireCheckpointHashMatch"],
            ),
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
            resampler=_requireStr("geometry.resampler", geometryRaw["resampler"]),
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
            candidateMinScore=_requireFloat(
                "tracking.candidateMinScore", trackingRaw["candidateMinScore"]
            ),
            stableFramesBeforeUpdate=_requireInt(
                "tracking.stableFramesBeforeUpdate", trackingRaw["stableFramesBeforeUpdate"]
            ),
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
            sequenceModel=_requireBool(
                "backendTuning.sequenceModel", tuningRaw["sequenceModel"]
            ),
            acceptAnyCandidate=_requireBool(
                "backendTuning.acceptAnyCandidate", tuningRaw["acceptAnyCandidate"]
            ),
            viewHorizontalFovCapRad=_optionalDegreesToRadians(
                "backendTuning.viewHorizontalFovCapDeg",
                tuningRaw["viewHorizontalFovCapDeg"],
            ),
            viewVerticalFovCapRad=_optionalDegreesToRadians(
                "backendTuning.viewVerticalFovCapDeg",
                tuningRaw["viewVerticalFovCapDeg"],
            ),
            fullViewSearch=_requireBool(
                "backendTuning.fullViewSearch", tuningRaw["fullViewSearch"]
            ),
            alignedSearch=_requireBool(
                "backendTuning.alignedSearch", tuningRaw["alignedSearch"]
            ),
            sphericalSearch=_requireBool(
                "backendTuning.sphericalSearch", tuningRaw["sphericalSearch"]
            ),
            predictiveSearch=_requireBool(
                "backendTuning.predictiveSearch", tuningRaw["predictiveSearch"]
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
            useMotionScore=_requireBool(
                "backendTuning.useMotionScore", tuningRaw["useMotionScore"]
            ),
            templateFovScale=_requireFloat(
                "backendTuning.templateFovScale", tuningRaw["templateFovScale"]
            ),
            onlineTemplate=_requireBool(
                "backendTuning.onlineTemplate", tuningRaw["onlineTemplate"]
            ),
            templateMinConfidence=_requireFloat(
                "backendTuning.templateMinConfidence", tuningRaw["templateMinConfidence"]
            ),
            holdWeakBox=_requireBool("backendTuning.holdWeakBox", tuningRaw["holdWeakBox"]),
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
