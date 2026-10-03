"""Strict, immutable configuration loading for runtime components."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, pi
from pathlib import Path
from typing import cast

from track360.core.errors import ConfigError

SUPPORTED_SCHEMA_VERSION = 1
VISUALIZATION_STAGES = frozenset({"local_rgb", "backend_box", "geometry_box"})
GEOMETRY_RESAMPLERS = frozenset({"cpu", "cuda"})
FUSION_BOX_MODES = frozenset({"reference_adaptive", "best_source"})


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
        if not isclose(self.maxFovRad, 2.0 * pi / 3.0, abs_tol=1e-9):
            raise ConfigError("geometry.maxFovDeg must be 120 for fixed search views")


@dataclass(frozen=True, slots=True)
class EvaluatorConfig:
    supportWeight: float = 0.25
    agreementWeight: float = 0.25
    minReacquireViews: int = 2
    successRate: float = 0.90
    firstRoundFusionOverlap: float = 0.30
    overlapThreshold: float = 0.70
    fusionSourceMinConfidence: float = 0.80
    fusionBoxMode: str = "reference_adaptive"

    def __post_init__(self) -> None:
        _requireProbability("evaluator.supportWeight", self.supportWeight)
        _requireProbability("evaluator.agreementWeight", self.agreementWeight)
        if self.supportWeight + self.agreementWeight > 1.0:
            raise ConfigError("evaluator support and agreement weights must sum to at most 1")
        if self.minReacquireViews <= 0:
            raise ConfigError("evaluator.minReacquireViews must be positive")
        _requireProbability("evaluator.successRate", self.successRate)
        _requireProbability(
            "evaluator.firstRoundFusionOverlap", self.firstRoundFusionOverlap
        )
        _requireProbability("evaluator.overlapThreshold", self.overlapThreshold)
        _requireProbability(
            "evaluator.fusionSourceMinConfidence", self.fusionSourceMinConfidence
        )
        if self.fusionBoxMode not in FUSION_BOX_MODES:
            raise ConfigError(
                "evaluator.fusionBoxMode must be 'reference_adaptive' or 'best_source'"
            )
        if self.firstRoundFusionOverlap >= self.overlapThreshold:
            raise ConfigError(
                "evaluator thresholds must satisfy firstRoundFusionOverlap < overlapThreshold"
            )


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
    scaleClusterTolerance: float = 0.50
    maxPredictionHorizon: int = 3
    guardYawStepRad: float = 2.0 * pi / 3.0
    minViewsForCommit: int = 2
    sameFrameEscalationEnabled: bool = True
    maxAttemptsPerFrame: int = 2
    maxViewsPerFrameTotal: int = 12
    uncertainFovScale: float = 1.25
    reacquireCooldownFrames: int = 2

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
        if not isfinite(self.scaleClusterTolerance) or self.scaleClusterTolerance <= 0.0:
            raise ConfigError("tracking.scaleClusterTolerance must be positive")
        if self.maxPredictionHorizon <= 0:
            raise ConfigError("tracking.maxPredictionHorizon must be positive")
        if not 0.0 < self.guardYawStepRad < pi:
            raise ConfigError("tracking.guardYawStepRad must be in (0, pi)")
        if self.minViewsForCommit <= 0:
            raise ConfigError("tracking.minViewsForCommit must be positive")
        if self.maxAttemptsPerFrame != 2:
            raise ConfigError("tracking.maxAttemptsPerFrame must be 2 for the two-round controller")
        minimumBudget = 12
        if self.maxViewsPerFrameTotal < max(minimumBudget, self.minViewsForCommit):
            raise ConfigError(
                "tracking.maxViewsPerFrameTotal must cover the configured state routes "
                "and all six cube-map faces"
            )
        if not isfinite(self.uncertainFovScale) or self.uncertainFovScale < 1.0:
            raise ConfigError("tracking.uncertainFovScale must be at least 1")
        if self.reacquireCooldownFrames < 0:
            raise ConfigError("tracking.reacquireCooldownFrames must be non-negative")


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    maxViewsPerFrame: int
    globalSearchInterval: int
    ringRadii: tuple[float, ...] = (1.0, 1.75, 2.5)
    viewsPerRing: tuple[int, ...] = (4, 8, 12)
    cubeMapOverlapRatio: float = 0.10
    maxCoveredCells: int = 256

    def __post_init__(self) -> None:
        if self.maxViewsPerFrame < 6 or self.globalSearchInterval <= 0:
            raise ConfigError(
                "recovery.maxViewsPerFrame must allow six cube-map faces "
                "and interval must be positive"
            )
        if not self.ringRadii or len(self.ringRadii) != len(self.viewsPerRing):
            raise ConfigError("recovery rings and viewsPerRing must have equal non-zero length")
        if any(not isfinite(radius) or radius <= 0.0 for radius in self.ringRadii):
            raise ConfigError("recovery.ringRadii must contain positive finite values")
        if any(viewCount <= 0 for viewCount in self.viewsPerRing):
            raise ConfigError("recovery.viewsPerRing must contain positive integers")
        _requireProbability("recovery.cubeMapOverlapRatio", self.cubeMapOverlapRatio)
        if self.maxCoveredCells <= 0:
            raise ConfigError("recovery.maxCoveredCells must be positive")


@dataclass(frozen=True, slots=True)
class BackendTuningConfig:
    """Controller and backend switches tuned for the ARTrackV2 backend.

    The defaults are the production operating point and must stay identical to
    ``configs/default.yaml``.  ``None`` means "no override": the corresponding
    ``evaluator`` value, fusion constant or geometry FOV limit applies.
    """

    acceptAnyCandidate: bool = True
    directMode: bool = False
    singleRound: bool = False
    adaptiveViewCount: bool = False
    singleView: bool = False
    singleViewHorizontalFovCapRad: float | None = pi / 2.0
    singleViewVerticalFovCapRad: float | None = pi / 2.0
    fourViewFovCapRad: float | None = None
    fullViewSearch: bool = False
    useMotionScore: bool = False
    templateFovScale: float = 2.5
    onlineTemplate: bool = True
    templateMinConfidence: float = 0.515
    allowSingleViewTemplate: bool = True
    holdWeakBox: bool = True
    fusionSourceMinConfidence: float | None = 0.35
    fusionOverlap: float | None = 0.45
    fusionBoxMode: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "acceptAnyCandidate",
            "directMode",
            "singleRound",
            "adaptiveViewCount",
            "singleView",
            "fullViewSearch",
            "useMotionScore",
            "onlineTemplate",
            "allowSingleViewTemplate",
            "holdWeakBox",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"backendTuning.{name} must be boolean")
        for name in ("singleViewHorizontalFovCapRad", "singleViewVerticalFovCapRad"):
            value = getattr(self, name)
            if value is not None and not 0.0 < value < pi:
                raise ConfigError(f"backendTuning.{name} must be in (0, pi)")
        if self.fourViewFovCapRad is not None and not pi / 6.0 <= self.fourViewFovCapRad < pi:
            raise ConfigError("backendTuning.fourViewFovCapDeg must be in [30, 180)")
        if not isfinite(self.templateFovScale) or self.templateFovScale < 1.0:
            raise ConfigError("backendTuning.templateFovScale must be at least 1")
        _requireProbability("backendTuning.templateMinConfidence", self.templateMinConfidence)
        if self.fusionSourceMinConfidence is not None:
            _requireProbability(
                "backendTuning.fusionSourceMinConfidence", self.fusionSourceMinConfidence
            )
        if self.fusionOverlap is not None:
            _requireProbability("backendTuning.fusionOverlap", self.fusionOverlap)
        if self.fusionBoxMode is not None and self.fusionBoxMode not in FUSION_BOX_MODES:
            raise ConfigError(
                "backendTuning.fusionBoxMode must be null, 'reference_adaptive' or 'best_source'"
            )


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
    evaluator: EvaluatorConfig
    motion: MotionConfig
    tracking: TrackingConfig
    recovery: RecoveryConfig
    backendTuning: BackendTuningConfig
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
            "evaluator",
            "motion",
            "tracking",
            "recovery",
            "backendTuning",
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
    evaluatorRaw = _section(
        root,
        "evaluator",
        {
            "supportWeight",
            "agreementWeight",
            "minReacquireViews",
            "successRate",
            "firstRoundFusionOverlap",
            "overlapThreshold",
            "fusionSourceMinConfidence",
            "fusionBoxMode",
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
            "scaleClusterTolerance",
            "maxPredictionHorizon",
            "guardYawStepDeg",
            "minViewsForCommit",
            "sameFrameEscalationEnabled",
            "maxAttemptsPerFrame",
            "maxViewsPerFrameTotal",
            "uncertainFovScale",
            "reacquireCooldownFrames",
        },
    )
    recoveryRaw = _section(
        root,
        "recovery",
        {
            "maxViewsPerFrame",
            "globalSearchInterval",
            "ringRadii",
            "viewsPerRing",
            "cubeMapOverlapRatio",
            "maxCoveredCells",
        },
    )
    tuningRaw = _section(
        root,
        "backendTuning",
        {
            "acceptAnyCandidate",
            "directMode",
            "singleRound",
            "adaptiveViewCount",
            "singleView",
            "singleViewHorizontalFovCapDeg",
            "singleViewVerticalFovCapDeg",
            "fourViewFovCapDeg",
            "fullViewSearch",
            "useMotionScore",
            "templateFovScale",
            "onlineTemplate",
            "templateMinConfidence",
            "allowSingleViewTemplate",
            "holdWeakBox",
            "fusionSourceMinConfidence",
            "fusionOverlap",
            "fusionBoxMode",
        },
    )
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
        evaluator=EvaluatorConfig(
            supportWeight=_requireFloat("evaluator.supportWeight", evaluatorRaw["supportWeight"]),
            agreementWeight=_requireFloat(
                "evaluator.agreementWeight", evaluatorRaw["agreementWeight"]
            ),
            minReacquireViews=_requireInt(
                "evaluator.minReacquireViews", evaluatorRaw["minReacquireViews"]
            ),
            successRate=_requireFloat("evaluator.successRate", evaluatorRaw["successRate"]),
            firstRoundFusionOverlap=_requireFloat(
                "evaluator.firstRoundFusionOverlap",
                evaluatorRaw["firstRoundFusionOverlap"],
            ),
            overlapThreshold=_requireFloat(
                "evaluator.overlapThreshold", evaluatorRaw["overlapThreshold"]
            ),
            fusionSourceMinConfidence=_requireFloat(
                "evaluator.fusionSourceMinConfidence",
                evaluatorRaw["fusionSourceMinConfidence"],
            ),
            fusionBoxMode=_requireStr(
                "evaluator.fusionBoxMode",
                evaluatorRaw["fusionBoxMode"],
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
            scaleClusterTolerance=_requireFloat(
                "tracking.scaleClusterTolerance", trackingRaw["scaleClusterTolerance"]
            ),
            maxPredictionHorizon=_requireInt(
                "tracking.maxPredictionHorizon", trackingRaw["maxPredictionHorizon"]
            ),
            guardYawStepRad=_degreesToRadians(
                "tracking.guardYawStepDeg",
                _requireFloat("tracking.guardYawStepDeg", trackingRaw["guardYawStepDeg"]),
            ),
            minViewsForCommit=_requireInt(
                "tracking.minViewsForCommit", trackingRaw["minViewsForCommit"]
            ),
            sameFrameEscalationEnabled=_requireBool(
                "tracking.sameFrameEscalationEnabled",
                trackingRaw["sameFrameEscalationEnabled"],
            ),
            maxAttemptsPerFrame=_requireInt(
                "tracking.maxAttemptsPerFrame", trackingRaw["maxAttemptsPerFrame"]
            ),
            maxViewsPerFrameTotal=_requireInt(
                "tracking.maxViewsPerFrameTotal", trackingRaw["maxViewsPerFrameTotal"]
            ),
            uncertainFovScale=_requireFloat(
                "tracking.uncertainFovScale", trackingRaw["uncertainFovScale"]
            ),
            reacquireCooldownFrames=_requireInt(
                "tracking.reacquireCooldownFrames", trackingRaw["reacquireCooldownFrames"]
            ),
        ),
        recovery=RecoveryConfig(
            maxViewsPerFrame=_requireInt(
                "recovery.maxViewsPerFrame", recoveryRaw["maxViewsPerFrame"]
            ),
            globalSearchInterval=_requireInt(
                "recovery.globalSearchInterval", recoveryRaw["globalSearchInterval"]
            ),
            ringRadii=_requireFloatTuple("recovery.ringRadii", recoveryRaw["ringRadii"]),
            viewsPerRing=_requireIntTuple("recovery.viewsPerRing", recoveryRaw["viewsPerRing"]),
            cubeMapOverlapRatio=_requireFloat(
                "recovery.cubeMapOverlapRatio", recoveryRaw["cubeMapOverlapRatio"]
            ),
            maxCoveredCells=_requireInt("recovery.maxCoveredCells", recoveryRaw["maxCoveredCells"]),
        ),
        backendTuning=BackendTuningConfig(
            acceptAnyCandidate=_requireBool(
                "backendTuning.acceptAnyCandidate", tuningRaw["acceptAnyCandidate"]
            ),
            directMode=_requireBool("backendTuning.directMode", tuningRaw["directMode"]),
            singleRound=_requireBool("backendTuning.singleRound", tuningRaw["singleRound"]),
            adaptiveViewCount=_requireBool(
                "backendTuning.adaptiveViewCount", tuningRaw["adaptiveViewCount"]
            ),
            singleView=_requireBool("backendTuning.singleView", tuningRaw["singleView"]),
            singleViewHorizontalFovCapRad=_optionalDegreesToRadians(
                "backendTuning.singleViewHorizontalFovCapDeg",
                tuningRaw["singleViewHorizontalFovCapDeg"],
            ),
            singleViewVerticalFovCapRad=_optionalDegreesToRadians(
                "backendTuning.singleViewVerticalFovCapDeg",
                tuningRaw["singleViewVerticalFovCapDeg"],
            ),
            fourViewFovCapRad=_optionalDegreesToRadians(
                "backendTuning.fourViewFovCapDeg", tuningRaw["fourViewFovCapDeg"]
            ),
            fullViewSearch=_requireBool(
                "backendTuning.fullViewSearch", tuningRaw["fullViewSearch"]
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
            allowSingleViewTemplate=_requireBool(
                "backendTuning.allowSingleViewTemplate", tuningRaw["allowSingleViewTemplate"]
            ),
            holdWeakBox=_requireBool("backendTuning.holdWeakBox", tuningRaw["holdWeakBox"]),
            fusionSourceMinConfidence=_optionalFloat(
                "backendTuning.fusionSourceMinConfidence",
                tuningRaw["fusionSourceMinConfidence"],
            ),
            fusionOverlap=_optionalFloat(
                "backendTuning.fusionOverlap", tuningRaw["fusionOverlap"]
            ),
            fusionBoxMode=(
                None
                if tuningRaw["fusionBoxMode"] is None
                else _requireStr("backendTuning.fusionBoxMode", tuningRaw["fusionBoxMode"])
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


def _optionalFloat(name: str, value: object) -> float | None:
    return None if value is None else _requireFloat(name, value)


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


def _requireFloatTuple(name: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{name} must be a non-empty list")
    result = tuple(_requireFloat(f"{name}[{index}]", item) for index, item in enumerate(value))
    return result


def _requireIntTuple(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{name} must be a non-empty list")
    result = tuple(_requireInt(f"{name}[{index}]", item) for index, item in enumerate(value))
    return result


def _degreesToRadians(name: str, valueDeg: float) -> float:
    if not 0.0 < valueDeg < 180.0:
        raise ConfigError(f"{name} must be in (0, 180), actual={valueDeg}")
    return valueDeg * pi / 180.0
