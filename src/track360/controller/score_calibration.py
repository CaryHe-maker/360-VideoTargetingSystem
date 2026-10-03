"""Versioned score-calibration artifacts bound to one inference checkpoint."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from track360.core.errors import ConfigError

CALIBRATION_FORMAT = "track360.score-calibration.v1"


@dataclass(frozen=True, slots=True)
class BetaCalibration:
    alpha: float
    beta: float
    intercept: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.alpha, self.beta, self.intercept)):
            raise ConfigError("appearance calibration parameters must be finite")
        if self.alpha <= 0.0 or self.beta <= 0.0:
            raise ConfigError("appearance beta calibration must be monotonic")


@dataclass(frozen=True, slots=True)
class ScoreCalibration:
    format: str
    checkpointSha256: str
    manifestSha256: str
    split: str
    appearanceInput: str
    appearance: BetaCalibration
    appearanceWeight: float
    motionWeight: float
    candidateMinScore: float
    fusionSourceMinConfidence: float

    def __post_init__(self) -> None:
        if self.format != CALIBRATION_FORMAT:
            raise ConfigError(f"unsupported score calibration format: {self.format}")
        for name, value in (
            ("appearanceWeight", self.appearanceWeight),
            ("motionWeight", self.motionWeight),
            ("candidateMinScore", self.candidateMinScore),
            ("fusionSourceMinConfidence", self.fusionSourceMinConfidence),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ConfigError(f"score calibration {name} must be in [0, 1]")
        if abs(self.appearanceWeight + self.motionWeight - 1.0) > 1e-9:
            raise ConfigError("score calibration weights must sum to one")
        if self.split != "calibration":
            raise ConfigError("score calibration artifact must be fitted on calibration split")
        if self.appearanceInput != "presence_quality_product":
            raise ConfigError(f"unsupported appearance input: {self.appearanceInput}")
        for name, value in (
            ("checkpointSha256", self.checkpointSha256),
            ("manifestSha256", self.manifestSha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ConfigError(f"score calibration {name} must be lowercase SHA-256")


UNCALIBRATED_STAGE3_SCORE_CALIBRATION = ScoreCalibration(
    format=CALIBRATION_FORMAT,
    checkpointSha256="0" * 64,
    manifestSha256="0" * 64,
    split="calibration",
    appearanceInput="presence_quality_product",
    appearance=BetaCalibration(1.0, 1.0, 0.0),
    appearanceWeight=1.0,
    motionWeight=0.0,
    candidateMinScore=0.50,
    fusionSourceMinConfidence=0.50,
)


def loadScoreCalibration(
    path: str | Path,
    *,
    checkpointPath: str | Path,
    candidateMinScore: float,
    fusionSourceMinConfidence: float,
    requireCheckpointHashMatch: bool = True,
) -> ScoreCalibration:
    artifactPath = Path(path).expanduser().resolve()
    try:
        raw = json.loads(artifactPath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(
            f"cannot read score calibration artifact {artifactPath}: {error}"
        ) from error
    root = _mapping("calibration", raw)
    _keys(
        "calibration",
        root,
        {
            "format",
            "checkpointSha256",
            "manifestSha256",
            "split",
            "appearanceInput",
            "appearance",
            "singleScore",
            "thresholds",
            "fit",
        },
    )
    appearance = _mapping("appearance", root["appearance"])
    _keys("appearance", appearance, {"method", "alpha", "beta", "intercept"})
    if appearance["method"] != "beta":
        raise ConfigError("appearance calibration method must be beta")
    singleScore = _mapping("singleScore", root["singleScore"])
    _keys("singleScore", singleScore, {"appearanceWeight", "motionWeight"})
    thresholds = _mapping("thresholds", root["thresholds"])
    _keys(
        "thresholds",
        thresholds,
        {"candidateMinScore", "fusionSourceMinConfidence"},
    )
    fit = _mapping("fit", root["fit"])
    _keys(
        "fit",
        fit,
        {
            "sampleCount",
            "positiveCount",
            "negativeCount",
            "sequenceCount",
            "rawBrier",
            "calibratedBrier",
            "rawEce",
            "calibratedEce",
            "prAuc",
            "rocAuc",
        },
    )
    for name, value in fit.items():
        _finiteNumber(f"fit.{name}", value)
    result = ScoreCalibration(
        format=_string("format", root["format"]),
        checkpointSha256=_string("checkpointSha256", root["checkpointSha256"]),
        manifestSha256=_string("manifestSha256", root["manifestSha256"]),
        split=_string("split", root["split"]),
        appearanceInput=_string("appearanceInput", root["appearanceInput"]),
        appearance=BetaCalibration(
            _finiteNumber("appearance.alpha", appearance["alpha"]),
            _finiteNumber("appearance.beta", appearance["beta"]),
            _finiteNumber("appearance.intercept", appearance["intercept"]),
        ),
        appearanceWeight=_finiteNumber(
            "singleScore.appearanceWeight", singleScore["appearanceWeight"]
        ),
        motionWeight=_finiteNumber("singleScore.motionWeight", singleScore["motionWeight"]),
        candidateMinScore=_finiteNumber(
            "thresholds.candidateMinScore", thresholds["candidateMinScore"]
        ),
        fusionSourceMinConfidence=_finiteNumber(
            "thresholds.fusionSourceMinConfidence",
            thresholds["fusionSourceMinConfidence"],
        ),
    )
    if requireCheckpointHashMatch:
        actualHash = sha256File(checkpointPath)
        if result.checkpointSha256 != actualHash:
            raise ConfigError(
                "score calibration checkpoint hash mismatch: "
                f"expected={result.checkpointSha256}, actual={actualHash}"
            )
    if abs(result.candidateMinScore - candidateMinScore) > 1e-9:
        raise ConfigError("tracking.candidateMinScore does not match calibration artifact")
    if abs(result.fusionSourceMinConfidence - fusionSourceMinConfidence) > 1e-9:
        raise ConfigError(
            "evaluator.fusionSourceMinConfidence does not match calibration artifact"
        )
    return result


def sha256File(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).expanduser().resolve().open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ConfigError(f"cannot hash file {path}: {error}") from error
    return digest.hexdigest()


def _mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{name} must be a mapping with string keys")
    return value


def _keys(name: str, value: dict[str, Any], expected: set[str]) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ConfigError(f"{name} fields invalid: missing={missing}, unknown={unknown}")


def _string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _finiteNumber(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ConfigError(f"{name} must be finite")
    return float(value)


__all__ = [
    "BetaCalibration",
    "CALIBRATION_FORMAT",
    "ScoreCalibration",
    "UNCALIBRATED_STAGE3_SCORE_CALIBRATION",
    "loadScoreCalibration",
    "sha256File",
]
