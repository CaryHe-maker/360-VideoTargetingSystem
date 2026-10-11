"""Seeding and run metadata so a result file can be traced back to what produced it."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from track360.core.config import AppConfig, ReproducibilityConfig

RUN_METADATA_SUFFIX = ".run.json"
_PACKAGES = ("track360", "torch", "torchvision", "numpy", "opencv-python-headless", "timm")
# Machine-specific locations: recorded in the run metadata, excluded from the hash.
_PATH_FIELDS = frozenset(
    {"sourcePath", "model.weights", "visualization.outputRoot"}
)


def sha256File(path: str | Path) -> str:
    """SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seedEverything(config: ReproducibilityConfig) -> None:
    """Seed every random source and pin cuDNN to deterministic kernels."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = config.deterministic
    torch.backends.cudnn.benchmark = not config.deterministic


def configSnapshot(config: AppConfig) -> dict[str, Any]:
    """Return the effective configuration as plain JSON-compatible values."""
    snapshot = _plain(config)
    assert isinstance(snapshot, dict)
    return snapshot


def configHash(config: AppConfig) -> str:
    """Hash the parameters that affect results, ignoring machine-specific paths."""
    snapshot = _plain(config, skip=_PATH_FIELDS)
    text = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collectRunMetadata(config: AppConfig) -> dict[str, Any]:
    weights = Path(config.model.weights)
    return {
        "createdAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": _gitState(),
        "configHash": configHash(config),
        "config": configSnapshot(config),
        "weightsSha256": sha256File(weights) if weights.is_file() else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _packageVersion(name) for name in _PACKAGES},
        "device": _deviceState(),
    }


def writeRunMetadata(resultPath: str | Path, config: AppConfig) -> Path:
    """Write ``<result>.run.json`` next to a published result file."""
    destination = Path(resultPath).expanduser().resolve()
    metadataPath = destination.with_name(destination.name + RUN_METADATA_SUFFIX)
    payload = json.dumps(collectRunMetadata(config), indent=2, ensure_ascii=False)
    metadataPath.write_text(payload + "\n", encoding="utf-8")
    return metadataPath


def _plain(value: Any, *, skip: frozenset[str] = frozenset(), prefix: str = "") -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, Any] = {}
        for field in fields(value):
            name = f"{prefix}{field.name}"
            if name not in skip:
                item = getattr(value, field.name)
                result[field.name] = _plain(item, skip=skip, prefix=f"{name}.")
        return result
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item, skip=skip, prefix=prefix) for item in value]
    return value


def _gitState() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _git(cwd: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _packageVersion(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _deviceState() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"cuda": False}
    if not torch.cuda.is_available():
        return {"cuda": False}
    return {
        "cuda": True,
        "gpu": torch.cuda.get_device_name(0),
        "driver": _nvidiaDriverVersion(),
        "cudaRuntime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnnDeterministic": bool(torch.backends.cudnn.deterministic),
    }


def _nvidiaDriverVersion() -> str | None:
    try:
        completed = subprocess.run(
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = completed.stdout.strip().splitlines()
    return lines[0].strip() if completed.returncode == 0 and lines else None


__all__ = [
    "RUN_METADATA_SUFFIX",
    "collectRunMetadata",
    "configHash",
    "configSnapshot",
    "seedEverything",
    "writeRunMetadata",
]
