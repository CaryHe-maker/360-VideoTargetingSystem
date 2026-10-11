"""AirSim360 sequence reader built on the common frame contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from track360.core.errors import DecodeError, ProtocolError
from track360.core.protocols import AirSim360DataSource as AirSim360DataSourceProtocol
from track360.core.types import (
    AirSim360Record,
    FrameIndex,
    FramePacket,
    SegmentationPlane,
    SequenceId,
)
from track360.io.image_reader import readImageArray, readRgbImage


@dataclass(slots=True)
class AirSim360SequenceSource(AirSim360DataSourceProtocol):
    """Read one AirSim360 RGB sequence with optional aligned masks."""

    rgbFolders: tuple[str, ...] = ("rgb", "raw")
    semanticFolders: tuple[str, ...] = ("semantic", "segmentation")
    instanceFolders: tuple[str, ...] = ("instance", "instances")
    classListNames: tuple[str, ...] = ("semantic_lists.txt", "semantic_list.txt", "classes.txt")
    maxFrames: int | None = None
    _root: Path | None = field(init=False, default=None, repr=False)
    _records: tuple[AirSim360Record, ...] = field(init=False, default=(), repr=False)
    _classNames: dict[int, str] = field(init=False, default_factory=dict, repr=False)
    _cursor: int = field(init=False, default=0, repr=False)
    _sequenceId: str = field(init=False, default="", repr=False)

    @property
    def frameCount(self) -> int:
        return len(self._records)

    def open(self, root: str, sequenceId: str | None = None) -> None:
        requestedRoot = Path(root).expanduser().resolve()
        sequenceRoot = requestedRoot / sequenceId if sequenceId else requestedRoot
        if sequenceId is None and not _looksLikeDatasetRoot(sequenceRoot):
            candidates = [
                p for p in requestedRoot.iterdir() if p.is_dir() and _looksLikeDatasetRoot(p)
            ]
            if len(candidates) == 1:
                sequenceRoot = candidates[0]
                sequenceId = sequenceRoot.name
        sequenceId = sequenceId or sequenceRoot.name or "sequence"
        if not sequenceRoot.exists():
            raise DecodeError(f"AirSim360 sequence not found: {sequenceRoot}")
        self.close()
        self._root = sequenceRoot
        self._sequenceId = sequenceId
        meta = self._loadMeta(sequenceRoot / "meta.json")
        self._classNames = meta.get("classNames", {})
        if not self._classNames:
            self._classNames = _readClassList(sequenceRoot, self.classListNames)
        records = self._buildRecords(sequenceRoot, meta)
        if self.maxFrames is not None:
            if self.maxFrames <= 0:
                raise DecodeError("maxFrames must be positive")
            records = records[: self.maxFrames]
        self._records = tuple(records)
        if not self._records:
            raise DecodeError(f"no AirSim360 RGB frames found in {sequenceRoot}")
        self._cursor = 0

    def read(self) -> FramePacket | None:
        if self._root is None:
            raise ProtocolError("AirSim360 source is not open")
        if self._cursor >= len(self._records):
            return None
        record = self._records[self._cursor]
        rgb = readRgbImage(record.rgbPath)
        segmentation = _readSegmentation(
            record.semanticPath,
            record.instancePath,
            self._classNames,
        )
        frame = FramePacket(
            sequenceId=SequenceId(record.sequenceId),
            frameIndex=record.frameIndex,
            timestampNs=int(self._cursor * 33_333_333),
            rgb=rgb,
            segmentation=segmentation,
        )
        self._cursor += 1
        return frame

    def close(self) -> None:
        self._root = None
        self._records = ()
        self._classNames = {}
        self._cursor = 0
        self._sequenceId = ""

    def _loadMeta(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as stream:
                raw = json.load(stream)
        except OSError as error:
            raise DecodeError(f"cannot read AirSim360 meta file {path}: {error}") from error
        except json.JSONDecodeError as error:
            raise DecodeError(f"invalid AirSim360 meta file {path}: {error}") from error
        if not isinstance(raw, dict):
            raise DecodeError(f"AirSim360 meta file must contain an object: {path}")
        classNames = raw.get("classNames", {})
        if isinstance(classNames, dict):
            classNames = {
                int(key): str(value)
                for key, value in classNames.items()
                if not isinstance(key, bool)
            }
        else:
            classNames = {}
        raw["classNames"] = classNames
        return raw

    def _buildRecords(
        self,
        sequenceRoot: Path,
        meta: dict[str, Any],
    ) -> list[AirSim360Record]:
        if "records" in meta and isinstance(meta["records"], list):
            return [
                AirSim360Record(
                    sequenceId=SequenceId(str(item.get("sequenceId", self._sequenceId))),
                    frameIndex=FrameIndex(int(item["frameIndex"])),
                    rgbPath=str(_resolvePath(sequenceRoot, item.get("rgbPath"))),
                    semanticPath=_optionalPath(sequenceRoot, item.get("semanticPath")),
                    instancePath=_optionalPath(sequenceRoot, item.get("instancePath")),
                )
                for item in meta["records"]
                if isinstance(item, dict) and "rgbPath" in item and "frameIndex" in item
            ]
        rgbPaths = _collectFrameFiles(_findFolder(sequenceRoot, self.rgbFolders), sequenceRoot)
        semanticPaths = _indexPaths(_findFolder(sequenceRoot, self.semanticFolders), sequenceRoot)
        instancePaths = _indexPaths(_findFolder(sequenceRoot, self.instanceFolders), sequenceRoot)
        records: list[AirSim360Record] = []
        for index, rgbPath in enumerate(rgbPaths):
            stem = rgbPath.stem
            frameKey = _frameKey(stem)
            records.append(
                AirSim360Record(
                    sequenceId=SequenceId(self._sequenceId),
                    frameIndex=FrameIndex(index),
                    rgbPath=str(rgbPath),
                    semanticPath=_pickIndexedPath(semanticPaths, stem, frameKey),
                    instancePath=_pickIndexedPath(instancePaths, stem, frameKey),
                )
            )
        return records


def _collectFrameFiles(folder: Path, fallbackRoot: Path) -> list[Path]:
    if not folder.exists():
        folder = fallbackRoot
    iterator = folder.iterdir()
    return [
        path
        for path in sorted(iterator, key=_frameSortKey)
        if path.is_file()
        and path.suffix.lower()
        in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
    ]


def _frameSortKey(path: Path) -> tuple[bool, int, str]:
    key = _frameKey(path.stem)
    return key is None, int(key or 0), path.name.lower()


def _indexPaths(folder: Path, fallbackRoot: Path) -> dict[str, Path]:
    if not folder.exists():
        folder = fallbackRoot
    result: dict[str, Path] = {}
    for path in sorted(folder.iterdir(), key=lambda value: value.name.lower()):
        if path.is_file():
            result[path.stem] = path
            key = _frameKey(path.stem)
            if key is not None:
                result.setdefault(key, path)
    return result


def _pickIndexedPath(index: dict[str, Path], stem: str, frameKey: str | None = None) -> str | None:
    if stem in index:
        return str(index[stem])
    if frameKey is not None and frameKey in index:
        return str(index[frameKey])
    return None


def _findFolder(root: Path, aliases: tuple[str, ...]) -> Path:
    for name in aliases:
        folder = root / name
        if folder.is_dir():
            return folder
    return root / aliases[0]


def _looksLikeDatasetRoot(root: Path) -> bool:
    if not root.is_dir():
        return False
    names = {p.name.lower() for p in root.iterdir() if p.is_dir()}
    return bool(names & {"rgb", "raw"})


def _frameKey(stem: str) -> str | None:
    match = re.search(r"(\d+)$", stem)
    return str(int(match.group(1))) if match else None


def _readClassList(root: Path, names: tuple[str, ...]) -> dict[int, str]:
    path = next((root / name for name in names if (root / name).is_file()), None)
    if path is None:
        return {}
    result: dict[int, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DecodeError(f"cannot read semantic class list {path}: {error}") from error
    for line in lines:
        fields = line.strip().split()
        if len(fields) < 2:
            continue
        if fields[0].lstrip("+-").isdigit():
            result[int(fields[0])] = " ".join(fields[1:])
        elif fields[-1].lstrip("+-").isdigit():
            result[int(fields[-1])] = " ".join(fields[:-1])
    return result


def _resolvePath(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise DecodeError("AirSim360 meta path is missing")
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _optionalPath(root: Path, value: Any) -> str | None:
    if value in {None, ""}:
        return None
    return str(_resolvePath(root, value))


def _readSegmentation(
    semanticPath: str | None,
    instancePath: str | None,
    classNames: dict[int, str],
) -> SegmentationPlane | None:
    if semanticPath is None and instancePath is None:
        return None
    semantic = _readMask(semanticPath, role="semantic")
    instance = _readMask(instancePath, role="instance")
    return SegmentationPlane(semantic=semantic, instance=instance, classNames=dict(classNames))


def _readMask(pathText: str | None, *, role: str = "semantic") -> np.ndarray | None:
    if pathText is None:
        return None
    path = Path(pathText)
    raw = readImageArray(path)
    # AirSim360 masks are commonly RGBA PNGs: semantic IDs are in alpha and
    # instance IDs are packed from the RGB colour. Scalar NPY/PNG masks keep
    # their original values for compatibility with the legacy layout.
    if raw.ndim == 3 and raw.shape[2] == 4:
        if role == "semantic":
            raw = raw[..., 3]
        else:
            rgb = raw[..., :3].astype(np.int64)
            raw = rgb[..., 0] | (rgb[..., 1] << 8) | (rgb[..., 2] << 16)
    mask = raw.astype(np.int32, copy=False)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise DecodeError(f"AirSim360 mask must have shape [H, W]: {path}")
    return mask


AirSim360DataSource = AirSim360SequenceSource

__all__ = ["AirSim360DataSource", "AirSim360SequenceSource"]
