"""360VOT benchmark reader.

A dataset root holds one entry per sequence, either an extracted directory or the
zip archive it is distributed as::

    <root>/0001/image/000000.jpg ...      <root>/0001.zip
    <root>/0001/label.json                  (same layout inside the archive)

The 360VOS training archives hold ``image/`` and ``mask/`` directly, without the
sequence directory and without ``label.json``.  They are read the same way; their
labels come from a separate label root (see ``datasets/mask_labels.py``).

``label.json`` maps each frame file name to four annotations.  Angles are in
degrees and follow the official toolkit (https://github.com/HuajianUP/360VOT):
``clon`` grows to the right and is 0 at the image center, ``clat`` grows upward,
which is this project's yaw / pitch.  Box centers are pixel indices, so pixel
``u`` covers ``[u, u + 1)`` in this project's pixel-edge coordinates.  A zero-sized
annotation marks a frame where the target is absent.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from math import radians
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from track360.core.errors import DecodeError, ProtocolError
from track360.core.types import BBoxXYWH, BFoV, FrameIndex, FramePacket, SequenceId
from track360.geometry.projection_math import makeSphericalPoint
from track360.io.image_reader import decodeRgbImage

LABEL_FILE = "label.json"
IMAGE_DIRECTORY = "image"
# Nominal capture rate; 360VOT frames carry no timestamps of their own.
FRAME_INTERVAL_NS = 33_333_333


class Vot360Representation(StrEnum):
    BBOX = "bbox"
    RBBOX = "rbbox"
    BFOV = "bfov"
    RBFOV = "rbfov"


@dataclass(frozen=True, slots=True)
class Vot360Annotation:
    """One frame's ground truth in project types; ``None`` means the target is absent."""

    frameIndex: int
    bfov: BFoV | None
    rbfov: BFoV | None
    bbox: BBoxXYWH | None

    @property
    def present(self) -> bool:
        return self.bfov is not None


class Vot360Sequence:
    """Frames and annotations of one sequence, read lazily from a directory or zip."""

    def __init__(self, location: Path, labelPath: Path | None = None) -> None:
        self._location = location
        self._labelPath = labelPath
        self._isArchive = location.is_file()
        self.name = location.stem if self._isArchive else location.name
        self._archive: zipfile.ZipFile | None = None
        self._memberPrefix: str | None = None
        self._labels: list[dict[str, Any]] | None = None
        self._frameNames: tuple[str, ...] = ()
        self._frameSize: tuple[int, int] | None = None

    @property
    def location(self) -> Path:
        return self._location

    @property
    def hasLabels(self) -> bool:
        if self._labelPath is not None:
            return self._labelPath.is_file()
        return self.hasMember(LABEL_FILE)

    @property
    def frameNames(self) -> tuple[str, ...]:
        if not self._frameNames:
            if self.hasLabels:
                self._loadLabels()
            else:
                self._frameNames = self._listImages()
        return self._frameNames

    @property
    def frameCount(self) -> int:
        return len(self.frameNames)

    @property
    def frameSize(self) -> tuple[int, int]:
        """Frame ``(width, height)`` in pixels, taken from the first image."""
        if self._frameSize is None:
            self.readRgb(0)
        assert self._frameSize is not None
        return self._frameSize

    def readRgb(self, frameIndex: int) -> NDArray[np.uint8]:
        name = self._frameName(frameIndex)
        member = f"{IMAGE_DIRECTORY}/{name}"
        rgb = decodeRgbImage(self.readMember(member), f"{self.name}/{member}")
        size = (int(rgb.shape[1]), int(rgb.shape[0]))
        if self._frameSize is None:
            self._frameSize = size
        elif size != self._frameSize:
            raise DecodeError(
                f"360VOT frame size changed within {self.name}: "
                f"expected={self._frameSize}, actual={size}, frame={name}"
            )
        return rgb

    def annotation(self, frameIndex: int) -> Vot360Annotation:
        self._frameName(frameIndex)
        self._loadLabels()
        assert self._labels is not None
        label = self._labels[frameIndex]
        return Vot360Annotation(
            frameIndex=frameIndex,
            bfov=_toBfov(label[Vot360Representation.BFOV]),
            rbfov=_toBfov(label[Vot360Representation.RBFOV]),
            bbox=_toErpBox(label[Vot360Representation.BBOX], *self.frameSize),
        )

    def initialBfov(self) -> BFoV:
        bfov = self.annotation(0).bfov
        if bfov is None:
            raise DecodeError(f"360VOT sequence {self.name} has no target in frame 0")
        return bfov

    def initialBbox(self) -> BBoxXYWH:
        bbox = self.annotation(0).bbox
        if bbox is None:
            raise DecodeError(f"360VOT sequence {self.name} has no target in frame 0")
        return bbox

    def groundTruth(
        self, representation: Vot360Representation | str
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Return ground truth rows in the official toolkit layout and a presence mask.

        ``bbox`` rows are ``[x1, y1, w, h]``, ``rbbox`` rows ``[cx, cy, w, h, rotation]``
        and ``bfov`` / ``rbfov`` rows ``[clon, clat, fov_h, fov_v, rotation]``, exactly
        as the toolkit's evaluation script builds them.
        """
        kind = Vot360Representation(representation)
        self._loadLabels()
        assert self._labels is not None
        rows: list[tuple[float, ...]] = []
        present: list[bool] = []
        for label in self._labels:
            item = label[kind]
            if kind in (Vot360Representation.BFOV, Vot360Representation.RBFOV):
                row = (item["clon"], item["clat"], item["fov_h"], item["fov_v"], item["rotation"])
                present.append(item["fov_h"] != 0 and item["fov_v"] != 0)
            else:
                present.append(item["w"] != 0 and item["h"] != 0)
                if kind is Vot360Representation.BBOX:
                    left, top = _rotatedTopLeft(item)
                    row = (left, top, item["w"], item["h"])
                else:
                    row = (item["cx"], item["cy"], item["w"], item["h"], item["rotation"])
            rows.append(tuple(float(value) for value in row))
        return np.asarray(rows, dtype=np.float64), np.asarray(present, dtype=np.bool_)

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def _frameName(self, frameIndex: int) -> str:
        names = self.frameNames
        if not 0 <= frameIndex < len(names):
            raise ProtocolError(
                f"360VOT frame index out of range: sequence={self.name}, "
                f"index={frameIndex}, frames={len(names)}"
            )
        return names[frameIndex]

    def readMember(self, member: str) -> bytes:
        """Read one file of the sequence, for example ``image/000000.jpg``."""
        try:
            if self._isArchive:
                return self._openArchive().read(self._prefix() + member)
            return (self._location / member).read_bytes()
        except (OSError, KeyError, zipfile.BadZipFile) as error:
            raise DecodeError(
                f"cannot read {member} from 360VOT sequence {self._location}: {error}"
            ) from error

    def hasMember(self, member: str) -> bool:
        if self._isArchive:
            try:
                self._openArchive().getinfo(self._prefix() + member)
            except KeyError:
                return False
            return True
        return (self._location / member).is_file()

    def _openArchive(self) -> zipfile.ZipFile:
        if self._archive is None:
            try:
                self._archive = zipfile.ZipFile(self._location)
            except (OSError, zipfile.BadZipFile) as error:
                raise DecodeError(
                    f"cannot open 360VOT archive {self._location}: {error}"
                ) from error
        return self._archive

    def _prefix(self) -> str:
        """Archives either wrap everything in ``<name>/`` or hold ``image/`` directly."""
        if self._memberPrefix is None:
            wrapped = f"{self.name}/"
            names = self._openArchive().namelist()
            self._memberPrefix = wrapped if any(n.startswith(wrapped) for n in names) else ""
        return self._memberPrefix

    def _listImages(self) -> tuple[str, ...]:
        folder = f"{IMAGE_DIRECTORY}/"
        if self._isArchive:
            start = self._prefix() + folder
            names = [
                name[len(start) :]
                for name in self._openArchive().namelist()
                if name.startswith(start) and not name.endswith("/")
            ]
        else:
            directory = self._location / IMAGE_DIRECTORY
            names = [entry.name for entry in directory.iterdir()] if directory.is_dir() else []
        if not names:
            raise DecodeError(f"no images in 360VOT sequence {self._location}")
        return tuple(sorted(names))

    def _loadLabels(self) -> None:
        if self._labels is not None:
            return
        try:
            if self._labelPath is not None:
                payload = json.loads(self._labelPath.read_bytes())
            else:
                payload = json.loads(self.readMember(LABEL_FILE))
        except OSError as error:
            raise DecodeError(f"cannot read 360VOT labels {self._labelPath}: {error}") from error
        except json.JSONDecodeError as error:
            raise DecodeError(f"invalid 360VOT label file in {self._location}: {error}") from error
        if not isinstance(payload, dict) or not payload:
            raise DecodeError(f"360VOT label file is empty or not a mapping: {self._location}")
        names = tuple(payload)
        if names != tuple(sorted(names)):
            raise DecodeError(f"360VOT label frames are not in order: {self._location}")
        labels = list(payload.values())
        for name, label in zip(names, labels, strict=True):
            _requireLabel(self.name, name, label)
        self._frameNames = names
        self._labels = labels


class Vot360Dataset:
    """All sequences under a 360VOT root, addressed by name in sorted order."""

    def __init__(self, root: str | Path, labelRoot: str | Path | None = None) -> None:
        """``labelRoot`` holds ``<sequence>/label.json`` for sequences shipped without one."""
        self.root = Path(root).expanduser().resolve()
        self.labelRoot = None if labelRoot is None else Path(labelRoot).expanduser().resolve()
        if not self.root.is_dir():
            raise DecodeError(f"360VOT dataset root is not a directory: {self.root}")
        locations: dict[str, Path] = {}
        for entry in sorted(self.root.iterdir()):
            if entry.is_file() and entry.suffix.lower() == ".zip":
                # An extracted directory wins over the archive it came from.
                locations.setdefault(entry.stem, entry)
            elif entry.is_dir() and (
                (entry / LABEL_FILE).is_file() or (entry / IMAGE_DIRECTORY).is_dir()
            ):
                locations[entry.name] = entry
        if not locations:
            raise DecodeError(f"no 360VOT sequences found in {self.root}")
        self._locations = dict(sorted(locations.items()))

    @property
    def sequenceNames(self) -> tuple[str, ...]:
        return tuple(self._locations)

    def __len__(self) -> int:
        return len(self._locations)

    def __iter__(self) -> Iterator[Vot360Sequence]:
        return (self.sequence(name) for name in self._locations)

    def sequence(self, name: str) -> Vot360Sequence:
        try:
            location = self._locations[name]
        except KeyError as error:
            raise DecodeError(f"unknown 360VOT sequence '{name}' in {self.root}") from error
        labelPath = None if self.labelRoot is None else self.labelRoot / name / LABEL_FILE
        return Vot360Sequence(location, labelPath)


@dataclass(slots=True)
class Vot360DataSource:
    """Read one 360VOT sequence through the common frame contract."""

    maxFrames: int | None = None
    labelRoot: str | Path | None = None
    _sequence: Vot360Sequence | None = field(init=False, default=None, repr=False)
    _cursor: int = field(init=False, default=0, repr=False)

    @property
    def sequence(self) -> Vot360Sequence:
        if self._sequence is None:
            raise ProtocolError("360VOT data source is not open")
        return self._sequence

    @property
    def frameCount(self) -> int:
        if self._sequence is None:
            return 0
        count = self._sequence.frameCount
        return count if self.maxFrames is None else min(count, self.maxFrames)

    def open(self, root: str, sequenceId: str | None = None) -> None:
        if self.maxFrames is not None and self.maxFrames <= 0:
            raise ProtocolError("maxFrames must be positive")
        self.close()
        dataset = Vot360Dataset(root, self.labelRoot)
        if sequenceId is None:
            if len(dataset) != 1:
                raise DecodeError(
                    f"360VOT root holds {len(dataset)} sequences; a sequence name is required"
                )
            sequenceId = dataset.sequenceNames[0]
        self._sequence = dataset.sequence(sequenceId)
        self._cursor = 0

    def read(self) -> FramePacket | None:
        sequence = self.sequence
        if self._cursor >= self.frameCount:
            return None
        frame = FramePacket(
            sequenceId=SequenceId(sequence.name),
            frameIndex=FrameIndex(self._cursor),
            timestampNs=self._cursor * FRAME_INTERVAL_NS,
            rgb=sequence.readRgb(self._cursor),
        )
        self._cursor += 1
        return frame

    def close(self) -> None:
        if self._sequence is not None:
            self._sequence.close()
        self._sequence = None
        self._cursor = 0


def _requireLabel(sequence: str, frameName: str, label: object) -> None:
    if not isinstance(label, dict):
        raise DecodeError(f"360VOT label is not a mapping: {sequence}/{frameName}")
    for kind in Vot360Representation:
        item = label.get(kind)
        keys = (
            ("clon", "clat", "fov_h", "fov_v", "rotation")
            if kind in (Vot360Representation.BFOV, Vot360Representation.RBFOV)
            else ("cx", "cy", "w", "h", "rotation")
        )
        if not isinstance(item, dict) or any(
            isinstance(item.get(key), bool) or not isinstance(item.get(key), (int, float))
            for key in keys
        ):
            raise DecodeError(f"360VOT label lacks a valid '{kind}': {sequence}/{frameName}")


def _toBfov(item: dict[str, Any]) -> BFoV | None:
    if item["fov_h"] == 0 or item["fov_v"] == 0:
        return None
    return BFoV(
        center=makeSphericalPoint(radians(item["clon"]), radians(item["clat"])),
        horizontalFovRad=radians(item["fov_h"]),
        verticalFovRad=radians(item["fov_v"]),
        rollRad=radians(item["rotation"]),
    )


def _toErpBox(item: dict[str, Any], frameWidthPx: int, frameHeightPx: int) -> BBoxXYWH | None:
    """Convert an axis-aligned label box to a seam-aware pixel-edge ERP box."""
    if item["w"] == 0 or item["h"] == 0:
        return None
    widthPx = min(float(item["w"]), float(frameWidthPx))
    heightPx = min(float(item["h"]), float(frameHeightPx))
    # Labels extend past either image border when the target crosses the seam; the
    # project convention keeps x inside the frame and lets x + width run past it.
    xPx = (float(item["cx"]) + 0.5 - 0.5 * widthPx) % frameWidthPx
    yPx = float(item["cy"]) + 0.5 - 0.5 * heightPx
    yPx = min(max(yPx, 0.0), frameHeightPx - heightPx)
    return BBoxXYWH(xPx=xPx, yPx=yPx, widthPx=widthPx, heightPx=heightPx)


def _rotatedTopLeft(item: dict[str, Any]) -> tuple[float, float]:
    """Top-left corner as the toolkit's ``Bbox`` computes it, rotation included."""
    rotation = radians(item["rotation"])
    halfWidth = item["w"] / 2.0
    halfHeight = item["h"] / 2.0
    left = item["cx"] - halfWidth * np.cos(rotation) + halfHeight * np.sin(rotation)
    top = item["cy"] - halfHeight * np.cos(rotation) - halfWidth * np.sin(rotation)
    return float(left), float(top)


__all__ = [
    "FRAME_INTERVAL_NS",
    "Vot360Annotation",
    "Vot360DataSource",
    "Vot360Dataset",
    "Vot360Representation",
    "Vot360Sequence",
]
