"""Result files in the layout the official 360VOT toolkit evaluates.

The toolkit takes one directory per representation, holding one sub-directory per
tracker with one ``<sequence>.txt`` per sequence and one line per frame::

    <root>/bbox/<tracker>/0001.txt     x1,y1,w,h                        (pixels)
    <root>/bfov/<tracker>/0001.txt     clon,clat,fov_h,fov_v,rotation   (degrees)

Box coordinates are pixel indices, half a pixel below this project's pixel-edge
coordinates.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import degrees
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from track360.core.errors import DecodeError, OutputError, ProtocolError
from track360.core.types import BBoxXYWH, BFoV, TrackResult

RESULT_PRECISION = 4
BBOX_DIRECTORY = "bbox"
BFOV_DIRECTORY = "bfov"


def formatBboxLine(bbox: BBoxXYWH, frameWidthPx: int) -> str:
    """Format an ERP box as ``x1,y1,w,h`` in the toolkit's pixel-index coordinates.

    A box that crosses the seam is written with a negative ``x1``.  The toolkit's
    dual success only compares a prediction against the ground truth and the ground
    truth shifted one image width to the left, so this is the one placement that
    scores against both ways the annotations write such a box (running past the
    left border or past the right one).
    """
    xPx = bbox.xPx - frameWidthPx if bbox.xPx + bbox.widthPx > frameWidthPx else bbox.xPx
    return _formatRow((xPx - 0.5, bbox.yPx - 0.5, bbox.widthPx, bbox.heightPx))


def formatBfovLine(bfov: BFoV) -> str:
    """Format a BFoV as ``clon,clat,fov_h,fov_v,rotation`` in degrees."""
    return _formatRow(
        (
            degrees(bfov.center.yawRad),
            degrees(bfov.center.pitchRad),
            degrees(bfov.horizontalFovRad),
            degrees(bfov.verticalFovRad),
            degrees(bfov.rollRad),
        )
    )


def resultPaths(root: str | Path, tracker: str, sequence: str) -> tuple[Path, Path]:
    """Return the ``(bbox, bfov)`` result paths of one sequence."""
    base = Path(root)
    name = f"{sequence}.txt"
    return base / BBOX_DIRECTORY / tracker / name, base / BFOV_DIRECTORY / tracker / name


def writeSequenceResults(
    root: str | Path,
    tracker: str,
    sequence: str,
    results: Sequence[TrackResult],
    frameWidthPx: int,
) -> tuple[Path, Path]:
    """Publish both result files of one sequence atomically."""
    if not results:
        raise OutputError(f"no results to write for sequence {sequence}")
    bboxPath, bfovPath = resultPaths(root, tracker, sequence)
    _writeLines(bboxPath, [formatBboxLine(item.bbox, frameWidthPx) for item in results])
    _writeLines(bfovPath, [formatBfovLine(item.bfov) for item in results])
    return bboxPath, bfovPath


def readResultFile(path: str | Path) -> NDArray[np.float64]:
    """Read a result file as the toolkit does: comma separated, else whitespace."""
    resultPath = Path(path)
    try:
        lines = resultPath.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DecodeError(f"cannot read result file {resultPath}: {error}") from error
    rows: list[list[float]] = []
    for number, line in enumerate(lines, start=1):
        parts = line.strip().split(",")
        if len(parts) < 2:
            parts = line.strip().split()
        try:
            rows.append([float(part) for part in parts])
        except ValueError as error:
            raise DecodeError(f"invalid result line {resultPath}:{number}: {line!r}") from error
    if not rows or len({len(row) for row in rows}) != 1:
        raise DecodeError(f"result file is empty or has ragged rows: {resultPath}")
    return np.asarray(rows, dtype=np.float64)


@dataclass(slots=True)
class ResultCollector:
    """An in-memory result sink for runs that publish several formats afterwards."""

    results: list[TrackResult] = field(default_factory=list)

    def open(self, destination: str) -> None:
        del destination
        self.results.clear()

    def write(self, result: TrackResult) -> None:
        if int(result.frameIndex) != len(self.results):
            raise OutputError(
                f"result frame order mismatch: expected={len(self.results)}, "
                f"actual={int(result.frameIndex)}"
            )
        self.results.append(result)

    def finalize(self, expectedFrameCount: int) -> None:
        if expectedFrameCount != len(self.results):
            raise OutputError(
                f"result count mismatch: expected={expectedFrameCount}, "
                f"actual={len(self.results)}"
            )


def _formatRow(values: Sequence[float]) -> str:
    if not np.isfinite(values).all():
        raise ProtocolError(f"result values must be finite: {values}")
    return ",".join(f"{value:.{RESULT_PRECISION}f}" for value in values)


def _writeLines(path: Path, lines: Sequence[str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        with partial.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    except OSError as error:
        raise OutputError(f"cannot write result file {path}: {error}") from error


__all__ = [
    "BBOX_DIRECTORY",
    "BFOV_DIRECTORY",
    "RESULT_PRECISION",
    "ResultCollector",
    "formatBboxLine",
    "formatBfovLine",
    "readResultFile",
    "resultPaths",
    "writeSequenceResults",
]
