"""Pure RGB annotation helpers for visualization output."""

from __future__ import annotations

from math import ceil, floor

import numpy as np
from numpy.typing import NDArray

from track360.core.errors import ProtocolError
from track360.core.types import BBoxXYWH

FLUORESCENT_GREEN_RGB = (57, 255, 20)
BOX_THICKNESS_PX = 2
LABEL_GAP_PX = 2

_GLYPHS = {
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "=": ("00000", "00000", "11111", "00000", "11111", "00000", "00000"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "N": ("10001", "11001", "11001", "10101", "10011", "10011", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "a": ("00000", "00000", "01110", "00001", "01111", "10001", "01111"),
    "c": ("00000", "00000", "01111", "10000", "10000", "10000", "01111"),
    "d": ("00001", "00001", "01101", "10011", "10001", "10011", "01101"),
    "e": ("00000", "00000", "01110", "10001", "11111", "10000", "01110"),
    "f": ("00111", "01000", "01000", "11110", "01000", "01000", "01000"),
    "n": ("00000", "00000", "10110", "11001", "10001", "10001", "10001"),
    "o": ("00000", "00000", "01110", "10001", "10001", "10001", "01110"),
    "r": ("00000", "00000", "10110", "11001", "10000", "10000", "10000"),
    "s": ("00000", "00000", "01111", "10000", "01110", "00001", "11110"),
    "t": ("01000", "01000", "11110", "01000", "01000", "01001", "00110"),
    "u": ("00000", "00000", "10001", "10001", "10001", "10011", "01101"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
}


def drawBoxRgb(
    rgb: NDArray[np.uint8],
    bbox: BBoxXYWH,
    *,
    wrapHorizontal: bool = False,
    label: str | None = None,
) -> NDArray[np.uint8]:
    """Return an RGB copy with one fluorescent-green XYWH box and optional label."""
    _requireRgb(rgb)
    heightPx, widthPx = rgb.shape[:2]
    output = rgb.copy()
    y0 = max(0, min(heightPx - 1, floor(bbox.yPx)))
    y1 = max(0, min(heightPx - 1, ceil(bbox.yPx + bbox.heightPx) - 1))
    if y1 < y0:
        raise ProtocolError(f"visualization bbox has no visible vertical area: {bbox}")

    if wrapHorizontal:
        _drawWrappedBox(output, bbox, y0, y1)
    else:
        _drawClippedBox(output, bbox, y0, y1)
    if label is not None:
        _drawLabel(output, bbox, label, y0, y1, wrapHorizontal=wrapHorizontal)
    return output


def _drawLabel(
    output: NDArray[np.uint8],
    bbox: BBoxXYWH,
    label: str,
    y0: int,
    y1: int,
    *,
    wrapHorizontal: bool,
) -> None:
    if not label or any(character not in _GLYPHS for character in label):
        raise ProtocolError(f"visualization label contains unsupported characters: {label!r}")
    heightPx, widthPx = output.shape[:2]
    scale = 2 if heightPx >= 128 and widthPx >= 160 else 1
    glyphHeight = 7 * scale
    advance = 6 * scale
    labelWidth = len(label) * advance - scale
    startX = floor(bbox.xPx) % widthPx if wrapHorizontal else max(0, floor(bbox.xPx))
    if not wrapHorizontal:
        startX = min(startX, max(0, widthPx - labelWidth))
    labelY = y1 + LABEL_GAP_PX + 1
    if labelY + glyphHeight > heightPx:
        labelY = max(0, y0 - LABEL_GAP_PX - glyphHeight)

    color = np.asarray(FLUORESCENT_GREEN_RGB, dtype=np.uint8)
    for characterIndex, character in enumerate(label):
        glyphX = startX + characterIndex * advance
        for rowIndex, row in enumerate(_GLYPHS[character]):
            for columnIndex, enabled in enumerate(row):
                if enabled == "0":
                    continue
                yStart = labelY + rowIndex * scale
                xStart = glyphX + columnIndex * scale
                for y in range(yStart, min(yStart + scale, heightPx)):
                    if wrapHorizontal:
                        columns = np.mod(np.arange(xStart, xStart + scale), widthPx)
                        output[y, columns] = color
                    elif xStart < widthPx:
                        output[y, xStart : min(xStart + scale, widthPx)] = color


def _drawClippedBox(
    output: NDArray[np.uint8],
    bbox: BBoxXYWH,
    y0: int,
    y1: int,
) -> None:
    widthPx = output.shape[1]
    x0 = max(0, min(widthPx - 1, floor(bbox.xPx)))
    x1 = max(0, min(widthPx - 1, ceil(bbox.xPx + bbox.widthPx) - 1))
    if x1 < x0:
        raise ProtocolError(f"visualization bbox has no visible horizontal area: {bbox}")
    _drawEdges(output, np.arange(x0, x1 + 1), x0, x1, y0, y1)


def _drawWrappedBox(
    output: NDArray[np.uint8],
    bbox: BBoxXYWH,
    y0: int,
    y1: int,
) -> None:
    widthPx = output.shape[1]
    if bbox.widthPx > widthPx:
        raise ProtocolError("wrapped visualization bbox cannot be wider than the RGB frame")
    startX = floor(bbox.xPx)
    endX = ceil(bbox.xPx + bbox.widthPx) - 1
    horizontalX = np.mod(np.arange(startX, endX + 1), widthPx)
    _drawEdges(output, horizontalX, startX % widthPx, endX % widthPx, y0, y1)


def _drawEdges(
    output: NDArray[np.uint8],
    horizontalX: NDArray[np.int64],
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> None:
    heightPx, widthPx = output.shape[:2]
    color = np.asarray(FLUORESCENT_GREEN_RGB, dtype=np.uint8)
    for offsetPx in range(BOX_THICKNESS_PX):
        topY = min(y0 + offsetPx, heightPx - 1)
        bottomY = max(y1 - offsetPx, 0)
        leftX = (x0 + offsetPx) % widthPx
        rightX = (x1 - offsetPx) % widthPx
        output[topY, horizontalX] = color
        output[bottomY, horizontalX] = color
        output[y0 : y1 + 1, leftX] = color
        output[y0 : y1 + 1, rightX] = color


def _requireRgb(rgb: NDArray[np.uint8]) -> None:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ProtocolError("visualization RGB must be a uint8 NumPy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0:
        raise ProtocolError(
            f"visualization RGB must have non-empty shape [H, W, 3], actual={rgb.shape}"
        )


__all__ = ["FLUORESCENT_GREEN_RGB", "drawBoxRgb"]
