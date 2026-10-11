"""Result formatting helpers for deterministic text output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from track360.core.errors import OutputError
from track360.core.types import BBoxXYWH, TrackResult

RESULT_PRECISION = 6


def formatResultLine(result: TrackResult) -> str:
    """Format one track result as the default development text line."""
    bbox = result.bbox
    return _formatBox(bbox)


def _formatBox(bbox: BBoxXYWH) -> str:
    return ",".join(
        f"{value:.{RESULT_PRECISION}f}"
        for value in (bbox.xPx, bbox.yPx, bbox.widthPx, bbox.heightPx)
    )


@dataclass(slots=True)
class TextResultWriter:
    """Write plain-text tracking results in a stable line format."""

    destination: Path | None = None

    def formatResult(self, result: TrackResult) -> str:
        return formatResultLine(result)

    def parseLine(self, line: str) -> BBoxXYWH:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise OutputError(f"invalid result line: {line!r}")
        try:
            xPx, yPx, widthPx, heightPx = (float(part) for part in parts)
        except ValueError as error:
            raise OutputError(f"invalid result line: {line!r}") from error
        return BBoxXYWH(xPx=xPx, yPx=yPx, widthPx=widthPx, heightPx=heightPx)



__all__ = ["RESULT_PRECISION", "TextResultWriter", "formatResultLine"]
