"""The 360VOTS sequence table (``360vots-info.csv``).

One row per source clip.  It links the two benchmarks built from the same clips:
the 360VOS split and ID a clip has, and its 360VOT ID when it is also one of the 120
360VOT test sequences.  It also carries the challenge attributes of every clip.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from track360.core.errors import DecodeError

ATTRIBUTE_COLUMN_START = 6
_TIME_RANGE = re.compile(r"^(\d{1,2}-\d{2}|start|end)$")


@dataclass(frozen=True, slots=True)
class VotsClip:
    """One row of the table."""

    clip: str  # source video ID plus the time range cut from it
    target: str
    vosSplit: str  # train | test | test-challange | none
    vosId: str | None  # 360VOS sequence name within its split, e.g. "012"
    votId: str | None  # 360VOT sequence name, e.g. "0006"
    multipleTargets: bool
    attributes: frozenset[str]

    @property
    def sourceVideo(self) -> str:
        """The video the clip was cut from: two clips of one video share scenes."""
        parts = self.clip.split("_")
        while len(parts) > 1 and _TIME_RANGE.match(parts[-1]):
            parts.pop()
        return "_".join(parts)


class VotsInfo:
    def __init__(self, clips: tuple[VotsClip, ...], attributeNames: tuple[str, ...]) -> None:
        self.clips = clips
        self.attributeNames = attributeNames

    def vosClips(self, split: str) -> tuple[VotsClip, ...]:
        return tuple(clip for clip in self.clips if clip.vosSplit == split and clip.vosId)

    def votClips(self) -> tuple[VotsClip, ...]:
        return tuple(clip for clip in self.clips if clip.votId is not None)

    def votAttributes(self) -> dict[str, frozenset[str]]:
        """Attributes of each 360VOT sequence, keyed by sequence name."""
        return {clip.votId: clip.attributes for clip in self.clips if clip.votId is not None}

    def vosAttributes(self, split: str) -> dict[str, frozenset[str]]:
        return {clip.vosId: clip.attributes for clip in self.vosClips(split) if clip.vosId}


def loadVotsInfo(path: str | Path) -> VotsInfo:
    tablePath = Path(path)
    try:
        with tablePath.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
    except OSError as error:
        raise DecodeError(f"cannot read 360VOTS info table {tablePath}: {error}") from error
    if not rows or len(rows[0]) <= ATTRIBUTE_COLUMN_START:
        raise DecodeError(f"360VOTS info table has an unexpected header: {tablePath}")
    # Headers look like "CB(cross border)"; the short code is the attribute name.
    names = tuple(cell.split("(", 1)[0].strip() for cell in rows[0][ATTRIBUTE_COLUMN_START:])
    clips: list[VotsClip] = []
    for row in rows[1:]:
        cells = [cell.strip() for cell in row] + [""] * (len(rows[0]) - len(row))
        split = cells[3]
        if not cells[0] or not split or "+" in cells[0]:
            continue  # blank lines and the summary rows at the end of the table
        clips.append(
            VotsClip(
                clip=cells[0],
                target=cells[2],
                vosSplit=split,
                vosId=cells[1] or None,
                votId=f"{int(cells[5]):04d}" if cells[5] else None,
                multipleTargets=bool(cells[4]),
                attributes=frozenset(
                    name
                    for name, cell in zip(names, cells[ATTRIBUTE_COLUMN_START:], strict=False)
                    if cell
                ),
            )
        )
    if not clips:
        raise DecodeError(f"360VOTS info table has no clips: {tablePath}")
    return VotsInfo(tuple(clips), names)


__all__ = ["VotsClip", "VotsInfo", "loadVotsInfo"]
