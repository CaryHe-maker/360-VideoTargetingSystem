"""Choose the tune set: 360VOS training sequences that cannot leak the 360VOT test set.

Most 360VOS training clips are the very clips 360VOT tests on, and a few more are cut
from the same videos.  Tuning on either would be tuning on the test set, so both are
excluded before anything is selected.  The selection itself looks only at the
attribute table and the sequence lengths, never at tracking results.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from track360.core.errors import DecodeError
from track360.datasets.vots_info import VotsClip, VotsInfo

TRAIN_SPLIT = "train"


@dataclass(frozen=True, slots=True)
class Exclusion:
    vosId: str
    reason: str
    detail: str


def excludedTrainClips(info: VotsInfo) -> tuple[Exclusion, ...]:
    """List every 360VOS training clip that must not be used for tuning, with the reason."""
    votByVideo: dict[str, list[str]] = {}
    for clip in info.votClips():
        assert clip.votId is not None
        votByVideo.setdefault(clip.sourceVideo, []).append(clip.votId)
    exclusions: list[Exclusion] = []
    for clip in info.vosClips(TRAIN_SPLIT):
        assert clip.vosId is not None
        if clip.votId is not None:
            exclusions.append(Exclusion(clip.vosId, "is_360vot_sequence", clip.votId))
        elif clip.sourceVideo in votByVideo:
            exclusions.append(
                Exclusion(
                    clip.vosId,
                    "same_video_as_360vot_sequence",
                    "+".join(sorted(votByVideo[clip.sourceVideo])),
                )
            )
        elif clip.multipleTargets:
            exclusions.append(Exclusion(clip.vosId, "multiple_targets", ""))
    return tuple(exclusions)


def leakFreePool(info: VotsInfo) -> tuple[VotsClip, ...]:
    """The training clips that are safe to tune on."""
    excluded = {item.vosId for item in excludedTrainClips(info)}
    return tuple(clip for clip in info.vosClips(TRAIN_SPLIT) if clip.vosId not in excluded)


def selectTuneSet(
    pool: tuple[VotsClip, ...],
    frameCounts: Mapping[str, int],
    *,
    size: int,
    maxFrames: int,
) -> tuple[VotsClip, ...]:
    """Greedily pick ``size`` clips that cover the challenge attributes evenly.

    Each step takes the clip whose attributes are rarest among the clips picked so
    far; ties go to the shorter clip, then the lower ID.  Clips longer than
    ``maxFrames`` are left out to keep a tuning run short.
    """
    candidates = [
        clip
        for clip in pool
        if clip.vosId is not None and frameCounts[clip.vosId] <= maxFrames
    ]
    covered: dict[str, int] = {}
    chosen: list[VotsClip] = []
    while candidates and len(chosen) < size:

        def rank(clip: VotsClip) -> tuple[float, int, str]:
            assert clip.vosId is not None
            gain = sum(1.0 / (1 + covered.get(name, 0)) for name in clip.attributes)
            return (-gain, frameCounts[clip.vosId], clip.vosId)

        best = min(candidates, key=rank)
        candidates.remove(best)
        chosen.append(best)
        for name in best.attributes:
            covered[name] = covered.get(name, 0) + 1
    return tuple(sorted(chosen, key=lambda clip: clip.vosId or ""))


def readSequenceFile(path: str | Path) -> list[str]:
    """Read sequence names, one per line; ``#`` starts a comment."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DecodeError(f"cannot read sequence file {path}: {error}") from error
    names = [line.split("#", 1)[0].strip() for line in lines]
    return [name for name in names if name]


__all__ = [
    "Exclusion",
    "excludedTrainClips",
    "leakFreePool",
    "readSequenceFile",
    "selectTuneSet",
]
