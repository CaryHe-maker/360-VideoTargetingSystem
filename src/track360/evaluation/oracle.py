"""Partial oracles: ground truth corrects a share of one stage's mistakes (E052).

Evaluation only.  Each stage of loss handling can be lifted from what it does now
towards a level a better method could plausibly reach, by letting the ground truth
correct a share ``p`` of that stage's mistakes, drawn per event:

    A  noticing     a stretch of frames whose box misses the target (IoU < 0.1) is,
                    from its third frame on, not trusted;
    B  search place an enlarged search that misses the target is centred near it, at
                    an offset drawn from where in-view searches find the target now;
    C  boxing       an enlarged search with the target in view but no candidate on it
                    gets a candidate near the target, with the search score and the
                    similarity of a real candidate on the target (it must still be taken);
    D  taking       a frame with a candidate on the target and no jump jumps to it;
    E  not jumping  a jump to a candidate off the target, or while the tracker's own
                    box is on the target, is not made.

The draws are seeded by the run's seed, the sequence, the stage and the event, so a
run is reproducible and a sequence whose events are never drawn tracks exactly as
without the oracle.  What each stage did is counted for the report.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from track360.core.types import BBoxXYWH, BFoV, ProjectedObservation, SphericalPoint
from track360.evaluation.loss_rate import dualIou
from track360.geometry import makeSphericalPoint

STAGES = ("A", "B", "C", "D", "E")
LOST_IOU, GOOD_IOU = 0.1, 0.5
# Stage A: the share of the stretch's frames that pass before it is noticed.
NOTICE_DELAY_FRAMES = 3


@dataclass(frozen=True, slots=True)
class OracleSettings:
    """The share of each stage's mistakes the ground truth corrects (0: none)."""

    shares: Mapping[str, float]
    # Distributions taken from base-2 runs: in-view reach of searches, and the search
    # score and similarity of candidates on the target (outputs/E052/distributions.npz).
    data: Path
    seed: int = 0

    @staticmethod
    def parse(text: str, data: Path, seed: int = 0) -> OracleSettings:
        shares = {}
        for item in text.split(","):
            stage, _, value = item.partition("=")
            stage = stage.strip().upper()
            if stage not in STAGES or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"bad oracle share '{item}'")
            shares[stage] = float(value)
        return OracleSettings(shares=shares, data=Path(data), seed=seed)

    def describe(self) -> dict[str, object]:
        return {"shares": dict(self.shares), "data": str(self.data), "seed": self.seed}


@dataclass
class SequenceOracle:
    settings: OracleSettings
    sequence: str
    truth: list[BFoV | None]
    truthBox: np.ndarray
    present: np.ndarray
    frameWidthPx: int
    frameHeightPx: int
    counts: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        data = np.load(self.settings.data)
        self._reach = np.sort(data["reach"])
        self._score = data["score"]
        self._similarity = data["similarity"]
        self._missRun = 0
        self._caught = False
        self.counts = {stage: [0, 0] for stage in STAGES}  # eligible, corrected

    # ---------------------------------------------------------------- shared
    def share(self, stage: str) -> float:
        return float(self.settings.shares.get(stage, 0.0))

    def _uniform(self, stage: str, frame: int, extra: str = "") -> float:
        key = f"{self.settings.seed}:{self.sequence}:{stage}:{frame}:{extra}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2.0**64

    def _draw(self, stage: str, frame: int) -> bool:
        self.counts[stage][0] += 1
        hit = self._uniform(stage, frame) < self.share(stage)
        self.counts[stage][1] += int(hit)
        return hit

    def _target(self, frame: int) -> BFoV | None:
        return self.truth[frame] if 0 <= frame < len(self.truth) else None

    def boxIou(self, frame: int, box: BBoxXYWH | None) -> float:
        if box is None or frame >= len(self.present) or not self.present[frame]:
            return 0.0
        row = np.asarray([[box.xPx, box.yPx, box.widthPx, box.heightPx]])
        return float(dualIou(self.truthBox[frame : frame + 1], row, self.frameWidthPx)[0])

    def onTarget(self, frame: int, bfov: BFoV) -> bool:
        """The rule of the event reports: centre within half the larger size."""
        target = self._target(frame)
        if target is None:
            return False
        size = max(_size(target), _size(bfov))
        return _angle(target.center, bfov.center) < 0.5 * size

    # ---------------------------------------------------------------- A: noticing
    def forceDoubt(self, frame: int, box: BBoxXYWH | None) -> bool:
        """Called once per frame with the box the tracker is about to commit."""
        if not self.share("A") or frame >= len(self.present) or not self.present[frame]:
            return False
        if self.boxIou(frame, box) >= LOST_IOU:
            self._missRun, self._caught = 0, False
            return False
        self._missRun += 1
        if self._missRun == 1:
            self._caught = self._draw("A", frame)
        return self._caught and self._missRun >= NOTICE_DELAY_FRAMES

    # ---------------------------------------------------------------- B: search place
    def scanCentre(self, frame: int, centre: SphericalPoint, halfWidthRad: float) -> SphericalPoint:
        target = self._target(frame)
        if not self.share("B") or target is None:
            return centre
        if _angle(target.center, centre) <= halfWidthRad:
            return centre
        if not self._draw("B", frame):
            return centre
        quantile = self._uniform("B", frame, "reach")
        reach = float(np.quantile(self._reach, quantile))
        bearing = 2.0 * math.pi * self._uniform("B", frame, "bearing")
        return _offset(target.center, reach * halfWidthRad, bearing)

    # ---------------------------------------------------------------- C: boxing
    def extraCandidate(
        self,
        frame: int,
        viewCentre: SphericalPoint,
        halfWidthRad: float,
        candidates: tuple[ProjectedObservation, ...],
        geometry: Any,
    ) -> ProjectedObservation | None:
        target = self._target(frame)
        if not self.share("C") or target is None:
            return None
        if _angle(target.center, viewCentre) > halfWidthRad:
            return None
        if any(self.onTarget(frame, c.bfov) for c in candidates):
            return None
        if not self._draw("C", frame):
            return None
        # A box near the target, as a detector would give it: shifted by up to 15% of
        # the size and scaled by up to +-15% (IoU about 0.6-0.8).
        size = _size(target)
        shift = 0.15 * size * self._uniform("C", frame, "shift")
        bearing = 2.0 * math.pi * self._uniform("C", frame, "bearing")
        scale = 0.85 + 0.30 * self._uniform("C", frame, "scale")
        bfov = BFoV(
            _offset(target.center, shift, bearing),
            min(target.horizontalFovRad * scale, math.pi * 0.99),
            min(target.verticalFovRad * scale, math.pi * 0.99),
        )
        bbox = geometry.bfovToBbox(bfov, self.frameWidthPx, self.frameHeightPx)
        pick = int(self._uniform("C", frame, "score") * len(self._score))
        score = float(np.clip(self._score[pick], 0.0, 1.0))
        return ProjectedObservation(
            viewId=candidates[0].viewId if candidates else 0,
            bfov=bfov,
            bbox=bbox,
            modelScore=score,
            appearanceScore=score,
            motionScore=0.0,
            scaleScore=1.0,
            fusedScore=score,
            singleScore=score,
            appearanceSimilarity=float(self._similarity[pick]),
        )

    # ---------------------------------------------------------------- D, E: jumping
    def takeMissed(
        self, frame: int, candidates: list[ProjectedObservation]
    ) -> ProjectedObservation | None:
        """D: of a frame that would not jump, the candidate on the target to jump to."""
        if not self.share("D"):
            return None
        good = [c for c in candidates if self.onTarget(frame, c.bfov)]
        if not good or not self._draw("D", frame):
            return None
        return max(good, key=lambda c: float(c.singleScore or c.fusedScore))

    def vetoJump(
        self, frame: int, chosen: ProjectedObservation, trackerBox: BBoxXYWH | None
    ) -> bool:
        """E: whether a wrong jump is called off."""
        if not self.share("E"):
            return False
        wrong = (not self.onTarget(frame, chosen.bfov)) or self.boxIou(
            frame, trackerBox
        ) >= GOOD_IOU
        return wrong and self._draw("E", frame)

    def report(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "shares": dict(self.settings.shares),
            "eligible": {stage: c[0] for stage, c in self.counts.items()},
            "corrected": {stage: c[1] for stage, c in self.counts.items()},
        }


def sequenceOracle(settings: OracleSettings, sequence: Any, name: str) -> SequenceOracle:
    """The oracle of one 360VOT sequence (``track360.datasets.vot360``)."""
    truthBox, present = sequence.groundTruth("bbox")
    width, height = sequence.frameSize
    truth = [sequence.annotation(index).bfov for index in range(sequence.frameCount)]
    return SequenceOracle(
        settings=settings,
        sequence=name,
        truth=truth,
        truthBox=np.asarray(truthBox, dtype=float),
        present=np.asarray(present, dtype=bool),
        frameWidthPx=int(width),
        frameHeightPx=int(height),
    )


def _size(bfov: BFoV) -> float:
    return math.sqrt(bfov.horizontalFovRad * bfov.verticalFovRad)


def _angle(first: SphericalPoint, second: SphericalPoint) -> float:
    cosine = first.x * second.x + first.y * second.y + first.z * second.z
    return math.acos(max(-1.0, min(1.0, cosine)))


def _offset(centre: SphericalPoint, angle: float, bearing: float) -> SphericalPoint:
    """The point ``angle`` radians from ``centre`` in direction ``bearing``."""
    lat1, lon1 = centre.pitchRad, centre.yawRad
    lat2 = math.asin(
        max(
            -1.0,
            min(
                1.0,
                math.sin(lat1) * math.cos(angle)
                + math.cos(lat1) * math.sin(angle) * math.cos(bearing),
            ),
        )
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angle) * math.cos(lat1),
        math.cos(angle) - math.sin(lat1) * math.sin(lat2),
    )
    lon2 = (lon2 + math.pi) % (2.0 * math.pi) - math.pi
    return makeSphericalPoint(lon2, lat2)
