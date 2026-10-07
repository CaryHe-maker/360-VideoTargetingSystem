"""Track-loss statistics on 360VOT BBox results.

A target counts as lost from the first frame of a run of at least
``LOST_MIN_RUN_FRAMES`` consecutive frames whose IoU is below ``LOST_IOU_THRESHOLD``;
every frame of such a run is a lost frame.  Shorter dips are not losses.  Frames where
the target is absent are never lost and do not interrupt a run.

The loss rate of a set of sequences is the number of lost frames over the number of
all frames, summed over the sequences, so long sequences weigh more.  The definition
is fixed: changing a constant makes earlier evaluation records incomparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from track360.third_party.vot360_toolkit import ope_benchmark as toolkit

LOST_IOU_THRESHOLD = 0.1
LOST_MIN_RUN_FRAMES = 5


@dataclass(frozen=True, slots=True)
class SequenceLoss:
    frames: int
    lostFrames: int
    firstLostFrame: int | None

    @property
    def lossRate(self) -> float:
        return self.lostFrames / self.frames if self.frames else 0.0


def dualIou(
    truth: NDArray[np.float64],
    results: NDArray[np.float64],
    frameWidthPx: int,
) -> NDArray[np.float64]:
    """Per-frame IoU of ``[x1, y1, w, h]`` rows, as the toolkit's dual success sees it.

    The better of the IoU with the ground truth and with the ground truth shifted one
    image width to the left, which is exactly what S_dual thresholds.
    """
    shifted = truth.copy()
    shifted[:, 0] -= frameWidthPx
    return np.maximum(
        toolkit.overlap_ratio(truth, results),
        toolkit.overlap_ratio(shifted, results),
    )


def lostFrameMask(
    iou: NDArray[np.float64],
    present: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Mark the frames that belong to a loss run; ``iou`` of absent frames is ignored."""
    lost = np.zeros(len(iou), dtype=bool)
    run: list[int] = []
    for index in range(len(iou)):
        if not present[index]:
            continue
        if iou[index] < LOST_IOU_THRESHOLD:
            run.append(index)
            continue
        if len(run) >= LOST_MIN_RUN_FRAMES:
            lost[run] = True
        run = []
    if len(run) >= LOST_MIN_RUN_FRAMES:
        lost[run] = True
    return lost


def sequenceLoss(
    truth: NDArray[np.float64],
    present: NDArray[np.bool_],
    results: NDArray[np.float64],
    frameWidthPx: int,
) -> SequenceLoss:
    """Loss statistics of one sequence from its BBox ground truth and results."""
    presentMask = np.asarray(present, dtype=bool)
    iou = np.zeros(len(truth), dtype=np.float64)
    if presentMask.any():
        iou[presentMask] = dualIou(truth[presentMask], results[presentMask], frameWidthPx)
    lost = lostFrameMask(iou, presentMask)
    indices = np.flatnonzero(lost)
    return SequenceLoss(
        frames=len(truth),
        lostFrames=int(lost.sum()),
        firstLostFrame=int(indices[0]) if indices.size else None,
    )


__all__ = [
    "LOST_IOU_THRESHOLD",
    "LOST_MIN_RUN_FRAMES",
    "SequenceLoss",
    "dualIou",
    "lostFrameMask",
    "sequenceLoss",
]
