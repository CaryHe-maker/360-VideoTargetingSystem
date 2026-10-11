"""How well a tracker's per-frame confidence tells good frames from lost ones.

The confidence is the only signal the tracker has at run time; this module compares it
with the IoU against ground truth, which only an evaluation has.  Everything is
computed on frames where the target is present, the first frame excluded (its box is
given, its confidence is a constant).

Frames are grouped the way the loss rate sees them (``evaluation/loss_rate.py``):

* ``good``: IoU of at least ``GOOD_IOU_THRESHOLD``;
* ``lost``: inside a loss run (IoU below 0.1 for at least 5 consecutive frames);
* ``weak``: everything else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from track360.core.errors import ProtocolError
from track360.evaluation.loss_rate import dualIou, lostFrameMask

GOOD_IOU_THRESHOLD = 0.5
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
IOU_BIN_EDGES = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)
ONSET_OFFSETS = (-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20)


@dataclass(frozen=True, slots=True)
class FrameTable:
    """Per-frame score, IoU and loss flag of every scored frame, with its sequence."""

    sequence: NDArray[np.str_]
    frame: NDArray[np.int64]
    score: NDArray[np.float64]
    iou: NDArray[np.float64]
    lost: NDArray[np.bool_]

    @property
    def good(self) -> NDArray[np.bool_]:
        return self.iou >= GOOD_IOU_THRESHOLD

    def __len__(self) -> int:
        return len(self.score)


@dataclass(frozen=True, slots=True)
class ThresholdRow:
    """What flagging every frame with a score below ``threshold`` would catch."""

    threshold: float
    lostFlagged: float  # share of lost frames flagged (higher is better)
    goodFlagged: float  # share of good frames flagged (lower is better)
    flaggedLost: float  # share of flagged frames that are lost
    flagged: float  # share of all frames flagged


def buildFrameTable(
    groundTruth: Mapping[str, tuple[NDArray[np.float64], NDArray[np.bool_]]],
    results: Mapping[str, NDArray[np.float64]],
    scores: Mapping[str, NDArray[np.float64]],
    frameWidthPx: int,
) -> FrameTable:
    """Join BBox results, scores and ground truth of the sequences that have all three."""
    names = sorted(set(results) & set(scores) & set(groundTruth))
    if not names:
        raise ProtocolError("no sequence has results, scores and ground truth")
    keys = ("seq", "frame", "score", "iou", "lost")
    columns: dict[str, list[NDArray]] = {key: [] for key in keys}
    for name in names:
        truth, present = groundTruth[name]
        rows, score = results[name], scores[name]
        if not len(truth) == len(rows) == len(score):
            raise ProtocolError(
                f"sequence {name}: ground truth, results and scores differ in length "
                f"({len(truth)}, {len(rows)}, {len(score)})"
            )
        presentMask = np.asarray(present, dtype=bool)
        iou = np.zeros(len(truth), dtype=np.float64)
        if presentMask.any():
            iou[presentMask] = dualIou(truth[presentMask], rows[presentMask], frameWidthPx)
        lost = lostFrameMask(iou, presentMask)
        keep = presentMask.copy()
        keep[0] = False
        index = np.flatnonzero(keep)
        columns["seq"].append(np.full(len(index), name))
        columns["frame"].append(index)
        columns["score"].append(score[index])
        columns["iou"].append(iou[index])
        columns["lost"].append(lost[index])
    return FrameTable(
        sequence=np.concatenate(columns["seq"]),
        frame=np.concatenate(columns["frame"]).astype(np.int64),
        score=np.concatenate(columns["score"]).astype(np.float64),
        iou=np.concatenate(columns["iou"]).astype(np.float64),
        lost=np.concatenate(columns["lost"]).astype(bool),
    )


def auroc(scores: NDArray[np.float64], positive: NDArray[np.bool_]) -> float | None:
    """Probability that a positive frame scores higher than a negative one (ties half)."""
    positives = int(positive.sum())
    negatives = len(positive) - positives
    if positives == 0 or negatives == 0:
        return None
    rankSum = _ranks(scores)[positive].sum()
    return float((rankSum - positives * (positives + 1) / 2.0) / (positives * negatives))


def spearman(first: NDArray[np.float64], second: NDArray[np.float64]) -> float | None:
    if len(first) < 2 or np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return None
    return float(np.corrcoef(_ranks(first), _ranks(second))[0, 1])


def pearson(first: NDArray[np.float64], second: NDArray[np.float64]) -> float | None:
    if len(first) < 2 or np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def percentiles(values: NDArray[np.float64]) -> dict[str, float] | None:
    if len(values) == 0:
        return None
    return {f"p{q}": float(np.percentile(values, q)) for q in PERCENTILES}


def thresholdTable(table: FrameTable, thresholds: Sequence[float]) -> tuple[ThresholdRow, ...]:
    rows = []
    good, lost = table.good, table.lost
    for threshold in thresholds:
        flagged = table.score < threshold
        rows.append(
            ThresholdRow(
                threshold=float(threshold),
                lostFlagged=_share(flagged & lost, lost),
                goodFlagged=_share(flagged & good, good),
                flaggedLost=_share(flagged & lost, flagged),
                flagged=float(flagged.mean()) if len(flagged) else 0.0,
            )
        )
    return tuple(rows)


def bestThreshold(table: FrameTable) -> ThresholdRow | None:
    """Threshold that best separates lost from good frames (maximum Youden index)."""
    if not table.lost.any() or not table.good.any():
        return None
    candidates = np.unique(np.quantile(table.score, np.linspace(0.0, 1.0, 401)))
    rows = thresholdTable(table, candidates)
    return max(rows, key=lambda row: row.lostFlagged - row.goodFlagged)


def onsetProfile(table: FrameTable) -> dict[int, dict[str, float | int]]:
    """Median score at fixed offsets around the first frame of every loss run.

    Offsets count scored frames of the same sequence, so frames where the target is
    absent are skipped.
    """
    values: dict[int, list[float]] = {offset: [] for offset in ONSET_OFFSETS}
    for name in np.unique(table.sequence):
        index = np.flatnonzero(table.sequence == name)
        lost, score = table.lost[index], table.score[index]
        starts = np.flatnonzero(lost & ~np.concatenate(([False], lost[:-1])))
        for start in starts:
            for offset in ONSET_OFFSETS:
                position = start + offset
                if 0 <= position < len(index):
                    values[offset].append(float(score[position]))
    return {
        offset: {"runs": len(items), "median": float(np.median(items))}
        for offset, items in values.items()
        if items
    }


def summarize(table: FrameTable) -> dict[str, object]:
    """Everything the report prints, as plain data."""
    good, lost = table.good, table.lost
    weak = ~good & ~lost
    best = bestThreshold(table)
    grid = np.round(np.arange(0.30, 0.701, 0.02), 2)
    perSequence = {}
    for name in np.unique(table.sequence):
        index = table.sequence == name
        perSequence[str(name)] = {
            "frames": int(index.sum()),
            "lostShare": float(lost[index].mean()),
            "medianScore": float(np.median(table.score[index])),
            "medianScoreGood": _median(table.score[index & good]),
            "medianScoreLost": _median(table.score[index & lost]),
            "aurocLost": auroc(-table.score[index], lost[index]),
            "spearman": spearman(table.score[index], table.iou[index]),
        }
    return {
        "frames": len(table),
        "sequences": int(len(np.unique(table.sequence))),
        "share": {
            "good": float(good.mean()),
            "weak": float(weak.mean()),
            "lost": float(lost.mean()),
        },
        "correlation": {
            "pearson": pearson(table.score, table.iou),
            "spearman": spearman(table.score, table.iou),
        },
        # 0.5: no better than chance; 1.0: a threshold separates the two groups perfectly.
        "auroc": {
            "lostVsRest": auroc(-table.score, lost),
            "lostVsGood": auroc(-table.score[good | lost], lost[good | lost]),
            "goodVsRest": auroc(table.score, good),
        },
        "scorePercentiles": {
            "all": percentiles(table.score),
            "good": percentiles(table.score[good]),
            "weak": percentiles(table.score[weak]),
            "lost": percentiles(table.score[lost]),
        },
        "scoreByIou": [
            {
                "iouFrom": float(low),
                "iouTo": float(min(high, 1.0)),
                "frames": int(((table.iou >= low) & (table.iou < high)).sum()),
                "score": percentiles(table.score[(table.iou >= low) & (table.iou < high)]),
            }
            for low, high in zip(IOU_BIN_EDGES[:-1], IOU_BIN_EDGES[1:], strict=True)
        ],
        "thresholds": [_rowPayload(row) for row in thresholdTable(table, grid)],
        "bestThreshold": None if best is None else _rowPayload(best),
        "lossOnset": onsetProfile(table),
        "perSequence": perSequence,
    }


def _ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average ranks starting at 1, ties sharing their mean rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts), dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    return sums[inverse] / counts[inverse]


def _share(numerator: NDArray[np.bool_], denominator: NDArray[np.bool_]) -> float:
    total = int(denominator.sum())
    return float(numerator.sum() / total) if total else 0.0


def _median(values: NDArray[np.float64]) -> float | None:
    return float(np.median(values)) if len(values) else None


def _rowPayload(row: ThresholdRow) -> dict[str, float]:
    return {
        "threshold": row.threshold,
        "lostFlagged": row.lostFlagged,
        "goodFlagged": row.goodFlagged,
        "flaggedLost": row.flaggedLost,
        "flagged": row.flagged,
    }


__all__ = [
    "GOOD_IOU_THRESHOLD",
    "FrameTable",
    "ThresholdRow",
    "auroc",
    "bestThreshold",
    "buildFrameTable",
    "onsetProfile",
    "pearson",
    "percentiles",
    "spearman",
    "summarize",
    "thresholdTable",
]
