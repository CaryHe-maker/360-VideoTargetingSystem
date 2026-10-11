"""360VOT benchmark scores, computed by the official toolkit's own metric code.

The numbers must be comparable with published results, so every per-sequence curve
comes from the vendored toolkit functions and the aggregation below repeats what the
toolkit's ``show_result`` prints:

* ``bbox`` results give S_dual (success AUC), P_dual (centers within 20 px),
  normalized P_dual and P_angle (centers within 3 degrees);
* ``bfov`` / ``rbfov`` results give S_sphere (spherical IoU AUC) and P_angle.

``bbox`` results also get this project's loss rate (``evaluation/loss_rate.py``), which
is not a toolkit metric.

Toolkit behavior worth knowing when reading a score:

* frames where the target is absent stay in the denominator, so they always count
  as failures;
* dual success compares a prediction with the ground truth and with the ground
  truth shifted one image width to the left, never to the right;
* P_dual treats a ground-truth center with a non-positive coordinate as a hit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from track360.core.errors import DecodeError, ProtocolError
from track360.evaluation.loss_rate import sequenceLoss
from track360.io.vot360_results import readResultFile
from track360.third_party.vot360_toolkit import ope_benchmark as toolkit

PRECISION_THRESHOLD_PX = 20
ANGLE_THRESHOLD_DEG = 3.0
SUPPORTED_REPRESENTATIONS = ("bbox", "bfov", "rbfov")

GroundTruth = Mapping[str, tuple[NDArray[np.float64], NDArray[np.bool_]]]


@dataclass(frozen=True, slots=True)
class Vot360SequenceScore:
    success: float
    anglePrecision: float
    precision: float | None = None
    normPrecision: float | None = None
    frames: int = 0
    lostFrames: int | None = None
    firstLostFrame: int | None = None


@dataclass(frozen=True, slots=True)
class Vot360Scores:
    """Benchmark scores of one tracker in one representation."""

    representation: str
    sequenceCount: int
    frameCount: int
    success: float
    anglePrecision: float
    precision: float | None
    normPrecision: float | None
    perSequence: dict[str, Vot360SequenceScore]
    successCurve: NDArray[np.float64]
    anglePrecisionCurve: NDArray[np.float64]
    lossRate: float | None = None

    @property
    def successName(self) -> str:
        return "S_dual" if self.representation == "bbox" else "S_sphere"

    def summary(self) -> dict[str, float | int | str | None]:
        return {
            "representation": self.representation,
            "sequences": self.sequenceCount,
            "frames": self.frameCount,
            self.successName: self.success,
            "P_dual": self.precision,
            "norm_P_dual": self.normPrecision,
            "P_angle": self.anglePrecision,
            "loss_rate": self.lossRate,
        }


def evaluateVot360(
    groundTruth: GroundTruth,
    results: Mapping[str, NDArray[np.float64]],
    representation: str,
    *,
    frameWidthPx: int = 3840,
    frameHeightPx: int = 1920,
) -> Vot360Scores:
    """Score one tracker's results against ground truth in the toolkit layout.

    ``groundTruth`` maps a sequence name to the rows and presence mask returned by
    ``Vot360Sequence.groundTruth``; ``results`` maps the same names to result rows.
    Every sequence in ``results`` is scored and must cover all of its frames.
    """
    if representation not in SUPPORTED_REPRESENTATIONS:
        raise ProtocolError(
            f"unsupported 360VOT representation '{representation}'; "
            f"supported: {', '.join(SUPPORTED_REPRESENTATIONS)}"
        )
    if not results:
        raise ProtocolError("no results to evaluate")
    columns = 4 if representation == "bbox" else 5
    gt: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for name, rows in results.items():
        if name not in groundTruth:
            raise ProtocolError(f"no ground truth for sequence '{name}'")
        truth, present = groundTruth[name]
        if rows.ndim != 2 or rows.shape[1] != columns:
            raise ProtocolError(
                f"{representation} results need {columns} columns: "
                f"sequence={name}, shape={rows.shape}"
            )
        if len(rows) != len(truth):
            raise ProtocolError(
                f"result length does not match the sequence: sequence={name}, "
                f"results={len(rows)}, frames={len(truth)}"
            )
        # The toolkit marks present frames with 1 in a float array.
        gt[name] = (truth, present.astype(np.float64))

    tracker = "tracker"
    wrapped = {tracker: dict(results)}
    spherical = representation != "bbox"
    if spherical:
        success = toolkit.eval_success(gt, tracker, wrapped, True)[tracker]
        angle = toolkit.eval_angle_precision(gt, tracker, wrapped, True, True)[tracker]
        precision = normPrecision = None
    else:
        success = toolkit.eval_success(gt, tracker, wrapped, False, frameWidthPx)[tracker]
        angle = toolkit.eval_angle_precision(
            gt, tracker, wrapped, False, False, frameWidthPx, frameHeightPx
        )[tracker]
        precision = toolkit.eval_precision(gt, tracker, wrapped)[tracker]
        normPrecision = toolkit.eval_norm_precision(gt, tracker, wrapped)[tracker]

    losses = (
        {}
        if spherical
        else {
            name: sequenceLoss(groundTruth[name][0], groundTruth[name][1], rows, frameWidthPx)
            for name, rows in results.items()
        }
    )
    angleIndex = int(round(ANGLE_THRESHOLD_DEG * 10))
    perSequence = {
        name: Vot360SequenceScore(
            success=float(np.mean(success[name])),
            anglePrecision=float(angle[name][angleIndex]),
            precision=(
                None if precision is None else float(precision[name][PRECISION_THRESHOLD_PX])
            ),
            normPrecision=None if normPrecision is None else float(np.mean(normPrecision[name])),
            frames=len(results[name]),
            lostFrames=losses[name].lostFrames if name in losses else None,
            firstLostFrame=losses[name].firstLostFrame if name in losses else None,
        )
        for name in results
    }
    successCurve = np.mean(list(success.values()), axis=0)
    angleCurve = np.mean(list(angle.values()), axis=0)
    return Vot360Scores(
        representation=representation,
        sequenceCount=len(results),
        frameCount=int(sum(len(rows) for rows in results.values())),
        success=float(np.mean(list(success.values()))),
        anglePrecision=float(angleCurve[angleIndex]),
        precision=(
            None
            if precision is None
            else float(np.mean(list(precision.values()), axis=0)[PRECISION_THRESHOLD_PX])
        ),
        normPrecision=(
            None if normPrecision is None else float(np.mean(list(normPrecision.values())))
        ),
        perSequence=perSequence,
        successCurve=successCurve,
        anglePrecisionCurve=angleCurve,
        lossRate=(
            None
            if spherical
            else sum(item.lostFrames for item in losses.values())
            / max(1, sum(item.frames for item in losses.values()))
        ),
    )


def loadTrackerResults(directory: str | Path) -> dict[str, NDArray[np.float64]]:
    """Load every ``<sequence>.txt`` of one tracker's result directory."""
    root = Path(directory)
    if not root.is_dir():
        raise DecodeError(f"result directory does not exist: {root}")
    results = {path.stem: readResultFile(path) for path in sorted(root.glob("*.txt"))}
    if not results:
        raise DecodeError(f"no result files in {root}")
    return results


__all__ = [
    "ANGLE_THRESHOLD_DEG",
    "PRECISION_THRESHOLD_PX",
    "SUPPORTED_REPRESENTATIONS",
    "Vot360Scores",
    "Vot360SequenceScore",
    "evaluateVot360",
    "loadTrackerResults",
]
