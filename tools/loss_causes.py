"""What was the target doing when the tracker lost it?

    python tools/loss_causes.py --dataset-root <dir> --label-root <dir> \\
        --run outputs/E015_tune --run outputs/E015_holdout --csv outputs/E019/loss_events.csv

A loss event is the start of a lost run of a result file (IoU < 0.1 for at least five
frames, the loss-rate definition).  The ground truth just before each event is examined
and the event is put into one class, checked in this order:

    disappeared   the target is absent in one of the frames around the event
    grew          its mean angular size grew by ``--size-ratio`` within the window
    shrank        its mean angular size shrank by that ratio within the window
    jumped        none of the above, and in one frame of the window it moved more than
                  ``--jump`` target sizes away from where constant velocity put it
    other         none of the above: the target was there, steady in size and position

One row per event goes to the CSV, with the raw measurements, so other thresholds can
be tried without recomputing.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path

import numpy as np

from track360.core.errors import Track360Error
from track360.datasets.vot360 import Vot360Dataset
from track360.evaluation.loss_rate import dualIou, lostFrameMask
from track360.evaluation.vot360_metrics import loadTrackerResults

CLASSES = ("disappeared", "grew", "shrank", "jumped", "other")
COLUMNS = (
    "run",
    "sequence",
    "frame",
    "lostFrames",
    "cause",
    "absentNearby",
    "absentBefore",
    "sizeRatio",
    "targetSizeDeg",
    "stepSizes",
    "stepDeg",
    "surpriseSizes",
    "surpriseDeg",
)
STEP_EDGES = (0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, float("inf"))


def _directions(rows: np.ndarray) -> np.ndarray:
    longitude, latitude = np.radians(rows[:, 0]), np.radians(rows[:, 1])
    return np.column_stack(
        [
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
            np.cos(latitude) * np.cos(longitude),
        ]
    )


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.arccos(np.clip(float(first @ second), -1.0, 1.0)))


def events(args: argparse.Namespace) -> list[dict]:
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    rows = []
    for run in args.run:
        results = loadTrackerResults(run / "bbox" / args.method)
        for name in sorted(results):
            sequence = dataset.sequence(name)
            truthBox, present = sequence.groundTruth("bbox")
            truthBfov, _ = sequence.groundTruth("bfov")
            frameWidthPx = sequence.frameSize[0]
            sequence.close()
            present = np.asarray(present, dtype=bool)
            iou = np.zeros(len(truthBox))
            iou[present] = dualIou(truthBox[present], results[name][present], frameWidthPx)
            lost = lostFrameMask(iou, present)
            direction = _directions(truthBfov)
            size = np.radians(np.sqrt(np.clip(truthBfov[:, 2] * truthBfov[:, 3], 0.0, None)))
            index = 0
            while index < len(lost):
                if not lost[index]:
                    index += 1
                    continue
                end = index
                while end + 1 < len(lost) and (lost[end + 1] or not present[end + 1]):
                    end += 1
                while not lost[end]:
                    end -= 1
                start = index
                low = max(0, start - args.window)
                window = [i for i in range(low, start + 1) if present[i]]
                before = bool((~present[max(0, start - args.window) : start + 1]).any())
                absent = before or bool((~present[start : start + args.window + 1]).any())
                ratio = float(size[start] / size[window[0]]) if size[window[0]] > 0 else 1.0
                step = surprise = 0.0
                stepDeg = surpriseDeg = 0.0
                for i in window[1:]:
                    if not present[i - 1]:
                        continue
                    moved = _angle(direction[i], direction[i - 1])
                    scale = max(size[i - 1], 1e-6)
                    if moved / scale > step:
                        step, stepDeg = moved / scale, math.degrees(moved)
                    if i >= 2 and present[i - 2]:
                        # Where constant velocity over the last two frames put the target.
                        previous, older = direction[i - 1], direction[i - 2]
                        predicted = 2.0 * float(previous @ older) * previous - older
                        predicted /= np.linalg.norm(predicted)
                        off = _angle(direction[i], predicted)
                        if off / scale > surprise:
                            surprise, surpriseDeg = off / scale, math.degrees(off)
                if absent:
                    cause = "disappeared"
                elif ratio >= args.size_ratio:
                    cause = "grew"
                elif ratio <= 1.0 / args.size_ratio:
                    cause = "shrank"
                elif surprise >= args.jump:
                    cause = "jumped"
                else:
                    cause = "other"
                rows.append(
                    {
                        "run": run.name,
                        "sequence": name,
                        "frame": start,
                        "lostFrames": int(lost[start : end + 1].sum()),
                        "cause": cause,
                        "absentNearby": int(absent),
                        "absentBefore": int(before),
                        "sizeRatio": round(ratio, 4),
                        "targetSizeDeg": round(math.degrees(size[start]), 3),
                        "stepSizes": round(step, 4),
                        "stepDeg": round(stepDeg, 3),
                        "surpriseSizes": round(surprise, 4),
                        "surpriseDeg": round(surpriseDeg, 3),
                    }
                )
                index = end + 1
    return rows


def report(rows: list[dict], args: argparse.Namespace) -> None:
    frames = sum(row["lostFrames"] for row in rows)
    print(
        f"{len(rows)} loss events, {frames} lost frames "
        f"(window {args.window} frames, size ratio {args.size_ratio}, jump {args.jump} sizes)"
    )
    print(f"  {'cause':<13}{'events':>8}{'share':>8}{'lost frames':>13}{'share':>8}")
    for cause in CLASSES:
        chosen = [row for row in rows if row["cause"] == cause]
        lostFrames = sum(row["lostFrames"] for row in chosen)
        print(
            f"  {cause:<13}{len(chosen):>8}{len(chosen) / max(1, len(rows)):>8.1%}"
            f"{lostFrames:>13}{lostFrames / max(1, frames):>8.1%}"
        )
    steady = [row for row in rows if row["cause"] in ("jumped", "other")]
    for key, title in (
        ("surpriseSizes", "largest one-frame departure from constant velocity, in target sizes"),
        ("stepSizes", "largest one-frame displacement, in target sizes"),
    ):
        print(f"\n{title} (events with the target present and steady in size: {len(steady)})")
        print(f"  {'range':<14}{'events':>8}{'share':>8}{'lost frames':>13}{'share':>8}")
        total = sum(row["lostFrames"] for row in steady)
        for low, high in zip(STEP_EDGES[:-1], STEP_EDGES[1:], strict=True):
            chosen = [row for row in steady if low <= row[key] < high]
            lostFrames = sum(row["lostFrames"] for row in chosen)
            label = f"{low:g} - {high:g}" if math.isfinite(high) else f">= {low:g}"
            print(
                f"  {label:<14}{len(chosen):>8}{len(chosen) / max(1, len(steady)):>8.1%}"
                f"{lostFrames:>13}{lostFrames / max(1, total):>8.1%}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--method", default="ours")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--window", type=int, default=5, help="frames looked at before the event")
    parser.add_argument("--size-ratio", type=float, default=1.5)
    parser.add_argument("--jump", type=float, default=0.5)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    try:
        rows = events(args)
    except (Track360Error, ValueError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    report(rows, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
