"""Judge the state machine of a run against the ground truth.

    python tools/state_trace.py --dataset-root <dir> --label-root <dir> \\
        --output-root outputs/E022_a2_tune --csv outputs/E022/a2_tune_frames.csv

Reads the per-frame state trace a run writes when loss handling is on
(``<output-root>/trace/<method>/<sequence>.csv``), adds the IoU of every frame, and
reports:

- how good the frames of each state really were (the state as a judgement);
- what raised each doubt and how often the doubt was right;
- every jump: whether it left a good box, and whether it landed on the target;
- why scan candidates were turned down, split by whether they were on the target.

``--csv`` writes the joined per-frame table, ``--json`` the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from track360.core.errors import Track360Error
from track360.datasets.vot360 import Vot360Dataset
from track360.evaluation.loss_rate import dualIou, lostFrameMask
from track360.evaluation.vot360_metrics import loadTrackerResults

STATES = ("TRACKING", "UNCERTAIN", "LOST")
GOOD_IOU, LOST_IOU = 0.5, 0.1
# A jump is judged by the mean IoU of the frames right after it.
AFTER_FRAMES = 10


def _direction(yawDeg: float, pitchDeg: float) -> np.ndarray:
    yaw, pitch = np.radians(yawDeg), np.radians(pitchDeg)
    return np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])


def analyse(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    root = args.output_root
    results = loadTrackerResults(root / "bbox" / args.method)
    frames: list[dict[str, object]] = []
    stateIou: dict[str, list[float]] = defaultdict(list)
    absent: Counter[str] = Counter()
    doubts: dict[str, list[bool]] = defaultdict(list)
    jumps: list[dict[str, object]] = []
    verdicts: dict[str, Counter[str]] = defaultdict(Counter)
    memory: Counter[str] = Counter()
    forwards = 0
    for name in sorted(results):
        path = root / "trace" / args.method / f"{name}.csv"
        if not path.exists():
            continue
        sequence = dataset.sequence(name)
        truthBox, present = sequence.groundTruth("bbox")
        truthBfov, _ = sequence.groundTruth("bfov")
        frameWidthPx = sequence.frameSize[0]
        sequence.close()
        # A run cut short with --max-frames has fewer results than the sequence has frames.
        count = len(results[name])
        truthBox, truthBfov = truthBox[:count], truthBfov[:count]
        present = np.asarray(present, dtype=bool)[:count]
        iou = np.zeros(len(truthBox))
        iou[present] = dualIou(truthBox[present], results[name][present], frameWidthPx)
        lost = lostFrameMask(iou, present)
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            index = int(row["frame"])
            state = row["modeAfter"]
            forwards += int(row["forwards"] or 0)
            memory[row["memory"] or "learned"] += 1
            if present[index]:
                stateIou[state].append(float(iou[index]))
            else:
                absent[state] += 1
            entered = row["modeBefore"] == "TRACKING" and state != "TRACKING"
            if entered and present[index]:
                # Right when the box at this frame or within the next frames is bad.
                ahead = iou[index : index + 5][present[index : index + 5]]
                doubts[row["reason"]].append(bool(ahead.size and ahead.min() < GOOD_IOU))
            candidates = json.loads(row["candidates"] or "[]")
            for candidate in candidates:
                onTarget = False
                if present[index]:
                    truth = _direction(truthBfov[index, 0], truthBfov[index, 1])
                    place = _direction(candidate["yawDeg"], candidate["pitchDeg"])
                    angle = np.degrees(np.arccos(np.clip(float(truth @ place), -1.0, 1.0)))
                    size = float(np.sqrt(max(truthBfov[index, 2] * truthBfov[index, 3], 1e-6)))
                    onTarget = angle < 0.5 * max(size, float(candidate["sizeDeg"]))
                verdicts["on target" if onTarget else "elsewhere"][candidate["verdict"]] += 1
            if row["action"] == "jump":
                before = iou[max(0, index - 5) : index][present[max(0, index - 5) : index]]
                after = iou[index : index + AFTER_FRAMES][present[index : index + AFTER_FRAMES]]
                jumps.append(
                    {
                        "sequence": name,
                        "frame": index,
                        "present": bool(present[index]),
                        "iouBefore": float(before.mean()) if before.size else None,
                        "iouAfter": float(after.mean()) if after.size else None,
                        "onTarget": bool(after.size and after.mean() >= LOST_IOU),
                        "leftGoodBox": bool(before.size and before.mean() >= GOOD_IOU),
                    }
                )
            frames.append(
                {
                    "sequence": name,
                    **{key: value for key, value in row.items() if key != "candidates"},
                    "present": int(present[index]),
                    "iou": round(float(iou[index]), 6),
                    "lostRun": int(lost[index]),
                    "candidateCount": len(candidates),
                }
            )
    total = sum(len(values) for values in stateIou.values())
    states = {}
    for state in STATES:
        values = np.asarray(stateIou.get(state, []))
        if not values.size and not absent[state]:
            continue
        states[state] = {
            "frames": int(values.size),
            "share": float(values.size / max(1, total)),
            "meanIou": float(values.mean()) if values.size else None,
            "good": float((values >= GOOD_IOU).mean()) if values.size else None,
            "lost": float((values < LOST_IOU).mean()) if values.size else None,
            "targetAbsent": int(absent[state]),
        }
    allIou = np.concatenate([np.asarray(v) for v in stateIou.values()]) if total else np.zeros(0)
    labels = np.concatenate(
        [np.full(len(v), s != "TRACKING") for s, v in stateIou.items()]
    ) if total else np.zeros(0, bool)
    summary = {
        "states": states,
        "goodFramesDoubted": float(labels[allIou >= GOOD_IOU].mean()) if total else None,
        "lostFramesDoubted": float(labels[allIou < LOST_IOU].mean()) if total else None,
        "doubts": {
            reason: {"count": len(hits), "right": float(np.mean(hits))}
            for reason, hits in sorted(doubts.items())
        },
        "jumps": {
            "count": len(jumps),
            "onTarget": sum(1 for jump in jumps if jump["onTarget"]),
            "leftGoodBox": sum(1 for jump in jumps if jump["leftGoodBox"]),
            "events": jumps,
        },
        "candidates": {group: dict(counts) for group, counts in verdicts.items()},
        "memory": dict(memory),
        "forwardsPerFrame": forwards / max(1, len(frames)),
    }
    return frames, summary


def report(summary: dict[str, object]) -> None:
    print(
        f"  {'state':<11}{'frames':>8}{'share':>8}{'mean IoU':>10}{'good':>8}{'lost':>8}"
        f"{'absent':>8}"
    )
    for state, row in summary["states"].items():
        if row["frames"]:
            print(
                f"  {state:<11}{row['frames']:>8}{row['share']:>8.1%}{row['meanIou']:>10.3f}"
                f"{row['good']:>8.1%}{row['lost']:>8.1%}{row['targetAbsent']:>8}"
            )
        else:
            print(f"  {state:<11}{0:>8}{'':>34}{row['targetAbsent']:>8}")
    if summary["goodFramesDoubted"] is not None:
        print(
            f"good frames not in TRACKING: {summary['goodFramesDoubted']:.1%}; "
            f"lost frames not in TRACKING: {summary['lostFramesDoubted']:.1%}"
        )
    print("doubts raised from TRACKING (right: the box is bad within five frames)")
    for reason, row in summary["doubts"].items():
        print(f"  {reason:<28}{row['count']:>6}{row['right']:>8.1%}")
    jumps = summary["jumps"]
    print(
        f"jumps: {jumps['count']}, on target {jumps['onTarget']}, "
        f"left a good box {jumps['leftGoodBox']}"
    )
    for group, counts in summary["candidates"].items():
        cells = ", ".join(f"{key} {value}" for key, value in sorted(counts.items()))
        print(f"candidates {group}: {cells}")
    print(f"tracker memory per frame: {summary['memory']}")
    print(f"forwards per frame: {summary['forwardsPerFrame']:.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", default="ours")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    try:
        frames, summary = analyse(args)
    except (Track360Error, ValueError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    if not frames:
        print("no state trace found: was the run made with loss handling on?", file=sys.stderr)
        return 2
    report(summary)
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(frames[0]))
            writer.writeheader()
            writer.writerows(frames)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
