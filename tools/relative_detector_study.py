"""Loss detectors built from relative quantities only, judged for how well they carry over.

    python tools/relative_detector_study.py --table outputs/E018/tune_freerun.csv \\
        --table outputs/E018/holdout_freerun.csv --events outputs/E019/loss_events.csv \\
        --sequence-file configs/splits/360vos_dev_train.txt --out outputs/E027

Works on the per-frame tables of ``tools/fusion_freerun.py``; the tracker is not run.
Only the sequences of ``--sequence-file`` are read.

A relative quantity compares a score with what the same sequence showed before, so it
has no unit and no level of its own:

    start      score / its median over the first frames of the sequence - 1
    baseline   score / the median of the frames trusted so far - 1 (a frame is trusted
               while it is not far below that median)
    rank       the share of the trusted earlier frames the score is not below
    trend      fast running mean / slow running mean - 1

Each is computed for the backend score and for the similarity to the frame-0 template.
A detector is one of them, or the plain mean of the two scores' values: nothing is
fitted.  The absolute detectors in use are listed for comparison.

Every detector is judged the way it would be deployed: the threshold for a false-alarm
target is taken from some sequences and applied to others (folds of whole sequences).
Reported are the false alarms it then really has and the lost frames it then finds, and
how much the real false-alarm rate varies from fold to fold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_analysis import Detector, causalDeviation, loadTable, lossParts  # noqa: E402

RATES = (0.01, 0.02, 0.05)
FOLDS = 11
START_FRAMES = 30
WARM_FRAMES = 10
RANK_GATE = 0.05
RATIO_GATE = 0.15
CHANNELS = ("backend", "template")


def ratioToStart(values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values))
    for index in range(1, len(values)):
        reference = float(np.median(values[: min(index, START_FRAMES)]))
        result[index] = values[index] / max(reference, 0.05) - 1.0
    return result


def selfRank(values: np.ndarray) -> np.ndarray:
    """Share of the trusted earlier frames this frame is at least as good as."""
    history: list[float] = []
    result = np.ones(len(values))
    for index, value in enumerate(values):
        if len(history) < WARM_FRAMES:
            history.append(float(value))
            continue
        result[index] = float(np.mean(np.asarray(history) <= value))
        # A frame at the very bottom of its own history is not added to it.
        if result[index] >= RANK_GATE:
            history.append(float(value))
    return result


def trend(values: np.ndarray) -> np.ndarray:
    fast = slow = float(values[0]) if len(values) else 0.0
    result = np.zeros(len(values))
    for index, value in enumerate(values):
        fast += 0.5 * (value - fast)
        slow += (2.0 / 31.0) * (value - slow)
        result[index] = fast / max(slow, 0.05) - 1.0
    return result


def smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Mean over the last frames: one odd frame does not raise a flag."""
    result = np.zeros(len(values))
    for index in range(len(values)):
        result[index] = values[max(0, index - window + 1) : index + 1].mean()
    return result


QUANTITIES = {
    "start": ratioToStart,
    "baseline": lambda values: causalDeviation(
        values, "ratio", 30, WARM_FRAMES, RATIO_GATE, 0.0
    ),
    "rank": selfRank,
    "trend": trend,
}


def detectors(table) -> dict[str, tuple[dict[str, np.ndarray], float | None]]:
    """Signals of every detector and the margin above the threshold that releases a latch."""
    items: dict[str, tuple[dict[str, np.ndarray], float | None]] = {
        "absolute: fused": ({name: seq["fused"] for name, seq in table.items()}, 0.2),
        "absolute: backend": ({name: seq["backend"] for name, seq in table.items()}, 0.2),
        "absolute: template": ({name: seq["template"] for name, seq in table.items()}, 0.2),
    }
    for label, function in QUANTITIES.items():
        margin = 0.2 if label == "rank" else None
        perChannel = {
            channel: {name: function(seq[channel]) for name, seq in table.items()}
            for channel in CHANNELS
        }
        for channel in CHANNELS:
            items[f"{label}: {channel}"] = (perChannel[channel], margin)
        both = {
            name: 0.5 * (perChannel["backend"][name] + perChannel["template"][name])
            for name in table
        }
        items[f"{label}: mean of both"] = (both, margin)
        items[f"{label}: mean of both, 3 frames"] = (
            {name: smooth(signal) for name, signal in both.items()},
            margin,
        )
    return items


def deployed(table, parts, signals, margin, latched: bool, folds, rate: float) -> dict:
    """Threshold from the other folds, applied to each fold in turn."""
    hits = {"good": 0, "goodFlag": 0, "lost": 0, "lostFlag": 0, "early": 0, "earlyFlag": 0}
    perFold = []
    for held in folds:
        rest = {name: table[name] for name in table if name not in held}
        fit = Detector({name: signals[name] for name in rest}, latched, margin)
        threshold = fit.threshold(rest, rate)
        flags = Detector({name: signals[name] for name in held}, latched, margin).flags(threshold)
        good = flagged = 0
        for name in held:
            sequence = table[name]
            good += int(sequence["good"].sum())
            flagged += int(flags[name][sequence["good"]].sum())
            hits["lost"] += int(sequence["lost"].sum())
            hits["lostFlag"] += int(flags[name][sequence["lost"]].sum())
            early = parts[name].get("early")
            if early is not None:
                hits["early"] += int(early.sum())
                hits["earlyFlag"] += int(flags[name][early].sum())
        hits["good"] += good
        hits["goodFlag"] += flagged
        if good >= 100:
            perFold.append(flagged / good)
    return {
        "falseAlarm": hits["goodFlag"] / max(1, hits["good"]),
        "detection": hits["lostFlag"] / max(1, hits["lost"]),
        "early": hits["earlyFlag"] / max(1, hits["early"]),
        "foldFalseAlarmMax": float(np.max(perFold)) if perFold else 0.0,
        "foldFalseAlarmMedian": float(np.median(perFold)) if perFold else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--table", type=Path, action="append", required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    wanted = {
        line.strip()
        for line in args.sequence_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    table = {}
    for path in args.table:
        table.update({name: seq for name, seq in loadTable(path).items() if name in wanted})
    missing = wanted - set(table)
    if missing:
        print(f"sequences without rows: {sorted(missing)}", file=sys.stderr)
        return 2
    parts = lossParts(table, args.events)
    order = sorted(table)
    np.random.default_rng(0).shuffle(order)
    folds = [order[index::FOLDS] for index in range(FOLDS)]
    good = sum(int(seq["good"].sum()) for seq in table.values())
    lost = sum(int(seq["lost"].sum()) for seq in table.values())
    print(f"{len(table)} sequences, {good} good frames, {lost} lost frames, {FOLDS} folds")
    payload: dict[str, object] = {"sequences": sorted(table), "folds": folds, "detectors": {}}
    items = detectors(table)
    for rate in RATES:
        key = f"{rate:.0%}"
        print(f"\nfalse-alarm target {key}: what the threshold of other sequences gives")
        print(
            f"  {'detector':<40}{'FA':>7}{'worst fold':>12}{'detected':>10}{'early':>8}"
        )
        for name, (signals, margin) in items.items():
            for latched in (False, True):
                row = deployed(table, parts, signals, margin, latched, folds, rate)
                label = name + (", latched" if latched else "")
                payload["detectors"].setdefault(label, {})[key] = row
                print(
                    f"  {label:<40}{row['falseAlarm']:>7.1%}{row['foldFalseAlarmMax']:>12.1%}"
                    f"{row['detection']:>10.1%}{row['early']:>8.1%}"
                )
    args.out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1) + "\n"
    (args.out / "relative.json").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
