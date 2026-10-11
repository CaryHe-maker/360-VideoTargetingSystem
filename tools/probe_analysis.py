"""Compare the signals a probed run recorded as detectors of a lost target.

    python tools/probe_analysis.py --dataset-root <dir> --label-root <dir> \\
        --fit outputs/E016_tune --check outputs/E016_holdout

Each signal is judged on the frames where the target is present: how well it separates
lost frames from good ones, and how many lost frames it flags when it may flag only 5%
of the good ones.  A logistic combination of signals is fitted on ``--fit`` and
evaluated on ``--check`` without refitting.  Needs the ``probe/`` and ``score/`` files
of ``tools/benchmark.py run --probe``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from track360.core.errors import Track360Error
from track360.datasets.vot360 import Vot360Dataset
from track360.evaluation.appearance_probe import PROBE_DIRECTORY, SIGNALS
from track360.evaluation.score_analysis import auroc, buildFrameTable, spearman
from track360.evaluation.vot360_metrics import loadTrackerResults
from track360.io.vot360_results import BBOX_DIRECTORY, SCORE_DIRECTORY, readScoreFile

HIGH_SCORE = 0.75
GOOD_FLAG_BUDGET = 0.05
SMOOTH_FRAMES = 5
COMBINATIONS = {
    "score+dinov2": ("score", "dinov2"),
    "score+dinov2+hist": ("score", "dinov2", "hist"),
    "score+clean+cleanIou": ("score", "clean", "cleanIou"),
    "all": ("score", "dinov2", "dino", "resnet18", "hist", "clean", "cleanIou"),
}


def loadRun(datasetRoot: str, labelRoot: str | None, outputRoot: Path, method: str) -> dict:
    """Per-frame IoU, loss flag and every signal of one run, on target-present frames."""
    dataset = Vot360Dataset(datasetRoot, labelRoot)
    results = loadTrackerResults(outputRoot / BBOX_DIRECTORY / method)
    scores = {
        path.stem: readScoreFile(path)
        for path in sorted((outputRoot / SCORE_DIRECTORY / method).glob("*.txt"))
    }
    groundTruth = {}
    for name in results:
        sequence = dataset.sequence(name)
        groundTruth[name] = sequence.groundTruth(BBOX_DIRECTORY)
        sequence.close()
    frameWidthPx = dataset.sequence(next(iter(results))).frameSize[0]
    table = buildFrameTable(groundTruth, results, scores, frameWidthPx)
    signals = {"score": table.score}
    for signal in SIGNALS:
        root = outputRoot / PROBE_DIRECTORY / method / signal
        if not root.is_dir():
            continue
        perSequence = {path.stem: np.loadtxt(path, ndmin=1) for path in root.glob("*.txt")}
        signals[signal] = np.asarray(
            [
                perSequence[str(name)][frame] if str(name) in perSequence else np.nan
                for name, frame in zip(table.sequence, table.frame, strict=True)
            ]
        )
    for signal in [name for name in signals if name != "cleanIou"]:
        signals[f"{signal}~{SMOOTH_FRAMES}"] = _causalMean(
            signals[signal], table.sequence, table.frame
        )
    return {"table": table, "signals": signals}


def _causalMean(values: np.ndarray, sequence: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Mean of each frame's value and the frames just before it, within one sequence."""
    smoothed = np.full(len(values), np.nan)
    for name in np.unique(sequence):
        index = np.flatnonzero(sequence == name)
        index = index[np.argsort(frame[index])]
        series = values[index]
        for position in range(len(index)):
            window = series[max(0, position - SMOOTH_FRAMES + 1) : position + 1]
            window = window[~np.isnan(window)]
            if window.size:
                smoothed[index[position]] = window.mean()
    return smoothed


def judge(values: np.ndarray, run: dict) -> dict:
    table = run["table"]
    valid = ~np.isnan(values)
    good, lost = table.good & valid, table.lost & valid
    highLost = lost & (run["signals"]["score"] >= HIGH_SCORE)
    row = {
        "frames": int(valid.sum()),
        "aurocLostVsGood": auroc(-values[good | lost], lost[good | lost]),
        "spearman": spearman(values[valid], table.iou[valid]),
        "goodMedian": float(np.median(values[good])) if good.any() else None,
        "lostMedian": float(np.median(values[lost])) if lost.any() else None,
    }
    if good.any() and lost.any():
        threshold = float(np.quantile(values[good], GOOD_FLAG_BUDGET))
        row["threshold"] = threshold
        row["lostCaught"] = float((values[lost] < threshold).mean())
        row["highScoreLostCaught"] = (
            float((values[highLost] < threshold).mean()) if highLost.any() else None
        )
    return row


def fitLogistic(features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plain logistic regression on standardized features; returns weights and scaling."""
    mean, std = features.mean(axis=0), features.std(axis=0) + 1e-9
    x = np.column_stack([(features - mean) / std, np.ones(len(features))])
    weights = np.zeros(x.shape[1])
    for _ in range(50):
        probability = 1.0 / (1.0 + np.exp(-x @ weights))
        gradient = x.T @ (probability - target) / len(target)
        curvature = (x * (probability * (1.0 - probability))[:, None]).T @ x / len(target)
        weights -= np.linalg.solve(curvature + 1e-6 * np.eye(x.shape[1]), gradient)
    return weights, np.vstack([mean, std])


def applyLogistic(features: np.ndarray, weights: np.ndarray, scaling: np.ndarray) -> np.ndarray:
    x = np.column_stack([(features - scaling[0]) / scaling[1], np.ones(len(features))])
    return 1.0 / (1.0 + np.exp(-x @ weights))


def combination(names: tuple[str, ...], fit: dict, runs: dict[str, dict]) -> dict | None:
    """Fit "is this frame good?" on the good and lost frames of ``fit``; score every run."""
    if any(name not in fit["signals"] for name in names):
        return None
    features = np.column_stack([fit["signals"][name] for name in names])
    table = fit["table"]
    keep = ~np.isnan(features).any(axis=1) & (table.good | table.lost)
    weights, scaling = fitLogistic(features[keep], table.good[keep].astype(float))
    output = {"weights": dict(zip([*names, "bias"], weights.tolist(), strict=True)), "runs": {}}
    for label, run in runs.items():
        stacked = np.column_stack([run["signals"][name] for name in names])
        values = np.full(len(stacked), np.nan)
        valid = ~np.isnan(stacked).any(axis=1)
        values[valid] = applyLogistic(stacked[valid], weights, scaling)
        output["runs"][label] = judge(values, run)
    return output


def formatRows(title: str, rows: dict[str, dict]) -> str:
    lines = [
        title,
        f"  {'signal':<24}{'AUROC':>7}{'Spearman':>10}{'good med':>10}{'lost med':>10}"
        f"{'lost caught':>13}{'high-score lost caught':>24}",
    ]
    for name, row in rows.items():
        def cell(key: str, width: int, digits: int = 3) -> str:
            value = row.get(key)
            return f"{'-':>{width}}" if value is None else f"{value:>{width}.{digits}f}"

        lines.append(
            f"  {name:<24}{cell('aurocLostVsGood', 7)}{cell('spearman', 10)}"
            f"{cell('goodMedian', 10)}{cell('lostMedian', 10)}"
            f"{cell('lostCaught', 13)}{cell('highScoreLostCaught', 24)}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument(
        "--fit", type=Path, required=True, help="run the combinations are fitted on"
    )
    parser.add_argument("--check", type=Path, default=None, help="run they are checked on")
    parser.add_argument("--method", default="ours")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        runs = {"fit": loadRun(args.dataset_root, args.label_root, args.fit, args.method)}
        if args.check is not None:
            runs["check"] = loadRun(args.dataset_root, args.label_root, args.check, args.method)
    except (Track360Error, ValueError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    payload: dict[str, object] = {"budget": GOOD_FLAG_BUDGET, "highScore": HIGH_SCORE}
    for label, run in runs.items():
        table = run["table"]
        highLost = table.lost & (run["signals"]["score"] >= HIGH_SCORE)
        rows = {name: judge(values, run) for name, values in run["signals"].items()}
        payload[label] = {
            "frames": len(table),
            "lostShare": float(table.lost.mean()),
            "highScoreLostShareOfLost": float(highLost.sum() / max(1, table.lost.sum())),
            "signals": rows,
        }
        root = args.fit if label == "fit" else args.check
        print(
            formatRows(
                f"{label}: {root}  ({len(table)} frames, lost {table.lost.mean():.3f}, of which "
                f"score >= {HIGH_SCORE}: {highLost.sum() / max(1, table.lost.sum()):.3f}; "
                f"'caught' = flagged while flagging {GOOD_FLAG_BUDGET:.0%} of good frames)",
                rows,
            )
        )
        print()
    combined = {}
    for name, members in COMBINATIONS.items():
        result = combination(members, runs["fit"], runs)
        if result is not None:
            combined[name] = result
    for label in runs:
        print(
            formatRows(
                f"logistic combinations fitted on 'fit', evaluated on '{label}'",
                {name: result["runs"][label] for name, result in combined.items()},
            )
        )
        print()
    payload["combinations"] = combined
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
