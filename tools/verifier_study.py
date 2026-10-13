"""Can a small model on the backend's own token judge a box better than the score does?

    python tools/verifier_study.py --dataset-root <dir> --label-root <dir> \\
        --signals outputs/E045 --features outputs/E025/features_tune.csv \\
        --features outputs/E025/features_holdout.csv --json outputs/E045/study.json

Works on what ``tools/backend_signals.py`` recorded; the tracker is not run.

Part 1, the tracked box (``<signals>/freerun``): tell lost frames from good ones.  The
baseline is what the system has today, the backend score and the template similarity
with their smoothed forms (the seven features of evaluation-log E025, read from its
feature tables).  Added to it: the coordinate distributions' entropy, peak and gap, and
a logistic model on the 768-value score token.

Part 2, search candidates (``<signals>/candidates.npz``): tell a stateless pass that
boxed the target (IoU >= 0.5) from one that boxed something else (IoU < 0.1).  The
baseline is the score of that pass.

Every model is fitted on three quarters of the sequences and judged on the rest, so no
sequence is in both.  Reported: AUROC, and the share of lost frames (or of true
candidates) caught when 2% and 5% of the good frames (or of false candidates) are let
through wrongly.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np

from track360.datasets.vot360 import Vot360Dataset
from track360.evaluation.loss_rate import dualIou, lostFrameMask
from track360.evaluation.vot360_metrics import loadTrackerResults

E025_FEATURES = (
    "backendMean5",
    "template",
    "templateEmaGap",
    "backend",
    "backendEmaGap",
    "templateStd10",
    "backendCusum",
)
RATES = (0.02, 0.05)
FOLDS = 4
RIDGES = (1.0, 10.0, 100.0)


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Probability that a positive scores above a negative (ties count half)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average the ranks of equal scores.
    values, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(values))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    pos = int(positive.sum())
    neg = len(positive) - pos
    if not pos or not neg:
        return float("nan")
    return float((ranks[positive].sum() - pos * (pos + 1) / 2) / (pos * neg))


def caught(scores: np.ndarray, positive: np.ndarray, rate: float) -> float:
    """Share of positives above the threshold that lets ``rate`` of the negatives through."""
    threshold = np.quantile(scores[~positive], 1.0 - rate)
    return float((scores[positive] > threshold).mean())


def fitLogistic(matrix: np.ndarray, positive: np.ndarray, ridge: float):
    """Class-balanced logistic regression with an L2 penalty; returns a scoring function."""
    mean, spread = matrix.mean(axis=0), matrix.std(axis=0) + 1e-6
    design = np.column_stack([np.ones(len(matrix)), (matrix - mean) / spread])
    weight = np.where(positive, 0.5 / positive.sum(), 0.5 / (~positive).sum()) * len(positive)
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    beta = np.zeros(design.shape[1])
    for _ in range(25):
        probability = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30, 30)))
        gradient = design.T @ (weight * (probability - positive)) + penalty @ beta
        curvature = (design * (weight * probability * (1 - probability))[:, None]).T @ design
        step = np.linalg.solve(curvature + penalty + 1e-6 * np.eye(len(beta)), gradient)
        beta -= step
        if float(np.abs(step).max()) < 1e-6:
            break

    def score(other: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(other)), (other - mean) / spread]) @ beta

    return score


def folds(sequences: np.ndarray, seed: int = 0) -> list[np.ndarray]:
    names = np.unique(sequences)
    np.random.default_rng(seed).shuffle(names)
    return [np.isin(sequences, names[index::FOLDS]) for index in range(FOLDS)]


def crossValidated(
    matrix: np.ndarray, positive: np.ndarray, sequences: np.ndarray, ridge: float
) -> np.ndarray:
    """Scores of every row from a model that did not see its sequence."""
    scores = np.zeros(len(positive))
    for held in folds(sequences):
        scores[held] = fitLogistic(matrix[~held], positive[~held], ridge)(matrix[held])
    return scores


def judge(label: str, scores: np.ndarray, positive: np.ndarray) -> dict[str, float]:
    entry = {"auroc": auroc(scores, positive)}
    for rate in RATES:
        entry[f"caught@{rate:g}"] = caught(scores, positive, rate)
    print(
        f"  {label:<46}{entry['auroc']:>7.3f}"
        + "".join(f"{entry[f'caught@{rate:g}']:>10.1%}" for rate in RATES)
    )
    return entry


def header(title: str, positives: str) -> None:
    print(f"\n{title}")
    print(
        f"  {'model':<46}{'AUROC':>7}"
        + "".join(f"{f'{positives} @{rate:.0%}':>10}" for rate in RATES)
    )


def trackedBox(args: argparse.Namespace) -> dict[str, object]:
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    boxes = loadTrackerResults(args.signals / "run" / "bbox" / "ours")
    old: dict[tuple[str, int], dict[str, str]] = {}
    for path in args.features or []:
        with path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                old[(row["sequence"], int(row["frame"]))] = row
    columns: dict[str, list] = {key: [] for key in ("sequence", "bad", "keep", "score")}
    scalars, tokens, e025 = [], [], []
    for path in sorted((args.signals / "freerun").glob("*.npz")):
        name = path.stem
        data = np.load(path)
        sequence = dataset.sequence(name)
        truthBox, present = sequence.groundTruth("bbox")
        width = sequence.frameSize[0]
        sequence.close()
        present = np.asarray(present, dtype=bool)
        iou = np.zeros(len(present))
        iou[present] = dualIou(truthBox[present], boxes[name][present], width)
        lost = lostFrameMask(iou, present)
        good = present & (iou >= 0.5)
        # Row i of the recording is frame i + 1.
        frames = np.arange(1, len(present))
        columns["sequence"] += [name] * len(frames)
        columns["bad"] += list(lost[frames])
        columns["keep"] += list(lost[frames] | good[frames])
        columns["score"] += list(data["score"])
        scalars.append(
            np.column_stack(
                [
                    data["coordEntropy"].mean(axis=1),
                    data["coordEntropy"].max(axis=1),
                    data["coordPeak"].min(axis=1),
                    data["hardSoftGap"].max(axis=1),
                ]
            )
        )
        tokens.append(data["scoreFeature"].astype(np.float32))
        e025.append(
            np.asarray(
                [
                    [float(old[(name, int(frame))][key]) for key in E025_FEATURES]
                    if (name, int(frame)) in old
                    else [np.nan] * len(E025_FEATURES)
                    for frame in frames
                ]
            )
        )
    keep = np.asarray(columns["keep"], dtype=bool)
    sequences = np.asarray(columns["sequence"])[keep]
    # "Positive" is a good frame, so a higher model score means "tracked", as the score.
    tracked = ~np.asarray(columns["bad"], dtype=bool)[keep]
    score = np.asarray(columns["score"], dtype=np.float64)[keep]
    scalar = np.concatenate(scalars)[keep]
    token = np.concatenate(tokens)[keep]
    table = np.concatenate(e025)[keep]
    known = ~np.isnan(table).any(axis=1)
    print(
        f"tracked box: {int(tracked.sum())} good frames, {int((~tracked).sum())} lost frames,"
        f" {len(np.unique(sequences))} sequences; E025 features for {known.mean():.1%} of them"
    )
    result: dict[str, object] = {}

    def lostCaught(label: str, scores: np.ndarray, rows: np.ndarray | None = None) -> None:
        part = slice(None) if rows is None else rows
        # Lost frames are the ones to catch: flip the sign so that they score high.
        result[label] = judge(label, -scores[part], ~tracked[part])

    header("part 1: is the tracked box wrong? (all frames)", "lost")
    lostCaught("backend score", score)
    for index, name in enumerate(("entropy mean", "entropy max", "peak min", "gap max")):
        sign = -1.0 if name.startswith(("entropy", "gap")) else 1.0
        lostCaught(f"coordinate {name}", sign * scalar[:, index])
    lostCaught(
        "score + coordinate scalars",
        crossValidated(np.column_stack([score, scalar]), tracked, sequences, 1.0),
    )
    for ridge in RIDGES:
        lostCaught(
            f"score token, ridge {ridge:g}", crossValidated(token, tracked, sequences, ridge)
        )
    if known.any():
        header("part 1, frames with E025 features (score and template similarity)", "lost")
        seven = table[known]
        lostCaught("backend score", score, known)
        base = crossValidated(seven, tracked[known], sequences[known], 1.0)
        result["E025 seven"] = judge("E025 seven features", -base, ~tracked[known])
        plus = crossValidated(
            np.column_stack([seven, scalar[known]]), tracked[known], sequences[known], 1.0
        )
        result["E025 seven + scalars"] = judge(
            "E025 seven + coordinate scalars", -plus, ~tracked[known]
        )
        for ridge in RIDGES:
            both = crossValidated(
                np.column_stack([seven, token[known]]), tracked[known], sequences[known], ridge
            )
            result[f"E025 seven + token, ridge {ridge:g}"] = judge(
                f"E025 seven + score token, ridge {ridge:g}", -both, ~tracked[known]
            )
    return result


def searchCandidates(args: argparse.Namespace) -> dict[str, object]:
    data = np.load(args.signals / "candidates.npz")
    iou = data["iou"]
    keep = (iou >= 0.5) | (iou < 0.1)
    onTarget = iou[keep] >= 0.5
    sequences = data["sequence"][keep]
    kind = data["kind"][keep]
    score = data["score"][keep].astype(np.float64)
    token = data["scoreFeature"][keep].astype(np.float32)
    scalar = np.column_stack(
        [
            data["coordEntropy"].mean(axis=1),
            data["coordEntropy"].max(axis=1),
            data["coordPeak"].min(axis=1),
            data["hardSoftGap"].max(axis=1),
        ]
    )[keep]
    print(
        f"\nsearch candidates: {int(onTarget.sum())} on the target, {int((~onTarget).sum())} on"
        f" something else, {int((~keep).sum())} in between left out; by view: "
        + ", ".join(
            f"{name} {int((kind == name).sum())} ({onTarget[kind == name].mean():.0%} on target)"
            for name in ("near", "far", "tracked")
        )
    )
    result: dict[str, object] = {}
    models = {
        "pass score": score,
        "score + coordinate scalars": crossValidated(
            np.column_stack([score, scalar]), onTarget, sequences, 1.0
        ),
        **{
            f"score token, ridge {ridge:g}": crossValidated(token, onTarget, sequences, ridge)
            for ridge in RIDGES
        },
    }
    header("part 2: is the candidate the target? (all views)", "true")
    for label, scores in models.items():
        result[label] = judge(label, scores, onTarget)
    # The hard case: the box a lost track sits on, against true finds.
    hard = (kind == "tracked") | onTarget
    header("part 2, true finds against the boxes a lost track sat on", "true")
    for label, scores in models.items():
        result[f"hard: {label}"] = judge(label, scores[hard], onTarget[hard])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--features", type=Path, action="append", default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    payload: dict[str, object] = {}
    if (args.signals / "freerun").is_dir():
        payload["trackedBox"] = trackedBox(args)
    if (args.signals / "candidates.npz").is_file():
        payload["candidates"] = searchCandidates(args)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
