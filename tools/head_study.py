"""Judges for noticing a loss and for confirming a find, with overfitting as the main penalty.

    python tools/head_study.py detection --dataset-root <dir> --label-root <dir> \\
        --signals outputs/E045 --features outputs/E025/features_tune.csv \\
        --features outputs/E025/features_holdout.csv --out outputs/E049/detection
    python tools/head_study.py candidates --runs outputs/E049/runs --out outputs/E049/candidates

The protocol is the one registered in evaluation-log E049, fixed before any model was
trained.  Every quantity is split in two:

    P   is the target inside the normal-size view around the box (or candidate)?
    Q   is the box (or candidate) the target?

and judged three ways: Q alone, P alone, and the product of the two probabilities, all
against the Q label.  Each is learned by a linear model and by gradient-boosted trees.

Overfitting is measured, not assumed away: nested cross-validation by sequence (outer:
4 folds, three different splits; inner: 3 folds, for the hyper-parameters only), the gap
between training and held-out AUROC (G1), the spread of held-out AUROC (G2), and how
far a threshold set on the training part overshoots its false-alarm rate on held-out
sequences (G3).  The penalised score is held-out AUROC - max(0, G1 - 0.03) - G2.
For candidates, G4 trains on the runs of one experiment and tests on the runs of the
other, on sequences the model never saw.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUTER_FOLDS, INNER_FOLDS, SPLITS = 4, 3, (0, 1, 2)
TARGET_RATES = (0.02, 0.05)
GAP_ALLOWANCE, GAP_LIMIT, OVERSHOOT_LIMIT, MARGIN = 0.03, 0.10, 2.0, 0.02
TOKEN_COMPONENTS = 16
E025_SEVEN = (
    "backendMean5",
    "template",
    "templateEmaGap",
    "backend",
    "backendEmaGap",
    "templateStd10",
    "backendCusum",
)


# ---------------------------------------------------------------- models


class Clip(BaseEstimator, TransformerMixin):
    """Clip each column to the 0.5-99.5 percentiles of the training part."""

    def fit(self, x, y=None):
        self.low_, self.high_ = np.percentile(x, [0.5, 99.5], axis=0)
        return self

    def transform(self, x):
        return np.clip(x, self.low_, self.high_)


class TokenPca(BaseEstimator, TransformerMixin):
    """Keep the scalar columns; replace the token columns by their first components."""

    def __init__(self, scalars: int = 0, components: int = TOKEN_COMPONENTS):
        self.scalars = scalars
        self.components = components

    def fit(self, x, y=None):
        token = x[:, self.scalars :]
        self.scale_ = StandardScaler().fit(token)
        self.pca_ = PCA(self.components, random_state=0).fit(self.scale_.transform(token))
        return self

    def transform(self, x):
        token = self.pca_.transform(self.scale_.transform(x[:, self.scalars :]))
        return np.column_stack([x[:, : self.scalars], token])


def linear(strength: float) -> Pipeline:
    return Pipeline(
        [
            ("clip", Clip()),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(C=strength, class_weight="balanced", max_iter=3000),
            ),
        ]
    )


def trees(scalars: int | None, **params: Any) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if scalars is not None:
        steps.append(("pca", TokenPca(scalars)))
    steps.append(
        (
            "model",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_features=0.5,
                early_stopping=False,
                class_weight="balanced",
                random_state=0,
                **params,
            ),
        )
    )
    return Pipeline(steps)


LINEAR_GRID = [{"strength": value} for value in (1.0, 0.1, 0.01, 0.001)]
# The stronger grids of the second round (E049, after the first round overfitted).
STRONG_LINEAR_GRID = [{"strength": value} for value in (1e-3, 1e-4, 1e-5)]
STRONG_TREE_GRID = [
    {"max_leaf_nodes": leaves, "min_samples_leaf": leaf, "l2_regularization": l2, "max_iter": it}
    for leaves, leaf, l2, it in product((3, 4), (500, 1500), (10.0, 100.0), (50, 150))
]
TREE_GRID = [
    {"max_leaf_nodes": leaves, "min_samples_leaf": leaf, "l2_regularization": l2, "max_iter": it}
    for leaves, leaf, l2, it in product((4, 8), (50, 200), (1.0, 10.0), (100, 300))
]

GRIDS = {"first": (LINEAR_GRID, TREE_GRID), "strong": (STRONG_LINEAR_GRID, STRONG_TREE_GRID)}


@dataclass
class Family:
    name: str
    columns: str  # "scalars" or "scalars+token"
    kind: str  # "linear" or "trees"

    def candidates(self, scalars: int) -> list[tuple[dict, Any]]:
        linearGrid, treeGrid = GRIDS[GRID["name"]]
        if self.kind == "linear":
            return [(params, linear(**params)) for params in linearGrid]
        token = scalars if self.columns == "scalars+token" else None
        return [(params, trees(token, **params)) for params in treeGrid]


GRID = {"name": "first"}


# ---------------------------------------------------------------- evaluation


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def splits(groups: np.ndarray, seed: int, folds: int):
    return GroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(
        np.zeros(len(groups)), groups=groups
    )


def chooseAndFit(family: Family, x, y, groups, scalars: int, seed: int):
    """Pick hyper-parameters by inner cross-validation on this training part, then refit."""
    options = family.candidates(scalars)
    if len(options) == 1:
        params, model = options[0]
        return params, clone(model).fit(x, y)
    scores = []
    for params, model in options:
        values = []
        for train, valid in splits(groups, seed + 100, INNER_FOLDS):
            fitted = clone(model).fit(x[train], y[train])
            values.append(auc(y[valid], fitted.predict_proba(x[valid])[:, 1]))
        scores.append(np.nanmean(values))
    best = int(np.nanargmax(scores))
    params, model = options[best]
    return params, clone(model).fit(x, y)


def thresholdAt(scores: np.ndarray, negative: np.ndarray, rate: float) -> float:
    return float(np.quantile(scores[negative], 1.0 - rate))


def rates(scores, positive, threshold) -> tuple[float, float]:
    """Share of positives above the threshold, and of negatives above it."""
    return float((scores[positive] > threshold).mean()), float(
        (scores[~positive] > threshold).mean()
    )


def summarise(records: list[dict]) -> dict:
    held = np.array([r["heldAuc"] for r in records])
    train = np.array([r["trainAuc"] for r in records])
    gap = float(np.nanmean(train - held))
    spread = float(np.nanstd(held, ddof=1))
    result = {
        "heldAuc": float(np.nanmean(held)),
        "trainAuc": float(np.nanmean(train)),
        "G1": gap,
        "G2": spread,
        "penalised": float(np.nanmean(held)) - max(0.0, gap - GAP_ALLOWANCE) - spread,
    }
    for rate in TARGET_RATES:
        result[f"recall@{rate:g}"] = float(np.nanmean([r[f"recall@{rate:g}"] for r in records]))
        result[f"falseAlarm@{rate:g}"] = float(
            np.nanmean([r[f"falseAlarm@{rate:g}"] for r in records])
        )
    result["G3"] = result["falseAlarm@0.05"] / 0.05
    result["params"] = [r["params"] for r in records]
    return result


# What a judge is for: noticing a loss means catching the negatives (lost frames) while
# flagging few positives (good frames); confirming a find means catching the positives
# (true candidates) while letting few negatives through.  Set by ``runSide``.
CATCH = {"negatives": False}


def evaluate(
    label: str,
    target: np.ndarray,
    groups: np.ndarray,
    predict,  # (train_index, held_index, seed) -> (train_scores, held_scores, params)
) -> tuple[dict, np.ndarray]:
    """Run the outer folds of all splits; return the summary and split-0 held-out scores.

    The threshold for each false-alarm rate is set on the training part and applied to
    the held-out sequences unchanged: the realised false-alarm rate there is G3.
    """
    records = []
    first = np.full(len(target), np.nan)
    for seed in SPLITS:
        for train, held in splits(groups, seed, OUTER_FOLDS):
            trainScores, heldScores, params = predict(train, held, seed)
            record = {
                "trainAuc": auc(target[train], trainScores),
                "heldAuc": auc(target[held], heldScores),
                "params": params,
            }
            for rate in TARGET_RATES:
                if CATCH["negatives"]:
                    # Flag the frames scored below the lowest ``rate`` of the good ones.
                    threshold = float(np.quantile(trainScores[target[train]], rate))
                    heldLow = heldScores < threshold
                    recall = float(heldLow[~target[held]].mean())
                    falseAlarm = float(heldLow[target[held]].mean())
                else:
                    threshold = thresholdAt(trainScores, ~target[train], rate)
                    recall, falseAlarm = rates(heldScores, target[held], threshold)
                record[f"recall@{rate:g}"], record[f"falseAlarm@{rate:g}"] = recall, falseAlarm
            records.append(record)
            if seed == SPLITS[0]:
                first[held] = heldScores
    summary = summarise(records)
    print(
        f"  {label:<44}{summary['heldAuc']:>7.3f}{summary['trainAuc']:>7.3f}{summary['G1']:>7.3f}"
        f"{summary['G2']:>7.3f}{summary['G3']:>7.2f}{summary['recall@0.02']:>8.1%}"
        f"{summary['recall@0.05']:>8.1%}{summary['penalised']:>9.3f}",
        flush=True,
    )
    return summary, first


def header(title: str) -> None:
    print(f"\n{title}")
    print(
        f"  {'judge':<44}{'held':>7}{'train':>7}{'G1':>7}{'G2':>7}{'G3':>7}"
        f"{'rec@2%':>8}{'rec@5%':>8}{'penal.':>9}"
    )


def bootstrapGain(target, groups, scores, reference, samples=2000, seed=0) -> dict:
    """Held-out AUROC gain over the reference, resampling whole sequences."""
    rng = np.random.default_rng(seed)
    names = np.unique(groups)
    index = {name: np.flatnonzero(groups == name) for name in names}
    gains = []
    for _ in range(samples):
        rows = np.concatenate([index[name] for name in rng.choice(names, len(names))])
        gains.append(auc(target[rows], scores[rows]) - auc(target[rows], reference[rows]))
    gains = np.array(gains)
    return {
        "gain": auc(target, scores) - auc(target, reference),
        "low": float(np.nanpercentile(gains, 2.5)),
        "high": float(np.nanpercentile(gains, 97.5)),
    }


def verdicts(results: dict[str, dict], baselines: list[str]) -> None:
    best = max(results[name]["penalised"] for name in baselines)
    print(f"\nbest reference, penalised score {best:.3f}; a judge must reach {best + MARGIN:.3f}")
    for name, item in results.items():
        if name in baselines:
            continue
        reasons = []
        if item["G1"] > GAP_LIMIT:
            reasons.append(f"G1 {item['G1']:.3f} > {GAP_LIMIT}")
        if item["G3"] > OVERSHOOT_LIMIT:
            reasons.append(f"G3 {item['G3']:.2f} > {OVERSHOOT_LIMIT}")
        if item["penalised"] < best + MARGIN:
            reasons.append(f"penalised {item['penalised']:.3f} < {best + MARGIN:.3f}")
        item["verdict"] = "passes" if not reasons else "out: " + "; ".join(reasons)
        print(f"  {name:<44}{item['verdict']}")


# ---------------------------------------------------------------- the two sides


def runSide(
    title: str,
    data: dict[str, np.ndarray],
    scalars: np.ndarray,
    token: np.ndarray,
    references: dict[str, np.ndarray],
    out: Path,
) -> dict:
    """Q, P and P x Q for both model kinds, against fixed references, all on the Q rows."""
    groups, q, p, qRows = data["groups"], data["q"], data["p"], data["qRows"]
    both = np.column_stack([scalars, token])
    nScalars = scalars.shape[1]
    results: dict[str, dict] = {}
    heldScores: dict[str, np.ndarray] = {}
    header(
        f"{title}: judged on the Q rows"
        f" ({int(q[qRows].sum())} positive, {int((~q[qRows]).sum())} negative)"
    )
    for name, values in references.items():

        def fixed(train, held, seed, values=values):
            return values[qRows][train], values[qRows][held], {}

        results[name], heldScores[name] = evaluate(name, q[qRows], groups[qRows], fixed)
    families = [
        Family("linear, scalars", "scalars", "linear"),
        Family("linear, scalars + token", "scalars+token", "linear"),
        Family("trees, scalars", "scalars", "trees"),
        Family("trees, scalars + token components", "scalars+token", "trees"),
    ]
    for family in families:
        x = scalars if family.columns == "scalars" else both
        cache: dict[tuple, Any] = {}

        def fitted(
            target: np.ndarray, rows: np.ndarray, train_groups_rows: np.ndarray, seed: int, key
        ):
            if key not in cache:
                cache[key] = chooseAndFit(
                    family,
                    x[train_groups_rows],
                    target[train_groups_rows],
                    groups[train_groups_rows],
                    nScalars,
                    seed,
                )
            return cache[key]

        def predictor(use: str, family=family, x=x):
            def predict(train, held, seed):
                # Indices are into the Q rows; P is trained on all rows of the same
                # training sequences, so it never sees a held-out sequence either.
                qIndex = np.flatnonzero(qRows)
                trainNames = np.unique(groups[qIndex[train]])
                pTrain = np.flatnonzero(np.isin(groups, trainNames))
                out_train, out_held, params = None, None, {}
                if use in ("Q", "PxQ"):
                    paramsQ, modelQ = fitted(
                        q, qRows, qIndex[train], seed, ("Q", seed, tuple(trainNames))
                    )
                    sq_train = modelQ.predict_proba(x[qIndex[train]])[:, 1]
                    sq_held = modelQ.predict_proba(x[qIndex[held]])[:, 1]
                    out_train, out_held, params["Q"] = sq_train, sq_held, paramsQ
                if use in ("P", "PxQ"):
                    paramsP, modelP = fitted(p, None, pTrain, seed, ("P", seed, tuple(trainNames)))
                    sp_train = modelP.predict_proba(x[qIndex[train]])[:, 1]
                    sp_held = modelP.predict_proba(x[qIndex[held]])[:, 1]
                    params["P"] = paramsP
                    if use == "P":
                        out_train, out_held = sp_train, sp_held
                    else:
                        out_train, out_held = out_train * sp_train, out_held * sp_held
                return out_train, out_held, params

            return predict

        for use in ("Q", "P", "PxQ"):
            name = f"{family.name}: {use}"
            results[name], heldScores[name] = evaluate(
                name, q[qRows], groups[qRows], predictor(use)
            )
    verdicts(results, list(references))
    best = max(references, key=lambda name: results[name]["penalised"])
    print(f"\nheld-out gain over '{best}' (split 0, sequences resampled):")
    for name, scores in heldScores.items():
        if name in references:
            continue
        gain = bootstrapGain(q[qRows], groups[qRows], scores, heldScores[best])
        results[name]["gainOverReference"] = {"reference": best, **gain}
        print(f"  {name:<44}{gain['gain']:+.3f} [{gain['low']:+.3f}, {gain['high']:+.3f}]")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(results, indent=1, default=str) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        out / "held_scores_split0.npz", **{k.replace(" ", "_"): v for k, v in heldScores.items()}
    )
    return results


def detectionData(args: argparse.Namespace):
    from track360.datasets.vot360 import Vot360Dataset
    from track360.evaluation.loss_rate import dualIou, lostFrameMask
    from track360.evaluation.vot360_metrics import loadTrackerResults

    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    boxes = loadTrackerResults(args.signals / "run" / "bbox" / "ours")
    spheres = loadTrackerResults(args.signals / "run" / "bfov" / "ours")
    old: dict[tuple[str, int], dict[str, str]] = {}
    for path in args.features:
        with path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                old[(row["sequence"], int(row["frame"]))] = row
    parts: dict[str, list] = {
        k: []
        for k in ("groups", "q", "qRows", "p", "scalars", "token", "score", "template", "seven")
    }
    for path in sorted((args.signals / "freerun").glob("*.npz")):
        name = path.stem
        data = np.load(path)
        sequence = dataset.sequence(name)
        truthBox, present = sequence.groundTruth("bbox")
        truthBfov, _ = sequence.groundTruth("bfov")
        width = sequence.frameSize[0]
        sequence.close()
        present = np.asarray(present, dtype=bool)
        iou = np.zeros(len(present))
        iou[present] = dualIou(truthBox[present], boxes[name][present], width)
        lost = lostFrameMask(iou, present)
        good = present & (iou >= 0.5)
        frames = np.arange(1, len(present))

        # P: the target's centre within two target sizes of the box's centre.
        def unit(rows):
            lon, lat = np.radians(rows[:, 0]), np.radians(rows[:, 1])
            return np.column_stack(
                [np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)]
            )

        angle = np.degrees(
            np.arccos(np.clip((unit(truthBfov) * unit(spheres[name])).sum(1), -1, 1))
        )
        size = np.sqrt(np.clip(truthBfov[:, 2] * truthBfov[:, 3], 1e-6, None))
        inside = present & (angle <= 2.0 * size)
        parts["groups"] += [name] * len(frames)
        parts["q"] += list(good[frames])
        parts["qRows"] += list((good | lost)[frames])
        parts["p"] += list(inside[frames])
        parts["score"] += list(data["score"])
        rows = [old.get((name, int(f))) for f in frames]
        parts["template"] += [float(r["template"]) if r else np.nan for r in rows]
        parts["seven"].append(
            np.array([[float(r[k]) for k in E025_SEVEN] if r else [np.nan] * 7 for r in rows])
        )
        parts["scalars"].append(
            np.column_stack(
                [
                    data["score"],
                    data["coordEntropy"].mean(1),
                    data["coordEntropy"].max(1),
                    data["coordPeak"].min(1),
                    data["hardSoftGap"].max(1),
                ]
            )
        )
        parts["token"].append(data["scoreFeature"].astype(np.float32))
    seven = np.concatenate(parts["seven"])
    if args.relative:
        # The sequence's own level over its first frames, which a running tracker has.
        names = np.asarray(parts["groups"])
        stacked = {
            "scalars": np.concatenate(parts["scalars"]),
            "token": np.concatenate(parts["token"]),
            "seven": seven,
        }
        for name in np.unique(names):
            rows = np.flatnonzero(names == name)
            first = rows[: args.relative]
            for key, values in stacked.items():
                values[rows] = values[rows] - np.nanmean(values[first], axis=0)
        parts["scalars"], parts["token"], seven = (
            [stacked["scalars"]],
            [stacked["token"]],
            stacked["seven"],
        )
    keep = ~np.isnan(seven).any(1)
    data = {
        "groups": np.asarray(parts["groups"])[keep],
        "q": np.asarray(parts["q"], dtype=bool)[keep],
        "qRows": np.asarray(parts["qRows"], dtype=bool)[keep],
        "p": np.asarray(parts["p"], dtype=bool)[keep],
    }
    scalars = np.column_stack([np.concatenate(parts["scalars"]), seven])[keep]
    token = np.concatenate(parts["token"])[keep]
    references = {
        "reference: backend score": np.asarray(parts["score"], dtype=float)[keep],
        "reference: DINOv2 template similarity": np.asarray(parts["template"], dtype=float)[keep],
    }
    return data, scalars, token, references, seven[keep]


def candidateData(args: argparse.Namespace):
    """Real search candidates of the runs under ``--runs``; one row per candidate."""
    parts: dict[str, list] = {
        k: []
        for k in (
            "groups",
            "experiment",
            "q",
            "qRows",
            "p",
            "scalars",
            "token",
            "search",
            "similarity",
        )
    }
    for path in sorted(args.runs.glob("*/trace_candidates.npz")):
        run = path.parent.name
        data = np.load(path)
        iou = data["iou"]
        q = iou >= 0.5
        qRows = (iou >= 0.5) | (iou < 0.1)
        p = data["present"] & (data["truthDistanceSizes"] <= 2.0)
        scan = data["scan"]
        n = len(iou)
        parts["groups"] += list(data["sequence"])
        parts["experiment"] += [run.split("_")[0]] * n
        parts["q"] += list(q)
        parts["qRows"] += list(qRows)
        parts["p"] += list(p)
        parts["search"] += list(data["originalScore"])
        parts["similarity"] += list(data["similarity"])
        parts["scalars"].append(
            np.column_stack(
                [
                    data["originalScore"],
                    data["score"],
                    data["similarity"],
                    np.log(np.clip(data["sizeDeg"], 1e-3, None)),
                    (scan == "1x").astype(float),
                    (scan == "4x").astype(float),
                    data["coordEntropy"].mean(1),
                    data["coordEntropy"].max(1),
                    data["coordPeak"].min(1),
                    data["hardSoftGap"].max(1),
                ]
            )
        )
        parts["token"].append(data["scoreFeature"].astype(np.float32))
    data = {
        "groups": np.asarray(parts["groups"]),
        "experiment": np.asarray(parts["experiment"]),
        "q": np.asarray(parts["q"], dtype=bool),
        "qRows": np.asarray(parts["qRows"], dtype=bool),
        "p": np.asarray(parts["p"], dtype=bool),
    }
    scalars = np.concatenate(parts["scalars"])
    token = np.concatenate(parts["token"])
    references = {
        "reference: search score": np.asarray(parts["search"], dtype=float),
        # The pass the label is read from: the fair reference for the Q label.
        "reference: recentred-pass score": scalars[:, 1].astype(float),
        "reference: DINOv2 similarity": np.asarray(parts["similarity"], dtype=float),
    }
    if args.relative_signals is not None:
        # Each sequence's own stateless token on the target in its first frames: the
        # stateless passes around the true target on frames 5-30 of E045 (a running
        # tracker would make such a pass on the frames it trusts after initialisation).
        base = np.load(args.relative_signals / "candidates.npz")
        early = (base["kind"] == "near") & (base["frame"] <= 30) & (base["iou"] >= 0.5)
        own = {
            name: base["scoreFeature"][early & (base["sequence"] == name)]
            .astype(np.float32)
            .mean(0)
            for name in np.unique(base["sequence"][early])
        }
        missing = sorted(set(np.unique(data["groups"])) - set(own))
        if missing:
            print(f"no early reference token for {missing}: those candidates are left out")
        keep = np.isin(data["groups"], list(own))
        token = token - np.stack(
            [own.get(name, np.zeros(token.shape[1])) for name in data["groups"]]
        )
        data = {key: value[keep] for key, value in data.items()}
        scalars, token = scalars[keep], token[keep]
        references = {key: value[keep] for key, value in references.items()}
    return data, scalars, token, references


def runTransfer(data, scalars, token, out: Path) -> dict:
    """G4: fit on the runs of one experiment, test on the other's runs of unseen sequences."""
    groups, q, qRows, experiment = data["groups"], data["q"], data["qRows"], data["experiment"]
    both = np.column_stack([scalars, token])
    names = np.unique(groups)
    results = {}
    print("\nG4: trained on E043 runs of 3/4 of the sequences, tested on E048 runs of the rest")
    for family in (
        Family("linear, scalars + token", "scalars+token", "linear"),
        Family("trees, scalars + token components", "scalars+token", "trees"),
    ):
        x = both
        held_auc, ref_auc = [], []
        for seed in SPLITS:
            for train_names, held_names in (
                (names[t], names[h])
                for t, h in GroupKFold(OUTER_FOLDS, shuffle=True, random_state=seed).split(
                    names, groups=names
                )
            ):
                train = qRows & (experiment == "E043") & np.isin(groups, train_names)
                held = qRows & (experiment == "E048") & np.isin(groups, held_names)
                _, model = chooseAndFit(
                    family, x[train], q[train], groups[train], scalars.shape[1], seed
                )
                held_auc.append(auc(q[held], model.predict_proba(x[held])[:, 1]))
                ref_auc.append(auc(q[held], scalars[held, 0]))
        results[family.name] = {
            "heldAuc": float(np.nanmean(held_auc)),
            "searchScoreAuc": float(np.nanmean(ref_auc)),
        }
        print(
            f"  {family.name:<44}held {np.nanmean(held_auc):.3f}"
            f"   search score on the same rows {np.nanmean(ref_auc):.3f}"
        )
    (out / "transfer.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("side", choices=("detection", "candidates"))
    parser.add_argument("--dataset-root")
    parser.add_argument("--label-root")
    parser.add_argument("--signals", type=Path)
    parser.add_argument("--features", type=Path, action="append")
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", choices=sorted(GRIDS), default="first")
    parser.add_argument(
        "--relative-signals",
        type=Path,
        default=None,
        help="candidates: subtract each sequence's early stateless token (from this E045 folder)",
    )
    parser.add_argument(
        "--relative",
        type=int,
        default=0,
        help="detection: subtract each sequence's mean over its first N frames from every feature",
    )
    args = parser.parse_args(argv)
    GRID["name"] = args.grid
    warnings.filterwarnings("ignore")
    if args.side == "detection":
        CATCH["negatives"] = True
        data, scalars, token, references, seven = detectionData(args)
        print(
            f"detection: {len(data['q'])} frames of {len(np.unique(data['groups']))} sequences; "
            f"P positive {data['p'].mean():.1%}"
        )
        # E025's seven features, fitted in the same nested way, as the strongest reference.
        results_seven = {}

        def sevenPredict(train, held, seed):
            rows = np.flatnonzero(data["qRows"])
            params, model = chooseAndFit(
                Family("linear", "scalars", "linear"),
                seven[rows[train]],
                data["q"][rows[train]],
                data["groups"][rows[train]],
                seven.shape[1],
                seed,
            )
            return (
                model.predict_proba(seven[rows[train]])[:, 1],
                model.predict_proba(seven[rows[held]])[:, 1],
                params,
            )

        header("detection reference fitted the same way")
        results_seven["reference: E025 seven, linear"], held = evaluate(
            "reference: E025 seven, linear",
            data["q"][data["qRows"]],
            data["groups"][data["qRows"]],
            sevenPredict,
        )
        references = {
            **references,
            "reference: E025 seven, linear": np.full(len(data["q"]), np.nan),
        }
        full = np.full(len(data["q"]), np.nan)
        full[np.flatnonzero(data["qRows"])] = held
        references["reference: E025 seven, linear"] = full
        runSide("detection", data, scalars, token, references, args.out)
    else:
        data, scalars, token, references = candidateData(args)
        print(
            f"candidates: {len(data['q'])} from {len(np.unique(data['experiment']))} experiments, "
            f"{len(np.unique(data['groups']))} sequences; P positive {data['p'].mean():.1%}"
        )
        runSide("candidates", data, scalars, token, references, args.out)
        runTransfer(data, scalars, token, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
