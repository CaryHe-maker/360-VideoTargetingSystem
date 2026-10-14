"""E049 round 4: P1, P2 and Q as separate quantities, with overfitting as the main penalty.

    python tools/head_study_r4.py candidates --runs outputs/E049/runs \
        --out outputs/E049/r4_candidates
    python tools/head_study_r4.py detection --dataset-root <dir> --label-root <dir> \\
        --signals outputs/E045 --features outputs/E025/features_tune.csv \\
        --features outputs/E025/features_holdout.csv --out outputs/E049/r4_detection

The quantities (registered in evaluation-log E049 before this round ran):

    P1  classification: is the target's centre inside the normal-size view around the
        box (within one half-width, i.e. two sizes)?
    P2  regression: how likely the backend boxes the target from where it is in that
        view.  Its label is the backend's measured success rate as a function of the
        target's distance from the view's centre (in half-widths), estimated on the
        training sequences of each fold only, falling with distance; 0 when the target
        is not in the picture.
    Q   regression: the IoU of the box with the ground truth, learned only on the rows
        whose target is inside the view.

Each is learned by a linear model on the scalar features and the score token, by the
same with the directions that most separate the training sequences projected out of
the token, and by gradient-boosted trees on the scalars and 16 token components.  Every
training row is weighted so that each sequence counts the same.

Every quantity is judged on the decision it serves, the Q label of the earlier rounds
(box on the target, IoU >= 0.5, against IoU < 0.1), with the protocol of the earlier
rounds: nested cross-validation by sequence (outer 4 folds x 3 splits, inner 3 folds),
G1 (train - held AUROC), G2 (spread), G3 (false alarms of a training threshold on held
sequences / 5%), and in addition G5 (the worst held sequence's AUROC) and the
count-based operating point (a threshold that accepts, on the training sequences, as
many rows as the reference rule does).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from head_study import E025_SEVEN, Clip, TokenPca  # noqa: E402

OUTER_FOLDS, INNER_FOLDS, SPLITS = 4, 3, (0, 1, 2)
GAP_ALLOWANCE, GAP_LIMIT, OVERSHOOT_LIMIT, MARGIN = 0.03, 0.10, 2.0, 0.02
SEQUENCE_DIRECTIONS = 8
DISTANCE_EDGES = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 1e9)

CLASSIFIER_C = (1e-2, 1e-3, 1e-4, 1e-5)
RIDGE_ALPHA = (1e2, 1e3, 1e4, 1e5)
TREE_GRID = [
    {"max_leaf_nodes": leaves, "min_samples_leaf": leaf, "l2_regularization": l2, "max_iter": it}
    for leaves, leaf, l2, it in product((3, 4), (500, 1500), (10.0, 100.0), (50, 150))
]


# ---------------------------------------------------------------- feature transforms


def sequenceDirections(token: np.ndarray, groups: np.ndarray, count: int) -> np.ndarray:
    """The directions in which the training sequences' mean tokens differ most."""
    scale = token.std(0) + 1e-6
    centred = (token - token.mean(0)) / scale
    means = np.stack([centred[groups == name].mean(0) for name in np.unique(groups)])
    _, _, vt = np.linalg.svd(means - means.mean(0), full_matrices=False)
    return vt[:count] / scale  # rows act on the raw token


def removeDirections(token: np.ndarray, directions: np.ndarray, centre: np.ndarray) -> np.ndarray:
    basis, _ = np.linalg.qr(directions.T)
    centred = token - centre
    return token - centred @ basis @ basis.T


# ---------------------------------------------------------------- models


def makeModel(kind: str, task: str, params: dict, scalars: int) -> Pipeline:
    if kind == "trees":
        estimator = (
            HistGradientBoostingClassifier(class_weight="balanced", **TREE_FIXED, **params)
            if task == "classify"
            else HistGradientBoostingRegressor(**TREE_FIXED, **params)
        )
        return Pipeline([("pca", TokenPca(scalars)), ("model", estimator)])
    estimator = (
        LogisticRegression(C=params["C"], class_weight="balanced", max_iter=3000)
        if task == "classify"
        else Ridge(alpha=params["alpha"])
    )
    return Pipeline([("clip", Clip()), ("scale", StandardScaler()), ("model", estimator)])


TREE_FIXED = {
    "learning_rate": 0.05,
    "max_features": 0.5,
    "early_stopping": False,
    "random_state": 0,
}


def grid(kind: str, task: str) -> list[dict]:
    if kind == "trees":
        return TREE_GRID
    return (
        [{"C": c} for c in CLASSIFIER_C]
        if task == "classify"
        else [{"alpha": a} for a in RIDGE_ALPHA]
    )


def weights(groups: np.ndarray) -> np.ndarray:
    """Every sequence counts the same, however many rows it has."""
    names, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return (len(groups) / len(names)) / counts[inverse]


def predict(model: Pipeline, task: str, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1] if task == "classify" else model.predict(x)


def ownScore(task: str, y: np.ndarray, s: np.ndarray) -> float:
    if task == "classify":
        return auc(y.astype(bool), s)
    rho = spearmanr(y, s).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def splits(groups: np.ndarray, seed: int, folds: int):
    return GroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(
        np.zeros(len(groups)), groups=groups
    )


class Head:
    """One quantity learned one way: fitted on a training part, scored anywhere."""

    def __init__(self, quantity: str, kind: str, remove: bool):
        self.quantity, self.kind, self.remove = quantity, kind, remove
        self.task = "classify" if quantity == "P1" else "regress"

    @property
    def name(self) -> str:
        how = {"linear": "linear", "trees": "trees"}[self.kind]
        return f"{self.quantity}, {how}{', sequence directions removed' if self.remove else ''}"

    def _features(self, x: np.ndarray, nScalars: int) -> np.ndarray:
        if not self.remove:
            return x
        token = removeDirections(x[:, nScalars:], self.directions_, self.centre_)
        return np.column_stack([x[:, :nScalars], token])

    def fit(self, data: dict, rows: np.ndarray, seed: int) -> Head:
        """Fit on ``rows`` (indices of training rows), choosing parameters inside them."""
        x, groups, n = data["x"], data["groups"], data["nScalars"]
        if self.remove:
            self.directions_ = sequenceDirections(x[rows, n:], groups[rows], SEQUENCE_DIRECTIONS)
            self.centre_ = x[rows, n:].mean(0)
        target, use = self._target(data, rows)
        train = rows[use]
        features = self._features(x[train], n)
        y, g = target[use], groups[train]
        best, bestScore = None, -np.inf
        options = grid(self.kind, self.task)
        if len(options) > 1:
            for params in options:
                values = []
                for inner, valid in splits(g, seed + 100, INNER_FOLDS):
                    model = makeModel(self.kind, self.task, params, n)
                    model.fit(features[inner], y[inner], model__sample_weight=weights(g[inner]))
                    values.append(
                        ownScore(self.task, y[valid], predict(model, self.task, features[valid]))
                    )
                if np.nanmean(values) > bestScore:
                    best, bestScore = params, float(np.nanmean(values))
        else:
            best = options[0]
        self.params_ = best
        self.model_ = makeModel(self.kind, self.task, best, n)
        self.model_.fit(features, y, model__sample_weight=weights(g))
        return self

    def _target(self, data: dict, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The label of each training row, and which training rows learn it."""
        if self.quantity == "P1":
            return data["inside"][rows].astype(float), np.ones(len(rows), dtype=bool)
        if self.quantity == "Q":
            return data["iou"][rows], data["inside"][rows]
        # P2: the backend's measured success rate at that distance, from these rows only.
        self.curve_ = successCurve(data, rows)
        label = np.where(data["present"][rows], self.curve_(data["distance"][rows]), 0.0)
        return label, np.ones(len(rows), dtype=bool)

    def score(self, data: dict, rows: np.ndarray) -> np.ndarray:
        return predict(self.model_, self.task, self._features(data["x"][rows], data["nScalars"]))


def successCurve(data: dict, rows: np.ndarray):
    """Share of rows whose box is on the target, by distance from the view's centre."""
    present = rows[data["present"][rows]]
    distance = np.minimum(data["distance"][present], DISTANCE_EDGES[-2])
    hit = (data["iou"][present] >= 0.5).astype(float)
    curve = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=0.0, y_max=1.0)
    curve.fit(distance, hit, sample_weight=weights(data["groups"][present]))
    return lambda d: curve.predict(np.minimum(d, DISTANCE_EDGES[-2]))


# ---------------------------------------------------------------- evaluation


def evaluateHead(head: Head | None, reference: np.ndarray | None, data: dict, catchNegatives: bool):
    """Outer folds of all splits; the decision label is ``decision`` on the ``decisionRows``."""
    decision, decisionRows, groups = data["decision"], data["decisionRows"], data["groups"]
    rows = np.arange(len(groups))
    records, first = [], np.full(len(groups), np.nan)
    for seed in SPLITS:
        for trainIdx, heldIdx in splits(groups, seed, OUTER_FOLDS):
            if head is not None:
                fitted = Head(head.quantity, head.kind, head.remove).fit(data, rows[trainIdx], seed)
                trainScores = fitted.score(data, rows[trainIdx])
                heldScores = fitted.score(data, rows[heldIdx])
                own = ownMetrics(fitted, data, rows[heldIdx], heldScores)
                params = fitted.params_
            else:
                trainScores, heldScores, own, params = (
                    reference[trainIdx],
                    reference[heldIdx],
                    {},
                    {},
                )
            tr, he = decisionRows[trainIdx], decisionRows[heldIdx]
            yt, yh = decision[trainIdx][tr], decision[heldIdx][he]
            st, sh = trainScores[tr], heldScores[he]
            record = {"trainAuc": auc(yt, st), "heldAuc": auc(yh, sh), "params": params, **own}
            # G3: a 5% false-alarm threshold set on the training part.
            if catchNegatives:
                threshold = float(np.quantile(st[yt], 0.05))
                flagged = sh < threshold
                record["recall@0.05"], record["falseAlarm@0.05"] = (
                    float(flagged[~yh].mean()),
                    float(flagged[yh].mean()),
                )
            else:
                threshold = float(np.quantile(st[~yt], 0.95))
                taken = sh > threshold
                record["recall@0.05"], record["falseAlarm@0.05"] = (
                    float(taken[yh].mean()),
                    float(taken[~yh].mean()),
                )
            # Count-based operating point: accept the share of rows the reference accepts.
            if data.get("referenceShare") is not None:
                share = data["referenceShare"]
                cut = float(np.quantile(st, 1.0 - share))
                taken = sh > cut
                record["countShareRatio"] = float(taken.mean() / share)
                record["countPrecision"] = float(yh[taken].mean()) if taken.any() else float("nan")
            # G5: the worst held sequence with enough rows of both kinds.
            per = []
            for name in np.unique(groups[heldIdx][he]):
                mask = groups[heldIdx][he] == name
                if (yh[mask]).sum() >= 5 and (~yh[mask]).sum() >= 5:
                    per.append(auc(yh[mask], sh[mask]))
            record["worstSequenceAuc"] = float(np.min(per)) if per else float("nan")
            records.append(record)
            if seed == SPLITS[0]:
                first[heldIdx] = heldScores
    return summarise(records), first


def ownMetrics(head: Head, data: dict, held: np.ndarray, scores: np.ndarray) -> dict:
    if head.quantity == "P1":
        return {"ownAuc": auc(data["inside"][held], scores)}
    if head.quantity == "Q":
        inside = data["inside"][held]
        return {"ownSpearman": ownScore("regress", data["iou"][held][inside], scores[inside])}
    label = np.where(data["present"][held], head.curve_(data["distance"][held]), 0.0)
    return {"ownSpearman": ownScore("regress", label, scores)}


def summarise(records: list[dict]) -> dict:
    held = np.array([r["heldAuc"] for r in records])
    train = np.array([r["trainAuc"] for r in records])
    gap, spread = float(np.nanmean(train - held)), float(np.nanstd(held, ddof=1))
    out = {
        "heldAuc": float(np.nanmean(held)),
        "trainAuc": float(np.nanmean(train)),
        "G1": gap,
        "G2": spread,
        "penalised": float(np.nanmean(held)) - max(0.0, gap - GAP_ALLOWANCE) - spread,
    }
    for key in (
        "recall@0.05",
        "falseAlarm@0.05",
        "countShareRatio",
        "countPrecision",
        "worstSequenceAuc",
        "ownAuc",
        "ownSpearman",
    ):
        values = [r[key] for r in records if key in r]
        if values:
            out[key] = float(np.nanmean(values))
    out["G3"] = out["falseAlarm@0.05"] / 0.05
    out["params"] = [r["params"] for r in records]
    return out


def line(name: str, s: dict) -> str:
    own = s.get("ownAuc", s.get("ownSpearman", float("nan")))
    count = s.get("countPrecision", float("nan"))
    ratio = s.get("countShareRatio", float("nan"))
    return (
        f"  {name:<46}{s['heldAuc']:>7.3f}{s['trainAuc']:>7.3f}{s['G1']:>7.3f}{s['G2']:>7.3f}"
        f"{s['G3']:>7.2f}{s['recall@0.05']:>8.1%}{s.get('worstSequenceAuc', float('nan')):>8.3f}"
        f"{count:>8.1%}{ratio:>7.2f}{own:>8.3f}{s['penalised']:>8.3f}"
    )


HEADER = (
    f"  {'judge':<46}{'held':>7}{'train':>7}{'G1':>7}{'G2':>7}{'G3':>7}{'rec@5%':>8}"
    f"{'worst':>8}{'cnt.prec':>8}{'cnt.x':>7}{'own':>8}{'penal.':>8}"
)


def bootstrapGain(data: dict, scores: np.ndarray, reference: np.ndarray, samples=2000) -> dict:
    rows = data["decisionRows"]
    y, g, s, r = data["decision"][rows], data["groups"][rows], scores[rows], reference[rows]
    names = np.unique(g)
    index = {name: np.flatnonzero(g == name) for name in names}
    rng = np.random.default_rng(0)
    gains = []
    for _ in range(samples):
        pick = np.concatenate([index[name] for name in rng.choice(names, len(names))])
        gains.append(auc(y[pick], s[pick]) - auc(y[pick], r[pick]))
    return {
        "gain": auc(y, s) - auc(y, r),
        "low": float(np.nanpercentile(gains, 2.5)),
        "high": float(np.nanpercentile(gains, 97.5)),
    }


def run(data: dict, references: dict[str, np.ndarray], out: Path, catchNegatives: bool) -> None:
    print(
        f"decision rows: {int(data['decision'][data['decisionRows']].sum())} positive,"
        f" {int((~data['decision'][data['decisionRows']]).sum())} negative;"
        f" P1 positive {data['inside'].mean():.1%};"
        f" Q rows (target inside) {int(data['inside'].sum())}"
    )
    print(HEADER, flush=True)
    results, firsts = {}, {}
    for name, values in references.items():
        results[name], firsts[name] = evaluateHead(None, values, data, catchNegatives)
        print(line(name, results[name]), flush=True)
    for quantity in ("P1", "P2", "Q"):
        for kind, remove in (("linear", False), ("linear", True), ("trees", False)):
            head = Head(quantity, kind, remove)
            results[head.name], firsts[head.name] = evaluateHead(head, None, data, catchNegatives)
            print(line(head.name, results[head.name]), flush=True)
    best = max(references, key=lambda name: results[name]["penalised"])
    bar = results[best]["penalised"] + MARGIN
    print(f"\nbest reference '{best}' (penalised {results[best]['penalised']:.3f}); bar {bar:.3f}")
    for name, item in results.items():
        if name in references:
            continue
        reasons = []
        if item["G1"] > GAP_LIMIT:
            reasons.append(f"G1 {item['G1']:.3f}")
        if item["G3"] > OVERSHOOT_LIMIT:
            reasons.append(f"G3 {item['G3']:.2f}")
        if item["penalised"] < bar:
            reasons.append(f"penalised {item['penalised']:.3f}")
        gain = bootstrapGain(data, firsts[name], firsts[best])
        item["gainOverReference"] = {"reference": best, **gain}
        if gain["low"] <= 0:
            reasons.append("gain interval reaches 0")
        item["verdict"] = "passes" if not reasons else "out: " + "; ".join(reasons)
        print(
            f"  {name:<46}gain {gain['gain']:+.3f} [{gain['low']:+.3f}, {gain['high']:+.3f}]"
            f"  {item['verdict']}"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(results, indent=1, default=str) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------- data


def candidateData(runs: Path):
    parts: dict[str, list] = {
        k: []
        for k in (
            "groups",
            "iou",
            "present",
            "distance",
            "scalars",
            "token",
            "search",
            "recentred",
            "verdict",
        )
    }
    for path in sorted(runs.glob("*/trace_candidates.npz")):
        d = np.load(path)
        scan = d["scan"]
        parts["groups"] += list(d["sequence"])
        parts["iou"] += list(d["iou"])
        parts["present"] += list(d["present"])
        # In half-widths of the normal view around the candidate (two candidate sizes).
        parts["distance"] += list(np.where(d["present"], d["truthDistanceSizes"] / 2.0, 1e9))
        parts["search"] += list(d["originalScore"])
        parts["recentred"] += list(d["score"])
        parts["verdict"] += list(d["verdict"])
        # The absolute candidate size is left out: it tells the sequence apart (E025).
        parts["scalars"].append(
            np.column_stack(
                [
                    d["originalScore"],
                    d["score"],
                    d["similarity"],
                    (scan == "1x").astype(float),
                    (scan == "4x").astype(float),
                    d["coordEntropy"].mean(1),
                    d["coordEntropy"].max(1),
                    d["coordPeak"].min(1),
                    d["hardSoftGap"].max(1),
                ]
            )
        )
        parts["token"].append(d["scoreFeature"].astype(np.float32))
    scalars, token = np.concatenate(parts["scalars"]), np.concatenate(parts["token"])
    iou = np.asarray(parts["iou"], dtype=float)
    present = np.asarray(parts["present"], dtype=bool)
    distance = np.asarray(parts["distance"], dtype=float)
    data = {
        "groups": np.asarray(parts["groups"]),
        "x": np.column_stack([scalars, token]),
        "nScalars": scalars.shape[1],
        "iou": iou,
        "present": present,
        "distance": distance,
        "inside": present & (distance <= 1.0),
        "decision": iou >= 0.5,
        "decisionRows": (iou >= 0.5) | (iou < 0.1),
        # The share of candidates the current rule (search score >= 0.70) accepts.
        "referenceShare": float((np.asarray(parts["search"]) >= 0.70).mean()),
    }
    references = {
        "reference: search score": np.asarray(parts["search"], dtype=float),
        "reference: recentred-pass score": np.asarray(parts["recentred"], dtype=float),
    }
    return data, references


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
        for k in (
            "groups",
            "iou",
            "lost",
            "present",
            "distance",
            "scalars",
            "token",
            "score",
            "template",
        )
    }

    def unit(rows):
        lon, lat = np.radians(rows[:, 0]), np.radians(rows[:, 1])
        return np.column_stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)])

    for path in sorted((args.signals / "freerun").glob("*.npz")):
        name = path.stem
        d = np.load(path)
        sequence = dataset.sequence(name)
        truthBox, present = sequence.groundTruth("bbox")
        truthBfov, _ = sequence.groundTruth("bfov")
        width = sequence.frameSize[0]
        sequence.close()
        present = np.asarray(present, dtype=bool)
        iou = np.zeros(len(present))
        iou[present] = dualIou(truthBox[present], boxes[name][present], width)
        lost = lostFrameMask(iou, present)
        angle = np.degrees(
            np.arccos(np.clip((unit(truthBfov) * unit(spheres[name])).sum(1), -1, 1))
        )
        size = np.sqrt(np.clip(truthBfov[:, 2] * truthBfov[:, 3], 1e-6, None))
        frames = np.arange(1, len(present))
        rows = [old.get((name, int(f))) for f in frames]
        if any(r is None for r in rows):
            continue
        parts["groups"] += [name] * len(frames)
        parts["iou"] += list(iou[frames])
        parts["lost"] += list(lost[frames])
        parts["present"] += list(present[frames])
        parts["distance"] += list(np.where(present, angle / (2.0 * size), 1e9)[frames])
        parts["score"] += list(d["score"])
        parts["template"] += [float(r["template"]) for r in rows]
        seven = np.array([[float(r[k]) for k in E025_SEVEN] for r in rows])
        parts["scalars"].append(
            np.column_stack(
                [
                    d["score"],
                    d["coordEntropy"].mean(1),
                    d["coordEntropy"].max(1),
                    d["coordPeak"].min(1),
                    d["hardSoftGap"].max(1),
                    seven,
                ]
            )
        )
        parts["token"].append(d["scoreFeature"].astype(np.float32))
    scalars, token = np.concatenate(parts["scalars"]), np.concatenate(parts["token"])
    iou = np.asarray(parts["iou"], dtype=float)
    lost = np.asarray(parts["lost"], dtype=bool)
    present = np.asarray(parts["present"], dtype=bool)
    distance = np.asarray(parts["distance"], dtype=float)
    data = {
        "groups": np.asarray(parts["groups"]),
        "x": np.column_stack([scalars, token]),
        "nScalars": scalars.shape[1],
        "iou": iou,
        "present": present,
        "distance": distance,
        "inside": present & (distance <= 1.0),
        "decision": iou >= 0.5,
        "decisionRows": (iou >= 0.5) | lost,
        "referenceShare": None,
    }
    references = {
        "reference: backend score": np.asarray(parts["score"], dtype=float),
        "reference: DINOv2 template similarity": np.asarray(parts["template"], dtype=float),
    }
    return data, references


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
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    if args.side == "candidates":
        data, references = candidateData(args.runs)
        run(data, references, args.out, catchNegatives=False)
    else:
        data, references = detectionData(args)
        run(data, references, args.out, catchNegatives=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
