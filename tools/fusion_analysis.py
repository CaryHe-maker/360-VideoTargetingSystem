"""Find how to mix the three confidence signals so that lost frames stand out.

    python tools/fusion_analysis.py --fit outputs/E018/tune_episodes.csv \\
        outputs/E018/tune_freerun.csv --check outputs/E018/holdout_episodes.csv \\
        outputs/E018/holdout_freerun.csv --json outputs/E018/fusion.json

The tables come from ``tools/fusion_dataset.py`` (each frame reached with true history)
and ``tools/fusion_freerun.py`` (a normal run).  On the frames where the target is
present, with "lost" meaning IoU < 0.1 and "good" IoU >= 0.5, the tool reports

* how the three signals and the IoU correlate,
* how well each signal alone separates lost from good frames,
* the weights of ``fused = wB * backend + wA * appearance + wM * motion`` that separate
  them best, searched on a grid over the ``--fit`` tables and then applied unchanged to
  the ``--check`` tables.

The signals, all in [0, 1]:

    backend     the tracker's own score
    appearance  the larger of the DINOv2 similarities to the template and to the recent
                reference, negative values counted as 0
    motion      exp(-(offset / sO)^2 / 2) * exp(-(logScale / sS)^2 / 2): 1 when the box is
                where and as large as the motion model predicted
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

from track360.evaluation.score_analysis import auroc, spearman

LOST_IOU = 0.1
GOOD_IOU = 0.5
WEIGHT_STEP = 0.05
OFFSET_SCALES = (0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5)
SIZE_SCALES = (0.1, 0.2, 0.3, 0.5, 1.0, float("inf"))
BUDGETS = (0.01, 0.05)
# How the appearance signal is built from the two similarities.
APPEARANCE = "max"


def loadTable(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    table = {
        key: np.asarray([float(row[key]) for row in rows])
        for key in ("present", "iou", "score", "simTemplate", "simRecent", "offset", "logScale")
    }
    table["variant"] = np.asarray([row["variant"] for row in rows])
    table["sequence"] = np.asarray([row["sequence"] for row in rows])
    return table


def signals(table: dict[str, np.ndarray], offsetScale: float, sizeScale: float) -> np.ndarray:
    """Backend, appearance and motion score of every row, each in [0, 1]."""
    backend = np.clip(table["score"], 0.0, 1.0)
    similarity = (
        table["simTemplate"]
        if APPEARANCE == "template"
        else np.maximum(table["simTemplate"], table["simRecent"])
    )
    appearance = np.clip(similarity, 0.0, 1.0)
    motion = np.exp(-0.5 * (table["offset"] / offsetScale) ** 2)
    if np.isfinite(sizeScale):
        motion = motion * np.exp(-0.5 * (table["logScale"] / sizeScale) ** 2)
    return np.column_stack([backend, appearance, motion])


def classes(table: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    present = table["present"] == 1
    return present & (table["iou"] >= GOOD_IOU), present & (table["iou"] < LOST_IOU)


def separation(values: np.ndarray, good: np.ndarray, lost: np.ndarray) -> dict[str, float]:
    """AUROC of lost against good, and the lost frames flagged within each budget."""
    both = good | lost
    result = {"auroc": float(auroc(-values[both], lost[both]) or float("nan"))}
    for budget in BUDGETS:
        threshold = float(np.quantile(values[good], budget))
        result[f"caught@{budget:.0%}"] = float((values[lost] < threshold).mean())
        result[f"threshold@{budget:.0%}"] = threshold
    return result


def weightGrid() -> list[tuple[float, float, float]]:
    steps = round(1.0 / WEIGHT_STEP)
    return [
        (b / steps, a / steps, (steps - b - a) / steps)
        for b, a in product(range(steps + 1), repeat=2)
        if b + a <= steps
    ]


def describe(name: str, table: dict[str, np.ndarray], scales: tuple[float, float]) -> dict:
    good, lost = classes(table)
    present = table["present"] == 1
    matrix = signals(table, *scales)
    columns = {
        "iou": table["iou"],
        "backend": matrix[:, 0],
        "appearance": matrix[:, 1],
        "motion": matrix[:, 2],
    }
    names = list(columns)
    correlation = {
        first: {
            second: float(spearman(columns[first][present], columns[second][present]) or 0.0)
            for second in names
        }
        for first in names
    }
    variants = {
        str(variant): {
            "rows": int((table["variant"] == variant).sum()),
            "lost": int((lost & (table["variant"] == variant)).sum()),
            "good": int((good & (table["variant"] == variant)).sum()),
        }
        for variant in np.unique(table["variant"])
    }
    return {
        "name": name,
        "rows": len(present),
        "present": int(present.sum()),
        "absent": int((~present).sum()),
        "good": int(good.sum()),
        "lost": int(lost.sum()),
        "sequences": int(len(np.unique(table["sequence"]))),
        "variants": variants,
        "correlation": correlation,
        "single": {
            key: separation(columns[key], good, lost) for key in ("backend", "appearance", "motion")
        },
        "templateOnly": separation(np.clip(table["simTemplate"], 0.0, 1.0), good, lost),
    }


def chooseScales(tables: dict[str, dict[str, np.ndarray]]) -> tuple[float, float]:
    """The motion-score scales with the best mean AUROC of the motion signal alone."""
    best, bestValue = (OFFSET_SCALES[0], SIZE_SCALES[0]), -1.0
    for offsetScale, sizeScale in product(OFFSET_SCALES, SIZE_SCALES):
        values = []
        for table in tables.values():
            good, lost = classes(table)
            motion = signals(table, offsetScale, sizeScale)[:, 2]
            values.append(separation(motion, good, lost)["auroc"])
        if np.mean(values) > bestValue:
            best, bestValue = (offsetScale, sizeScale), float(np.mean(values))
    return best


def searchWeights(
    tables: dict[str, dict[str, np.ndarray]], scales: tuple[float, float], metric: str = "auroc"
) -> list:
    """Every weight triple with ``metric`` on each table and their mean, best first."""
    prepared = []
    for table in tables.values():
        good, lost = classes(table)
        prepared.append((signals(table, *scales), good, lost))
    rows = []
    for weights in weightGrid():
        values = [
            separation(matrix @ np.asarray(weights), good, lost)[metric]
            for matrix, good, lost in prepared
        ]
        rows.append({"weights": weights, "auroc": values, "mean": float(np.mean(values))})
    return sorted(rows, key=lambda row: -row["mean"])


def logisticWeights(tables: dict[str, dict[str, np.ndarray]], scales: tuple[float, float]) -> dict:
    """Coefficients of a logistic "this frame is good" on the three signals."""
    features, target = [], []
    for table in tables.values():
        good, lost = classes(table)
        both = good | lost
        # Each table counts the same, however many rows it has.
        features.append((signals(table, *scales)[both], 1.0 / both.sum()))
        target.append(good[both].astype(float))
    x = np.vstack([np.column_stack([matrix, np.ones(len(matrix))]) for matrix, _ in features])
    w = np.concatenate([np.full(len(matrix), weight) for matrix, weight in features])
    y = np.concatenate(target)
    coefficients = np.zeros(4)
    for _ in range(100):
        probability = 1.0 / (1.0 + np.exp(-x @ coefficients))
        gradient = x.T @ (w * (probability - y))
        curvature = (x * (w * probability * (1.0 - probability))[:, None]).T @ x
        coefficients -= np.linalg.solve(curvature + 1e-9 * np.eye(4), gradient)
    positive = np.clip(coefficients[:3], 0.0, None)
    return {
        "backend": float(coefficients[0]),
        "appearance": float(coefficients[1]),
        "motion": float(coefficients[2]),
        "bias": float(coefficients[3]),
        "normalized": (positive / positive.sum()).round(3).tolist() if positive.sum() else None,
    }


def evaluate(
    tables: dict[str, dict[str, np.ndarray]],
    scales: tuple[float, float],
    weights: tuple[float, float, float],
) -> dict[str, dict[str, float]]:
    result = {}
    for name, table in tables.items():
        good, lost = classes(table)
        result[name] = separation(signals(table, *scales) @ np.asarray(weights), good, lost)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fit", type=Path, nargs="+", required=True)
    parser.add_argument("--check", type=Path, nargs="*", default=[])
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--appearance",
        choices=("max", "template"),
        default="max",
        help="max: the larger of the similarities to the template and to the recent "
        "reference; template: the similarity to the template only",
    )
    parser.add_argument(
        "--metric",
        default="auroc",
        choices=("auroc", "caught@1%", "caught@5%"),
        help="what the weight search maximizes",
    )
    args = parser.parse_args(argv)
    global APPEARANCE
    APPEARANCE = args.appearance
    fit = {path.stem: loadTable(path) for path in args.fit}
    check = {path.stem: loadTable(path) for path in args.check}
    scales = chooseScales(fit)
    print(f"motion score: offset scale {scales[0]} target sizes, log-size scale {scales[1]}")
    payload: dict[str, object] = {
        "motionScales": list(scales),
        "appearance": args.appearance,
        "metric": args.metric,
        "tables": {},
    }
    for group, tables in (("fit", fit), ("check", check)):
        for name, table in tables.items():
            info = describe(name, table, scales)
            payload["tables"][name] = {"group": group, **info}
            print(
                f"\n[{group}] {name}: {info['rows']} rows in {info['sequences']} sequences; "
                f"present {info['present']}, good {info['good']}, lost {info['lost']}, "
                f"absent {info['absent']}; variants {info['variants']}"
            )
            print("  Spearman        iou  backend  appearance  motion")
            for first, row in info["correlation"].items():
                print(f"  {first:<11}" + "".join(f"{value:>9.3f}" for value in row.values()))
            print("  alone           AUROC  caught@1%  caught@5%")
            for key, row in {**info["single"], "template only": info["templateOnly"]}.items():
                print(
                    f"  {key:<14}{row['auroc']:>7.3f}{row['caught@1%']:>11.3f}"
                    f"{row['caught@5%']:>11.3f}"
                )
    ranking = searchWeights(fit, scales, args.metric)
    best = tuple(ranking[0]["weights"])
    print(f"\nweights (backend, appearance, motion), mean {args.metric} over {list(fit)}")
    for row in ranking[:10]:
        print(f"  {row['weights']}  mean {row['mean']:.4f}  per table {np.round(row['auroc'], 4)}")
    references = {
        "backend only": (1.0, 0.0, 0.0),
        "appearance only": (0.0, 1.0, 0.0),
        "motion only": (0.0, 0.0, 1.0),
        "backend + appearance, equal": (0.5, 0.5, 0.0),
        "all equal": (0.35, 0.35, 0.3),
        "best on fit": best,
    }
    logistic = logisticWeights(fit, scales)
    print(f"\nlogistic regression on the fit tables: {logistic}")
    payload["ranking"] = ranking[:40]
    payload["logistic"] = logistic
    payload["evaluation"] = {}
    for label, weights in references.items():
        result = evaluate({**fit, **check}, scales, weights)
        payload["evaluation"][label] = {"weights": weights, "tables": result}
        print(f"\n{label} {weights}")
        for name, row in result.items():
            print(
                f"  {name:<22} AUROC {row['auroc']:.3f}  caught@1% {row['caught@1%']:.3f}"
                f" (fused < {row['threshold@1%']:.3f})  caught@5% {row['caught@5%']:.3f}"
                f" (fused < {row['threshold@5%']:.3f})"
            )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
