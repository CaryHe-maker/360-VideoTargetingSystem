"""Ways to turn the three scores into one lost / not-lost judgement, compared offline.

    python tools/score_rules.py --tune outputs/E018/tune_freerun.csv \\
        --holdout outputs/E018/holdout_freerun.csv --events outputs/E019/loss_events.csv \\
        --json outputs/E024/rules.json

Works on the per-frame tables of ``tools/fusion_freerun.py``; the tracker is not run.
Every rule gives one number per frame (smaller: more likely lost).  Rules with
parameters are fitted on the tune table only.

    fused        the weighted mean in use: 0.40 backend + 0.55 template + 0.05 motion
    linear       a logistic regression on backend, template, motion offset, size change
    quadratic    the same with squares and pairwise products: curved boundaries
    any-low      each score is replaced by its rank among the good frames of tune, and
                 the lowest rank counts: one score far out of its normal range is enough
                 however normal the others are
    +change      the rule also sees how far backend and template dropped below their
                 median over the last ten frames, and their lowest value in three frames

Each rule is measured as it is and latched (a flag holds until the number has been
clearly back for three frames), at the same false-alarm rate on good frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_analysis import Detector, loadTable, lossParts  # noqa: E402

RATES = (0.008, 0.02, 0.05)
BASE = ("backend", "template", "offset", "scale")
CHANGE_WINDOW, LOW_WINDOW = 10, 3
LATCH_MARGINS = (0.05, 0.10, 0.20)


def features(sequence: dict[str, np.ndarray], change: bool) -> np.ndarray:
    columns = [sequence[key] for key in BASE]
    if change:
        for key in ("backend", "template"):
            values = sequence[key]
            drop = np.zeros(len(values))
            low = np.zeros(len(values))
            for index in range(len(values)):
                past = values[max(0, index - CHANGE_WINDOW) : index]
                drop[index] = values[index] - (np.median(past) if len(past) else values[index])
                low[index] = values[max(0, index - LOW_WINDOW + 1) : index + 1].min()
            columns += [drop, low]
    return np.column_stack(columns)


def expand(matrix: np.ndarray) -> np.ndarray:
    """Add squares and pairwise products."""
    count = matrix.shape[1]
    extra = [matrix[:, i] * matrix[:, j] for i in range(count) for j in range(i, count)]
    return np.column_stack([matrix, *extra])


class Logistic:
    def __init__(self, quadratic: bool) -> None:
        self.quadratic = quadratic

    def _design(self, matrix: np.ndarray) -> np.ndarray:
        scaled = (matrix - self.mean) / self.spread
        if self.quadratic:
            scaled = expand(scaled)
        return np.column_stack([np.ones(len(scaled)), scaled])

    def fit(self, matrix: np.ndarray, good: np.ndarray) -> Logistic:
        self.mean, self.spread = matrix.mean(axis=0), matrix.std(axis=0) + 1e-9
        design = self._design(matrix)
        weights = np.zeros(design.shape[1])
        # Both classes count the same however many frames each has.
        sample = np.where(good, 0.5 / good.sum(), 0.5 / (~good).sum()) * len(good)
        for _ in range(50):
            probability = 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -30, 30)))
            gradient = design.T @ (sample * (probability - good)) + 1e-2 * weights
            curvature = (design * (sample * probability * (1 - probability))[:, None]).T @ design
            weights -= np.linalg.solve(curvature + 1e-2 * np.eye(len(weights)), gradient)
        self.weights = weights
        return self

    def score(self, matrix: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self._design(matrix) @ self.weights, -30, 30)))


class AnyLow:
    """The lowest rank of a frame's scores among the good frames of the fitting table."""

    def fit(self, matrix: np.ndarray, good: np.ndarray) -> AnyLow:
        self.reference = np.sort(matrix[good], axis=0)
        return self

    def score(self, matrix: np.ndarray) -> np.ndarray:
        ranks = [
            np.searchsorted(self.reference[:, column], matrix[:, column], side="right")
            / len(self.reference)
            for column in range(matrix.shape[1])
        ]
        return np.min(ranks, axis=0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tune", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    tune, holdout = loadTable(args.tune), loadTable(args.holdout)
    parts = {"tune": lossParts(tune, args.events), "holdout": lossParts(holdout, args.events)}
    tables = {"tune": tune, "holdout": holdout}

    def stack(table, change: bool):
        matrix = np.concatenate([features(seq, change) for seq in table.values()])
        good = np.concatenate([seq["good"] for seq in table.values()])
        lost = np.concatenate([seq["lost"] for seq in table.values()])
        return matrix, good, lost

    rules: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "fused": {
            split: {name: seq["fused"] for name, seq in table.items()}
            for split, table in tables.items()
        }
    }
    makers = {
        "linear": lambda: Logistic(False),
        "quadratic": lambda: Logistic(True),
        "any-low": AnyLow,
    }
    for change in (False, True):
        matrix, good, lost = stack(tune, change)
        keep = good | lost
        for label, maker in makers.items():
            model = maker().fit(matrix[keep], good[keep])
            name = label + (" +change" if change else "")
            rules[name] = {
                split: {seq: model.score(features(data, change)) for seq, data in table.items()}
                for split, table in tables.items()
            }

    payload: dict[str, object] = {}
    for rate in RATES:
        key = f"{rate:.1%}"
        payload[key] = {}
        print(f"\nfalse alarms {key}: detection on tune | hold-out (re-matched) | hold-out with")
        print("the tune threshold: its false alarms and detection; early = first five lost frames")
        print(
            f"  {'rule':<30}{'tune':>7}{'early':>7}{'hold':>8}{'early':>7}"
            f"{'FA':>8}{'det':>7}"
        )
        for name, signals in rules.items():
            for latched in (False, True):
                margin = None
                if latched:
                    # The release margin is chosen on tune, as the threshold is.
                    margin = max(
                        LATCH_MARGINS,
                        key=lambda m: Detector(signals["tune"], True, m).measure(
                            tune,
                            parts["tune"],
                            Detector(signals["tune"], True, m).threshold(tune, 0.02),
                        )["all"],
                    )
                onTune = Detector(signals["tune"], latched, margin)
                onHold = Detector(signals["holdout"], latched, margin)
                threshold = onTune.threshold(tune, rate)
                rows = {
                    "tune": onTune.measure(tune, parts["tune"], threshold),
                    "holdout": onHold.measure(
                        holdout, parts["holdout"], onHold.threshold(holdout, rate)
                    ),
                    "transfer": onHold.measure(holdout, parts["holdout"], threshold),
                }
                label = name + (f", latched +{margin:g}" if latched else "")
                payload[key][label] = rows
                print(
                    f"  {label:<30}{rows['tune']['all']:>7.1%}{rows['tune']['early']:>7.1%}"
                    f"{rows['holdout']['all']:>8.1%}{rows['holdout']['early']:>7.1%}"
                    f"{rows['transfer']['falseAlarm']:>8.1%}{rows['transfer']['all']:>7.1%}"
                )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
