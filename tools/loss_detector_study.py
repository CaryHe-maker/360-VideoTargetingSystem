"""Search for the best way to tell lost frames from the scores of each frame.

    python tools/loss_detector_study.py \\
        --tune outputs/E018/tune_freerun.csv --tune-run outputs/E016f_tune \\
        --holdout outputs/E018/holdout_freerun.csv --holdout-run outputs/E016f_holdout \\
        --events outputs/E019/loss_events.csv --out outputs/E025

Works on a free-running probed run (``tools/benchmark.py run --probe dinov2``) and its
per-frame table (``tools/fusion_freerun.py``); the tracker is not run.

Every frame gets a set of features computed from the present and earlier frames only:
the raw scores, transformed scores, how the scores changed, how much the image in the
box changed from frame to frame, and how the box itself moved and changed.  The feature
table is written to ``<out>/features_<split>.csv``.

A detector is a logistic regression on a subset of the features.  The subset is grown
greedily: at each step the feature is added that most raises the detection of lost
frames at fixed false-alarm rates, measured on sequences the model was not fitted on
(folds of whole sequences).  Selecting on the 25 tune sequences alone picked features
that describe the sequences rather than the loss (the target's size) and failed on
hold-out, so the selection uses all sequences of both tables, and the features that
identify a sequence are left out.  Detection is reported for the tune and the hold-out
sequences separately, always from models that did not see the sequence; a fit on tune
applied to hold-out shows how well a threshold carries over.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_analysis import Detector, loadTable, lossParts  # noqa: E402

from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402

RATES = (0.008, 0.02, 0.05)
FOLDS = 10
MAX_FEATURES = 8
MIN_GAIN = 0.005
# Describes the sequence, not the frame: a model can use it to recognise sequences.
EXCLUDED = ("sizeLog",)
# A latched flag is released this far above the threshold (logit units; 0.2 for fused).
LOGIT_RELEASE = 1.0
INITIAL_FRAMES = 30
REFERENCE_SETS = {
    "raw (4)": ("backend", "template", "offset", "scale"),
    "smoothed pair (2)": ("backendMean5", "template"),
    "pair + trends (4)": ("backendMean5", "template", "backendEmaGap", "templateEmaGap"),
    "raw + change (E024)": (
        "backend",
        "template",
        "offset",
        "scale",
        "backendDrop10",
        "templateDrop10",
        "backendMin3",
        "templateMin3",
    ),
}


def _rolling(values: np.ndarray, window: int, reducer, *, includeNow: bool) -> np.ndarray:
    """``reducer`` over the last ``window`` values before (or up to) each frame."""
    result = np.zeros(len(values))
    for index in range(len(values)):
        stop = index + 1 if includeNow else index
        past = values[max(0, stop - window) : stop]
        result[index] = reducer(past) if len(past) else values[index]
    return result


def _ema(values: np.ndarray, span: float) -> np.ndarray:
    result, rate = np.zeros(len(values)), 2.0 / (span + 1.0)
    state = values[0] if len(values) else 0.0
    for index, value in enumerate(values):
        state += rate * (value - state)
        result[index] = state
    return result


def _cusum(deviation: np.ndarray, slack: float) -> np.ndarray:
    """Accumulated shortfall: grows while the value stays below its usual level."""
    result, state = np.zeros(len(deviation)), 0.0
    for index, value in enumerate(deviation):
        state = max(0.0, state - value - slack)
        result[index] = state
    return result


def sequenceFeatures(
    sequence: dict[str, np.ndarray], appearance: np.ndarray, sphere: np.ndarray
) -> dict[str, np.ndarray]:
    frames = sequence["frame"]
    backend, template = sequence["backend"], sequence["template"]
    offset, scale = -sequence["offset"], -sequence["scale"]
    feature = appearance[frames].astype(np.float64)
    previous = appearance[frames - 1].astype(np.float64)
    continuity = np.sum(feature * previous, axis=1)
    recent = np.zeros(len(frames))
    for index, frame in enumerate(frames):
        window = appearance[max(0, frame - 5) : frame].astype(np.float64).mean(axis=0)
        recent[index] = float(feature[index] @ window) / (np.linalg.norm(window) + 1e-9)
    # The first frames of a sequence are the only ones known to show the target.
    initialSoFar = np.asarray(
        [np.median(template[: min(index, INITIAL_FRAMES)]) if index else template[0]
         for index in range(len(template))]
    )
    box = sphere[frames]
    before = sphere[np.maximum(frames - 5, 0)]
    size = np.sqrt(np.clip(box[:, 2] * box[:, 3], 1e-6, None))
    sizeBefore = np.sqrt(np.clip(before[:, 2] * before[:, 3], 1e-6, None))
    aspect = np.log(np.clip(box[:, 2], 1e-6, None) / np.clip(box[:, 3], 1e-6, None))
    aspectBefore = np.log(np.clip(before[:, 2], 1e-6, None) / np.clip(before[:, 3], 1e-6, None))
    longitude, latitude = np.radians(sphere[:, 0]), np.radians(sphere[:, 1])
    direction = np.column_stack(
        [
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
            np.cos(latitude) * np.cos(longitude),
        ]
    )
    step = np.degrees(
        np.arccos(np.clip(np.sum(direction[frames] * direction[frames - 1], axis=1), -1, 1))
    ) / size
    backendDrop30 = backend - _rolling(backend, 30, np.median, includeNow=False)
    templateDrop30 = template - _rolling(template, 30, np.median, includeNow=False)
    clipped = np.clip(backend, 1e-4, 1 - 1e-4)
    return {
        "backend": backend,
        "template": template,
        "offset": offset,
        "scale": scale,
        "backendLogit": np.log(clipped / (1 - clipped)),
        "offsetLog": np.log(offset + 0.01),
        "scaleLog": np.log(scale + 0.005),
        "backendDrop10": backend - _rolling(backend, 10, np.median, includeNow=False),
        "templateDrop10": template - _rolling(template, 10, np.median, includeNow=False),
        "backendDrop30": backendDrop30,
        "templateDrop30": templateDrop30,
        "backendMin3": _rolling(backend, 3, np.min, includeNow=True),
        "templateMin3": _rolling(template, 3, np.min, includeNow=True),
        "backendMean5": _rolling(backend, 5, np.mean, includeNow=True),
        "templateMean5": _rolling(template, 5, np.mean, includeNow=True),
        "backendStd10": _rolling(backend, 10, np.std, includeNow=True),
        "templateStd10": _rolling(template, 10, np.std, includeNow=True),
        "backendEmaGap": _ema(backend, 3) - _ema(backend, 30),
        "templateEmaGap": _ema(template, 3) - _ema(template, 30),
        "backendCusum": _cusum(backendDrop30, 0.02),
        "templateCusum": _cusum(templateDrop30, 0.02),
        "templateVsStart": template - initialSoFar,
        "templateRatioStart": template / np.clip(initialSoFar, 0.05, None),
        "continuity": continuity,
        "continuity5": recent,
        "continuityMin3": _rolling(continuity, 3, np.min, includeNow=True),
        "continuityDrop10": continuity - _rolling(continuity, 10, np.median, includeNow=False),
        "sizeChange5": np.abs(np.log(size / sizeBefore)),
        "aspectChange5": np.abs(aspect - aspectBefore),
        "step": step,
        "stepLog": np.log(step + 0.01),
        "stepChange": np.abs(step - np.r_[step[:1], step[:-1]]),
        "sizeLog": np.log(size),
    }


def build(tablePath: Path, runRoot: Path, method: str, output: Path) -> dict[str, dict]:
    table = loadTable(tablePath)
    spheres = loadTrackerResults(runRoot / "bfov" / method)
    names: list[str] = []
    for name, sequence in table.items():
        appearance = np.load(runRoot / "probe" / method / "features" / f"{name}.npy")
        sequence["features"] = sequenceFeatures(sequence, appearance, spheres[name])
        names = list(sequence["features"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", "frame", "good", "lost", *names])
        for name, sequence in table.items():
            for index, frame in enumerate(sequence["frame"]):
                writer.writerow(
                    [name, int(frame), int(sequence["good"][index]), int(sequence["lost"][index])]
                    + [round(float(sequence["features"][key][index]), 6) for key in names]
                )
    return table


def matrixOf(table, names, sequences=None):
    chosen = list(table) if sequences is None else sequences
    matrix = np.concatenate(
        [np.column_stack([table[seq]["features"][key] for key in names]) for seq in chosen]
    )
    good = np.concatenate([table[seq]["good"] for seq in chosen])
    lost = np.concatenate([table[seq]["lost"] for seq in chosen])
    return matrix, good, lost


class Logistic:
    def fit(self, matrix: np.ndarray, good: np.ndarray) -> Logistic:
        # Clip to the bulk of the training values so single extreme frames do not decide.
        self.low, self.high = np.percentile(matrix, [0.5, 99.5], axis=0)
        clipped = np.clip(matrix, self.low, self.high)
        self.mean, self.spread = clipped.mean(axis=0), clipped.std(axis=0) + 1e-9
        design = self._design(matrix)
        weights = np.zeros(design.shape[1])
        sample = np.where(good, 0.5 / good.sum(), 0.5 / (~good).sum()) * len(good)
        for _ in range(40):
            probability = 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -30, 30)))
            gradient = design.T @ (sample * (probability - good)) + 1e-2 * weights
            curvature = (design * (sample * probability * (1 - probability))[:, None]).T @ design
            weights -= np.linalg.solve(curvature + 1e-2 * np.eye(len(weights)), gradient)
        self.weights = weights
        return self

    def _design(self, matrix: np.ndarray) -> np.ndarray:
        scaled = (np.clip(matrix, self.low, self.high) - self.mean) / self.spread
        return np.column_stack([np.ones(len(scaled)), scaled])

    def score(self, matrix: np.ndarray) -> np.ndarray:
        """The logit: larger means more likely tracked."""
        return self._design(matrix) @ self.weights


def fitOn(table, names, sequences) -> Logistic:
    matrix, good, lost = matrixOf(table, names, sequences)
    keep = good | lost
    return Logistic().fit(matrix[keep], good[keep])


def signalsOf(model: Logistic, table, names, sequences=None) -> dict[str, np.ndarray]:
    chosen = list(table) if sequences is None else sequences
    return {
        seq: model.score(np.column_stack([table[seq]["features"][key] for key in names]))
        for seq in chosen
    }


def crossValidated(table, names, folds) -> dict[str, np.ndarray]:
    """Scores of every sequence from a model that did not see it."""
    signals: dict[str, np.ndarray] = {}
    for held in folds:
        rest = [seq for seq in table if seq not in held]
        signals.update(signalsOf(fitOn(table, names, rest), table, names, held))
    return signals


def detection(table, parts, signals, latched=False, margin=None) -> dict[str, dict]:
    detector = Detector(signals, latched, margin)
    return {
        f"{rate:.1%}": detector.measure(table, parts, detector.threshold(table, rate))
        for rate in RATES
    }


def meanDetection(rows: dict[str, dict]) -> float:
    return float(np.mean([row["all"] for row in rows.values()]))


def auroc(values: np.ndarray, good: np.ndarray, lost: np.ndarray) -> float:
    order = np.argsort(np.concatenate([values[good], values[lost]]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    goodCount, lostCount = int(good.sum()), int(lost.sum())
    area = (ranks[:goodCount].sum() - goodCount * (goodCount + 1) / 2) / (goodCount * lostCount)
    return float(max(area, 1 - area))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tune", type=Path, required=True)
    parser.add_argument("--tune-run", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--holdout-run", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--method", default="ours")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--features",
        default=None,
        help="comma-separated feature set to report instead of searching for one",
    )
    args = parser.parse_args(argv)
    tune = build(args.tune, args.tune_run, args.method, args.out / "features_tune.csv")
    holdout = build(args.holdout, args.holdout_run, args.method, args.out / "features_holdout.csv")
    both = {**tune, **holdout}
    parts = {**lossParts(tune, args.events), **lossParts(holdout, args.events)}
    names = [name for name in next(iter(tune.values()))["features"] if name not in EXCLUDED]
    order = sorted(both)
    rng = np.random.default_rng(0)
    rng.shuffle(order)
    folds = [order[index::FOLDS] for index in range(FOLDS)]
    payload: dict[str, object] = {"folds": folds}
    splits = {"tune": tune, "holdout": holdout, "all": both}

    def subset(signals: dict[str, np.ndarray], table) -> dict[str, np.ndarray]:
        return {name: signals[name] for name in table}

    # 1. How much each feature says on its own, on each table.
    alone = {}
    for split in ("tune", "holdout"):
        matrix, good, lost = matrixOf(splits[split], names)
        alone[split] = {
            name: auroc(matrix[:, index], good, lost) for index, name in enumerate(names)
        }
    payload["aurocAlone"] = alone
    print("each feature alone (AUROC, good frames against lost frames): tune, hold-out")
    for name in sorted(names, key=lambda name: -min(alone["tune"][name], alone["holdout"][name])):
        print(f"  {name:<22}{alone['tune'][name]:>7.3f}{alone['holdout'][name]:>7.3f}")

    # 2. Greedy selection over all sequences, scored on sequences left out of the fit.
    chosen: list[str] = []
    best = 0.0
    steps = []
    if args.features:
        chosen = args.features.split(",")
    while not args.features and len(chosen) < MAX_FEATURES:
        trials = {
            name: meanDetection(
                detection(both, parts, crossValidated(both, [*chosen, name], folds))
            )
            for name in names
            if name not in chosen
        }
        name = max(trials, key=trials.get)
        if trials[name] - best < MIN_GAIN:
            break
        chosen.append(name)
        best = trials[name]
        steps.append({"added": name, "meanDetection": best})
        print(f"step {len(chosen)}: + {name:<22} mean detection {best:.1%}")
    payload["selection"] = steps

    # 3. The selected detector and the reference ones.
    sets = {
        **REFERENCE_SETS,
        f"selected ({len(chosen)})": tuple(chosen),
        "all features": tuple(names),
    }
    report: dict[str, object] = {}
    entries: dict[str, dict] = {
        "fused (in use)": {
            "left out": {name: seq["fused"] for name, seq in both.items()},
            "from tune": {name: seq["fused"] for name, seq in holdout.items()},
            "tuneSelf": {name: seq["fused"] for name, seq in tune.items()},
            "margin": 0.2,
        }
    }
    for label, members in sets.items():
        fromTune = fitOn(tune, list(members), list(tune))
        final = fitOn(both, list(members), list(both))
        entries[label] = {
            "left out": crossValidated(both, list(members), folds),
            "from tune": signalsOf(fromTune, holdout, list(members)),
            "tuneSelf": signalsOf(fromTune, tune, list(members)),
            "margin": LOGIT_RELEASE,
            "weights": dict(zip(["bias", *members], map(float, final.weights), strict=True)),
            "low": dict(zip(members, map(float, final.low), strict=True)),
            "high": dict(zip(members, map(float, final.high), strict=True)),
            "mean": dict(zip(members, map(float, final.mean), strict=True)),
            "spread": dict(zip(members, map(float, final.spread), strict=True)),
        }
    for label, entry in entries.items():
        report[label] = {
            key: entry.get(key) for key in ("weights", "low", "high", "mean", "spread")
        }
        for latched in (False, True):
            margin = entry["margin"] if latched else None
            key = "latched" if latched else "plain"
            onTune = Detector(entry["tuneSelf"], latched, margin)
            onHold = Detector(entry["from tune"], latched, margin)
            report[label][key] = {
                split: detection(table, parts, subset(entry["left out"], table), latched, margin)
                for split, table in splits.items()
            }
            report[label][key]["transfer"] = {
                f"{rate:.1%}": onHold.measure(holdout, parts, onTune.threshold(tune, rate))
                for rate in RATES
            }
    payload["detectors"] = report
    for rate in RATES:
        key = f"{rate:.1%}"
        print(f"\nfalse alarms {key}. Detection on sequences left out of the fit: all, tune,")
        print("hold-out (early = first five lost frames); then a fit on tune with its threshold")
        print("applied to hold-out: the false alarms and the detection it gives there")
        print(
            f"  {'detector':<34}{'all':>7}{'early':>7}{'tune':>7}{'hold':>7}"
            f"{'FA':>8}{'det':>7}"
        )
        for label, item in report.items():
            for mode in ("plain", "latched"):
                rows = item[mode]
                print(
                    f"  {label + (', latched' if mode == 'latched' else ''):<34}"
                    f"{rows['all'][key]['all']:>7.1%}{rows['all'][key]['early']:>7.1%}"
                    f"{rows['tune'][key]['all']:>7.1%}{rows['holdout'][key]['all']:>7.1%}"
                    f"{rows['transfer'][key]['falseAlarm']:>8.1%}"
                    f"{rows['transfer'][key]['all']:>7.1%}"
                )
    selected = report[f"selected ({len(chosen)})"]["weights"]
    print("\nweights of the selected detector (features standardised; positive: looks tracked)")
    for name, value in selected.items():
        print(f"  {name:<22}{value:+.3f}")
    (args.out / "study.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
