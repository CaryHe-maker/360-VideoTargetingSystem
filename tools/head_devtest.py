"""Freeze judges on dev train and score them once on dev test (evaluation log E049/E050).

    python tools/head_devtest.py candidates --train-runs outputs/E049/runs \\
        --test-runs outputs/E050/trace --heads "Q, linear" --out outputs/E050/candidates
    python tools/head_devtest.py detection --dataset-root <dir> --label-root <dir> \\
        --train-signals outputs/E045 --test-signals outputs/E050/signals \\
        --features outputs/E025/features_tune.csv --features outputs/E025/features_holdout.csv \\
        --heads "P1, trees" --out outputs/E050/detection

Each named head is learned on all dev-train sequences exactly as in round 4 of E049
(hyper-parameters by 3-fold cross-validation by sequence, then a fit on everything),
then frozen together with its thresholds: the score that lets 5% of the dev-train
negatives through, and the score that accepts the same share of dev-train rows as
the reference rule.  Nothing is chosen on dev test.  Reported on dev test: AUROC of
each head and reference, the gain over the best reference with a 95% interval from
resampling dev-test sequences, the false alarms and catches at the frozen threshold,
and the worst sequence.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from head_study_r4 import Head, auc, candidateData, detectionData  # noqa: E402


def parseHead(text: str) -> Head:
    quantity, _, rest = text.partition(",")
    rest = rest.strip()
    kind = "trees" if rest.startswith("trees") else "linear"
    return Head(quantity.strip(), kind, "removed" in rest)


def bootstrap(y, g, s, r, samples=2000):
    names = np.unique(g)
    index = {name: np.flatnonzero(g == name) for name in names}
    rng = np.random.default_rng(0)
    gains = []
    for _ in range(samples):
        pick = np.concatenate([index[name] for name in rng.choice(names, len(names))])
        gains.append(auc(y[pick], s[pick]) - auc(y[pick], r[pick]))
    return float(np.nanpercentile(gains, 2.5)), float(np.nanpercentile(gains, 97.5))


def worst(y, g, s):
    values = [
        auc(y[g == name], s[g == name])
        for name in np.unique(g)
        if y[g == name].sum() >= 5 and (~y[g == name]).sum() >= 5
    ]
    return float(np.min(values)) if values else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("side", choices=("detection", "candidates"))
    parser.add_argument("--dataset-root")
    parser.add_argument("--label-root")
    parser.add_argument("--train-runs", type=Path)
    parser.add_argument("--test-runs", type=Path)
    parser.add_argument("--train-signals", type=Path)
    parser.add_argument("--test-signals", type=Path)
    parser.add_argument("--features", type=Path, action="append")
    parser.add_argument("--heads", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    catchNegatives = args.side == "detection"
    if args.side == "candidates":
        train, trainRefs = candidateData(args.train_runs)
        test, testRefs = candidateData(args.test_runs)
    else:
        args.signals = args.train_signals
        train, trainRefs = detectionData(args)
        args.signals = args.test_signals
        test, testRefs = detectionData(args)
    common = sorted(set(train["groups"]) & set(test["groups"]))
    if common:
        raise SystemExit(f"dev train and dev test share sequences: {common}")
    rows, testRows = train["decisionRows"], test["decisionRows"]
    y, yt = train["decision"][rows], test["decision"][testRows]
    g = test["groups"][testRows]
    print(
        f"dev train: {len(np.unique(train['groups']))} sequences; dev test:"
        f" {len(np.unique(test['groups']))} sequences, {int(yt.sum())} positive and"
        f" {int((~yt).sum())} negative decision rows"
    )
    results: dict[str, dict] = {}
    scores: dict[str, np.ndarray] = {}
    trainScores: dict[str, np.ndarray] = {}
    for name, values in testRefs.items():
        scores[name], trainScores[name] = values[testRows], trainRefs[name][rows]
    for text in args.heads:
        head = parseHead(text).fit(train, np.arange(len(train["groups"])), seed=0)
        scores[head.name] = head.score(test, np.arange(len(test["groups"])))[testRows]
        trainScores[head.name] = head.score(train, np.arange(len(train["groups"])))[rows]
        results[head.name] = {"params": head.params_}
    # The reference to beat is the best one on dev train: nothing is chosen on dev test.
    best = max(testRefs, key=lambda name: auc(y, trainScores[name]))
    share = train.get("referenceShare")
    print(
        f"\n  {'judge':<46}{'AUROC':>7}{'catch@5%':>10}{'false':>8}{'worst':>8}"
        f"{'cnt.prec':>10}{'cnt.x':>7}"
    )
    for name, s in scores.items():
        st = trainScores[name]
        entry = results.setdefault(name, {})
        entry["auroc"] = auc(yt, s)
        if catchNegatives:
            cut = float(np.quantile(st[y], 0.05))
            flagged = s < cut
            entry["catch"], entry["falseAlarm"] = (
                float(flagged[~yt].mean()),
                float(flagged[yt].mean()),
            )
        else:
            cut = float(np.quantile(st[~y], 0.95))
            taken = s > cut
            entry["catch"], entry["falseAlarm"] = float(taken[yt].mean()), float(taken[~yt].mean())
        entry["worstSequenceAuc"] = worst(yt, g, s)
        line = (
            f"  {name:<46}{entry['auroc']:>7.3f}{entry['catch']:>10.1%}{entry['falseAlarm']:>8.1%}"
            f"{entry['worstSequenceAuc']:>8.3f}"
        )
        if share is not None:
            cut = float(np.quantile(trainScores[name], 1.0 - share))
            taken = s > cut
            entry["countPrecision"] = float(yt[taken].mean()) if taken.any() else float("nan")
            entry["countShareRatio"] = float(taken.mean() / share)
            line += f"{entry['countPrecision']:>10.1%}{entry['countShareRatio']:>7.2f}"
        print(line)
    print(f"\ngain over the best dev-train reference ('{best}'), dev-test sequences resampled:")
    for name in scores:
        if name in testRefs:
            continue
        low, high = bootstrap(yt, g, scores[name], scores[best])
        gain = results[name]["auroc"] - results[best]["auroc"]
        results[name]["gain"] = {"reference": best, "gain": gain, "low": low, "high": high}
        print(f"  {name:<46}{gain:+.3f} [{low:+.3f}, {high:+.3f}]")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "devtest.json").write_text(
        json.dumps(results, indent=1, default=str) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
