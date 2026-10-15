"""Train, export and set the operating points of the candidate judge (evaluation log E051).

    python tools/candidate_judge_export.py --runs outputs/E049/runs \\
        --base-runs outputs/E043_base2_s1 ... --out outputs/E051/judge

The judge is the E049 round-6 combination, P1 (trees) x Q (linear).  It is trained
four times by sequence (GroupKFold, 4 folds), each time without one quarter of the
dev-train sequences, and once on all of them.  In a tracking run a sequence is judged
by the judge that never saw it, so dev train can be used to test the judge in the
tracker.  The export holds plain arrays only; it is checked against scikit-learn.

The two thresholds are set on the out-of-fold scores of every candidate, before any
tracking run, against the current rule (search score >= 0.70) on the same candidates:

    T1  the lowest threshold whose accepted candidates are at least as often on the
        target (IoU >= 0.5) as those of the current rule;
    T2  the threshold that accepts as many candidates as the current rule.

Also reported: the extra forwards the judge would cost on the recorded runs, per
frame and per sequence, and how many targets a search-score prefilter would drop.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from head_study_r4 import Combo, candidateData  # noqa: E402

from track360.runtime.candidate_judge import CandidateJudge, treeArrays, writeJudges  # noqa: E402

FOLDS = 4
RULE_SCORE = 0.70
PREFILTERS = (0.1, 0.2, 0.3, 0.4, 0.5)


def exportCombo(combo: Combo, sequences: list[str]) -> dict[str, np.ndarray]:
    p, q = combo.heads_["P1"].model_, combo.heads_["Q"].model_
    pca, trees = p.named_steps["pca"], p.named_steps["model"]
    clip, scale, ridge = (q.named_steps[k] for k in ("clip", "scale", "model"))
    return {
        "sequences": np.asarray(sequences),
        "qLow": clip.low_,
        "qHigh": clip.high_,
        "qMean": scale.mean_,
        "qScale": scale.scale_,
        "qCoef": ridge.coef_,
        "qIntercept": np.asarray(float(ridge.intercept_)),
        "pTokenMean": pca.scale_.mean_,
        "pTokenScale": pca.scale_.scale_,
        "pPcaMean": pca.pca_.mean_,
        "pPcaComponents": pca.pca_.components_,
        **treeArrays(
            [round_[0] for round_ in trees._predictors],
            float(np.ravel(trees._baseline_prediction)[0]),
        ),
    }


def lowestAtPrecision(scores: np.ndarray, correct: np.ndarray, target: float) -> tuple[float, int]:
    order = np.argsort(-scores, kind="stable")
    hits = np.cumsum(correct[order])
    precision = hits / np.arange(1, len(order) + 1)
    passing = np.flatnonzero(precision >= target)
    k = int(passing.max()) + 1
    return float(scores[order][k - 1]), k


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--base-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    data, references = candidateData(args.runs)
    groups = data["groups"]
    rows = np.arange(len(groups))
    names = np.unique(groups)
    print(f"{len(rows)} candidates, {len(names)} sequences")

    judges: dict[str, dict[str, np.ndarray]] = {}
    oof = np.full(len(rows), np.nan)
    folds = GroupKFold(n_splits=FOLDS, shuffle=True, random_state=0)
    fitted: dict[str, tuple[Combo, np.ndarray]] = {}
    for fold, (train, held) in enumerate(folds.split(rows, groups=groups)):
        combo = Combo("P1treesxQ").fit(data, rows[train], seed=0)
        oof[held] = combo.score(data, rows[held])
        heldNames = sorted(set(groups[held]))
        judges[f"fold{fold}"] = exportCombo(combo, heldNames)
        fitted[f"fold{fold}"] = (combo, held)
        print(f"  fold {fold}: {len(heldNames)} held sequences, params {combo.params_}")
    full = Combo("P1treesxQ").fit(data, rows, seed=0)
    judges["all"] = exportCombo(full, [])
    fitted["all"] = (full, rows)
    print(f"  all: params {full.params_}")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "candidate_judge.npz"
    writeJudges(path, judges)

    # The exported arrays must give scikit-learn's scores.  Each part is compared on
    # scikit-learn's own inputs; the whole judge differs on the few rows that lie on a
    # tree's split point, where rounding at 1e-15 decides the side.
    judge = CandidateJudge(path, threshold=0.0)
    worstGap, flipped = 0.0, 0
    for label, (combo, part) in fitted.items():
        exported = judge._judges[label]
        x = data["x"][part]
        p, q = combo.heads_["P1"].model_, combo.heads_["Q"].model_
        reduced = p.named_steps["pca"].transform(x)
        treesGap = np.abs(exported.trees.probability(reduced) - p.predict_proba(x)[:, 1])
        scaled = (np.clip(x, exported.qLow, exported.qHigh) - exported.qMean) / exported.qScale
        qGap = np.abs(scaled @ exported.qCoef + exported.qIntercept - q.predict(x))
        worstGap = max(worstGap, float(treesGap.max()), float(qGap.max()))
        flipped += int((np.abs(exported.score(x) - combo.score(data, part)) > 1e-6).sum())
    total = sum(len(item[1]) for item in fitted.values())
    print(
        f"export check: largest difference of a part {worstGap:.2e}; whole judge differs on"
        f" {flipped} of {total} rows"
    )
    if worstGap > 1e-9 or flipped > 0.005 * total:
        raise SystemExit("the exported judge does not reproduce scikit-learn")

    search = references["reference: search score"]
    correct = data["iou"] >= 0.5
    accepted = search >= RULE_SCORE
    p0, count = float(correct[accepted].mean()), int(accepted.sum())
    t1, k1 = lowestAtPrecision(oof, correct, p0)
    order = np.sort(oof)[::-1]
    t2 = float(order[count - 1])
    hit = lambda taken: (int(taken.sum()), int((taken & correct).sum()))  # noqa: E731
    lines = [
        f"current rule (search >= {RULE_SCORE}): accepts {count}, on target"
        f" {int((accepted & correct).sum())} ({p0:.1%}); targets among candidates"
        f" {int(correct.sum())}",
    ]
    operating = {}
    for name, threshold in (("T1", t1), ("T2", t2)):
        taken = oof >= threshold
        n, good = hit(taken)
        operating[name] = {"threshold": threshold, "accepted": n, "onTarget": good}
        lines.append(
            f"{name}: judge >= {threshold:.4f} accepts {n}, on target {good}"
            f" ({good / n:.1%}); {good / max(1, int((accepted & correct).sum())):.2f}x"
            " the targets of the current rule"
        )

    # Cost on the recorded runs: one extra pass per candidate.
    perRun = {}
    for base in args.base_runs:
        traced = args.runs / base.name / "trace_candidates.npz"
        if not traced.exists():
            continue
        found = np.load(traced)["sequence"]
        reports = sorted((base / "reports" / "ours").glob("*.json"))
        rows_ = []
        for report in reports:
            if report.name == "run.json":
                continue
            entry = json.loads(report.read_text(encoding="utf-8"))
            extra = int((found == entry["sequence"]).sum())
            rows_.append((entry["sequence"], entry["frames"], entry["forwards"], extra))
        perRun[base.name] = rows_
    lines.append("\ncost on the recorded base-2 runs (forwards per frame):")
    lines.append(f"  {'run':<16}{'now':>8}{'judge':>8}{'max now':>9}{'max judge':>11}")
    costs = {}
    for run, rows_ in perRun.items():
        frames = sum(r[1] for r in rows_)
        now = sum(r[2] for r in rows_) / frames
        judged = sum(r[2] + r[3] for r in rows_) / frames
        maxNow = max(r[2] / r[1] for r in rows_)
        maxJudged = max((r[2] + r[3]) / r[1] for r in rows_)
        costs[run] = {"now": now, "judge": judged, "maxNow": maxNow, "maxJudge": maxJudged}
        lines.append(f"  {run:<16}{now:>8.3f}{judged:>8.3f}{maxNow:>9.3f}{maxJudged:>11.3f}")

    lines.append("\nprefilter on the search score: candidates kept, targets lost")
    prefilter = {}
    for low in PREFILTERS:
        kept = search >= low
        lost = int((correct & ~kept).sum())
        prefilter[low] = {"kept": float(kept.mean()), "targetsLost": lost / int(correct.sum())}
        lines.append(
            f"  search >= {low:.1f}: keeps {kept.mean():.1%} of candidates,"
            f" loses {lost} of {int(correct.sum())} targets ({lost / correct.sum():.1%})"
        )
    text = "\n".join(lines)
    print(text)
    (args.out / "operating_points.json").write_text(
        json.dumps(
            {
                "ruleShareOnTarget": p0,
                "ruleAccepted": count,
                "operatingPoints": operating,
                "cost": costs,
                "prefilter": prefilter,
                "exportGap": worstGap,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.out / "oof_scores.npz", score=oof, sequence=groups, iou=data["iou"], search=search
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
