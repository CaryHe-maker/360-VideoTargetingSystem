"""Score a parameter sweep against a baseline with the E082 overfitting checks.

    python tools/sweep_report.py --dataset-root <dir> --label-root <dir> \\
        --base outputs/E080a --runs enter60=outputs/E082enter60 ... --seeds 1,2,3,4,5 \\
        --out outputs/E082/round1

For every configuration (roots before ``_s<seed>``) and the baseline, on the dev-train
sequences: S_dual per sequence and seed, and from the per-sequence means over the
seeds:

- the difference to the baseline with a 95% interval from resampling sequences;
- the difference on each half of the fixed split (``--folds``);
- the mean difference without the two sequences that gained most, and how many
  sequences gained or lost more than 0.005;
- the difference of each seed;
- the regression check of the V2.1 plan: the summed difference to baseline 1
  (``--base1``) over the sequences where baseline 1 loses less than 5% of the frames,
  next to the baseline's;
- events per run: frames distrusted while wrong / right, jumps by what they did
  (wrong to right, right to wrong, wrong to wrong, right to right), releases and their
  share on the target, mean forwards per frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_report import sDual  # noqa: E402
from size_study import sequenceData  # noqa: E402

from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.loss_rate import dualIou  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402

SAMPLES = 10_000


def scores(dataset, root: str, seeds, names) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for seed in seeds:
        for name, value in sDual(dataset, Path(f"{root}_s{seed}"), names).items():
            out.setdefault(name, []).append(value)
    return out


def events(dataset, root: str, seeds, names) -> np.ndarray:
    t = np.zeros(11)
    for seed in seeds:
        run = Path(f"{root}_s{seed}")
        results = loadTrackerResults(run / "bbox" / "ours")
        for name in names:
            d = sequenceData(dataset, run, name, results)
            if d is None:
                continue
            present, iou = d["present"], d["iou"]

            def ok(f: int) -> bool:
                return f < len(present) and bool(present[f]) and iou[f] >= 0.5

            def bad(f: int) -> bool:
                return f < len(present) and ((not present[f]) or iou[f] < 0.1)

            with (run / "trace" / "ours" / f"{name}.csv").open(encoding="utf-8") as stream:
                for r in csv.DictReader(stream):
                    f = int(r["frame"])
                    untrusted = r["modeAfter"] != "TRACKING"
                    t[0] += untrusted and bad(f)
                    t[1] += untrusted and ok(f)
                    if r["action"] == "jump":
                        before, after = ok(f - 1), ok(f)
                        t[
                            2
                            + (
                                0
                                if not before and after
                                else 1
                                if before and not after
                                else 2
                                if not before
                                else 3
                            )
                        ] += 1
                    if r["reason"] == "RELEASED":
                        t[6] += 1
                        t[7] += ok(f)
                    t[8] += int(r["forwards"] or 1)
                    t[9] += 1
    return t / max(1, len(seeds))


def interval(diff: np.ndarray, rng) -> tuple[float, float]:
    means = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(SAMPLES)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument(
        "--sequence-file", type=Path, default=Path("configs/splits/360vos_dev_train.txt")
    )
    parser.add_argument(
        "--folds", type=Path, default=Path("configs/splits/360vos_dev_train_folds.json")
    )
    parser.add_argument("--base", required=True, help="baseline root before _s<seed>")
    parser.add_argument("--base1", default="outputs/E043_base1")
    parser.add_argument("--base1-seeds", default="1,2,3,4,5")
    parser.add_argument("--runs", nargs="+", required=True, help="label=root")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    names = readSequenceFile(args.sequence_file)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    seeds = [int(s) for s in args.seeds.split(",")]
    rng = np.random.default_rng(0)
    base = scores(dataset, args.base, seeds, names)
    base1 = scores(dataset, args.base1, [int(s) for s in args.base1_seeds.split(",")], names)
    # Where baseline 1 loses little (the regression check of the V2.1 plan, section 0.5):
    # the share of frames with the target present and IoU < 0.1, as the archive counts.
    loss1: dict[str, list[float]] = {}
    for seed in [int(s) for s in args.base1_seeds.split(",")]:
        results = loadTrackerResults(Path(f"{args.base1}_s{seed}") / "bbox" / "ours")
        for n in names:
            sequence = dataset.sequence(n)
            truth, present = sequence.groundTruth("bbox")
            width = sequence.frameSize[0]
            sequence.close()
            present = np.asarray(present, bool)
            iou = np.zeros(len(present))
            iou[present] = dualIou(truth[present], results[n][present], width)
            loss1.setdefault(n, []).append(float((present & (iou < 0.1)).sum()) / len(present))
    low = [n for n in names if np.mean(loss1[n]) < 0.05]
    baseMean = {n: float(np.mean(v)) for n, v in base.items()}
    base1Mean = {n: float(np.mean(v)) for n, v in base1.items()}
    baseEvents = events(dataset, args.base, seeds, names)
    lines = [
        f"seeds {seeds}; baseline {args.base}: S_dual {np.mean(list(baseMean.values())):.4f};"
        f" regression sum vs baseline 1 over {len(low)} low-loss sequences"
        f" {sum(baseMean[n] - base1Mean[n] for n in low):+.4f}",
        "label: S_dual | diff [95%] | fold A / B | w/o best 2 | up / down | per seed |"
        " regression vs base1 (baseline's)",
    ]
    summary = {}

    def eventLine(label: str, e: np.ndarray) -> str:
        return (
            f"    {label}: distrusted wrong/right {e[0]:.0f}/{e[1]:.0f} | jumps w>r {e[2]:.1f}"
            f" r>w {e[3]:.1f} w>w {e[4]:.1f} r>r {e[5]:.1f} | releases {e[6]:.1f}"
            f" ({e[7] / max(e[6], 1e-9):.1%}) | forwards/frame {e[8] / max(e[9], 1):.3f}"
        )

    for item in args.runs:
        label, root = item.split("=")
        own = scores(dataset, root, seeds, names)
        mean = {n: float(np.mean(v)) for n, v in own.items()}
        common = [n for n in names if n in mean and n in baseMean]
        diff = np.array([mean[n] - baseMean[n] for n in common])
        lo, hi = interval(diff, rng)
        foldA = np.mean([mean[n] - baseMean[n] for n in folds["A"] if n in mean])
        foldB = np.mean([mean[n] - baseMean[n] for n in folds["B"] if n in mean])
        ordered = np.sort(diff)[::-1]
        perSeed = [
            np.mean([own[n][i] for n in common]) - np.mean([base[n][i] for n in common])
            for i in range(len(seeds))
        ]
        regression = sum(mean[n] - base1Mean[n] for n in low if n in mean)
        lines.append(
            f"  {label}: {np.mean([mean[n] for n in common]):.4f} | {diff.mean():+.4f}"
            f" [{lo:+.4f}, {hi:+.4f}] | {foldA:+.4f} / {foldB:+.4f} |"
            f" {ordered[2:].mean():+.4f} | {(diff > 0.005).sum()} / {(diff < -0.005).sum()} |"
            f" {' '.join(f'{x:+.3f}' for x in perSeed)} | {regression:+.4f}"
        )
        top = sorted(zip(common, diff, strict=True), key=lambda x: -abs(x[1]))[:5]
        lines.append("    largest: " + ", ".join(f"{n} {d:+.3f}" for n, d in top))
        e = events(dataset, root, seeds, names)
        lines.append(eventLine("events", e))
        summary[label] = {
            "diff": float(diff.mean()),
            "interval": [lo, hi],
            "foldA": float(foldA),
            "foldB": float(foldB),
            "withoutBest2": float(ordered[2:].mean()),
            "up": int((diff > 0.005).sum()),
            "down": int((diff < -0.005).sum()),
            "perSeed": [float(x) for x in perSeed],
            "regression": float(regression),
            "perSequence": {n: float(mean[n] - baseMean[n]) for n in common},
            "events": e.tolist(),
        }
    lines.append(eventLine("baseline events", baseEvents))
    text = "\n".join(lines)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.txt").write_text(text + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
