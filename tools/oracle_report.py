"""Summarise partial-oracle runs (evaluation log E052).

    python tools/oracle_report.py --dataset-root <dir> --label-root <dir> \\
        --run A=outputs/E052_Ahi --run E=outputs/E052_Ehi --seeds 1,2 --out outputs/E052/phase1

For every configuration and seed, the S_dual of all dev-train sequences: a run made
only on some sequences is completed with the same seed's base-2 run (``--fill``),
whose result those sequences would have had (an oracle that is never drawn changes
nothing).  The S_dual difference to base 2 and to base 1 of the same seeds, with a
95% interval from resampling sequences (paired, means over seeds), and the levels the
stages reached: detection rate, gates 1-3 with any candidate on the target, and the
precision of the jumps.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_report  # noqa: E402
import loss_report  # noqa: E402
import search_outcomes  # noqa: E402

from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.vot360_metrics import evaluateVot360, loadTrackerResults  # noqa: E402

SAMPLES = 10_000


def sDual(dataset: Vot360Dataset, root: Path, names: list[str]) -> dict[str, float]:
    """S_dual per sequence, as the benchmark tool scores it."""
    results = loadTrackerResults(root / "bbox" / "ours")
    out = {}
    for name in names:
        if name not in results:
            continue
        sequence = dataset.sequence(name)
        truth = {name: sequence.groundTruth("bbox")}
        width, height = sequence.frameSize
        sequence.close()
        scores = evaluateVot360(
            truth, {name: results[name]}, "bbox", frameWidthPx=width, frameHeightPx=height
        )
        out[name] = float(scores.perSequence[name].success)
    return out


def levels(args: argparse.Namespace, root: Path, only: set[str] | None) -> dict[str, float]:
    ns = argparse.Namespace(
        dataset_root=args.dataset_root,
        label_root=args.label_root,
        output_root=root,
        method="ours",
        only=only,
        tier=None,
        baseline_p50_ms=None,
    )
    notice = loss_report.noticing(ns)
    attempts = [a for a in search_outcomes.analyse(ns) if only is None or a["sequence"] in only]
    enlarged = [a for a in attempts if a["scan"] != "1x" and a["present"]]
    inView = [a for a in enlarged if float(a.get("reach", 9.0)) <= 1.0]
    table = judge_report.counts(root, ns, Vot360Dataset(args.dataset_root, args.label_root))
    totals = sum(v for k, v in table.items() if only is None or k in only)
    return {
        "detectionRate": notice["detectionRate"],
        "gate1": len(inView) / len(enlarged) if enlarged else float("nan"),
        "gate2": totals[1] / totals[0] if totals[0] else float("nan"),
        "gate3": totals[2] / totals[1] if totals[1] else float("nan"),
        "precision": totals[4] / totals[3] if totals[3] else float("nan"),
        "jumps": int(totals[3]),
    }


def interval(diff: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(diff), size=(SAMPLES, len(diff)))
    return tuple(float(x) for x in np.percentile(diff[picks].mean(1), [2.5, 97.5]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--sequence-file", type=Path, required=True, help="all dev-train names")
    parser.add_argument("--run", action="append", required=True, help="label=root before _s<seed>")
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--base2", default="outputs/E043_base2")
    parser.add_argument("--base1", default="outputs/E043_base1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    names = readSequenceFile(args.sequence_file)
    seeds = [int(s) for s in args.seeds.split(",")]
    scores: dict[str, list[dict[str, float]]] = {}
    for label, root in (("base1", args.base1), ("base2", args.base2)):
        scores[label] = [sDual(dataset, Path(f"{root}_s{s}"), names) for s in seeds]
    report: dict[str, object] = {"seeds": seeds, "configurations": {}}
    lines = [f"S_dual over {len(names)} sequences, seeds {seeds}"]
    base2Levels = [levels(args, Path(f"{args.base2}_s{s}"), None) for s in seeds]
    report["base2Levels"] = base2Levels
    for item in args.run:
        label, _, root = item.partition("=")
        perSeed, ran, levelRows = [], set(), []
        for index, seed in enumerate(seeds):
            own = sDual(dataset, Path(f"{root}_s{seed}"), names)
            ran |= set(own)
            # Sequences the run left out keep the base-2 result of the same seed.
            perSeed.append({**scores["base2"][index], **own})
            levelRows.append(levels(args, Path(f"{root}_s{seed}"), None))
        scores[label] = perSeed
        report["configurations"][label] = {
            "root": root,
            "sequencesRun": len(ran),
            "levels": levelRows,
        }
    table = {
        label: np.asarray([[run[n] for n in names] for run in runs]).mean(0)
        for label, runs in scores.items()
    }
    for label in table:
        if label in ("base1", "base2"):
            lines.append(f"  {label:<10} {table[label].mean():.4f}")
            continue
        entry = report["configurations"][label]
        for reference in ("base2", "base1"):
            diff = table[label] - table[reference]
            low, high = interval(diff)
            entry[f"vs{reference}"] = {"difference": float(diff.mean()), "low": low, "high": high}
        lv = entry["levels"]
        mean = {k: float(np.nanmean([row[k] for row in lv])) for k in lv[0]}
        entry["meanLevels"] = mean
        b2, b1 = entry["vsbase2"], entry["vsbase1"]
        lines.append(
            f"  {label:<10} {table[label].mean():.4f}  vs base2 {b2['difference']:+.4f}"
            f" [{b2['low']:+.4f}, {b2['high']:+.4f}]  vs base1 {b1['difference']:+.4f}"
            f" [{b1['low']:+.4f}, {b1['high']:+.4f}]  ran {entry['sequencesRun']}"
        )
        lines.append(
            "             detection {detectionRate:.1%}  gate1 {gate1:.1%}  gate2 {gate2:.1%}"
            "  gate3 {gate3:.1%}  precision {precision:.1%}  jumps {jumps:.0f}".format(**mean)
        )
    b2mean = {
        k: float(np.nanmean([row[k] for row in base2Levels])) for k in base2Levels[0]
    }
    lines.append(
        "  base2 levels: detection {detectionRate:.1%}  gate1 {gate1:.1%}  gate2 {gate2:.1%}"
        "  gate3 {gate3:.1%}  precision {precision:.1%}  jumps {jumps:.0f}".format(**b2mean)
    )
    report["perSequence"] = {label: dict(zip(names, map(float, v))) for label, v in table.items()}
    text = "\n".join(lines)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
