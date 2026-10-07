"""Compare a run's per-frame confidence with its IoU against ground truth.

    python tools/score_analysis.py --dataset-root <dir> --output-root <dir> --method ours

Needs the ``score/`` files ``tools/benchmark.py run`` writes next to the results.
Answers whether the confidence can stand in for "the IoU is low" at run time, and
where a threshold on it would have to sit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from track360.core.errors import Track360Error
from track360.datasets.tune_split import readSequenceFile
from track360.datasets.vot360 import Vot360Dataset
from track360.evaluation.score_analysis import GOOD_IOU_THRESHOLD, buildFrameTable, summarize
from track360.evaluation.vot360_metrics import loadTrackerResults
from track360.io.vot360_results import BBOX_DIRECTORY, SCORE_DIRECTORY, readScoreFile

ALL_PERCENTILES = ("p1", "p5", "p25", "p50", "p75", "p95", "p99")
BIN_PERCENTILES = ("p5", "p25", "p50", "p75", "p95")


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--sequence-file", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None, help="also write the result as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = buildParser().parse_args(argv)
    try:
        summary = analyze(
            args.dataset_root,
            args.label_root,
            args.output_root,
            args.method,
            None if args.sequence_file is None else readSequenceFile(args.sequence_file),
        )
    except (Track360Error, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(formatSummary(summary))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


def analyze(
    datasetRoot: str,
    labelRoot: str | None,
    outputRoot: str,
    method: str,
    only: list[str] | None,
) -> dict[str, object]:
    dataset = Vot360Dataset(datasetRoot, labelRoot)
    output = Path(outputRoot)
    results = loadTrackerResults(output / BBOX_DIRECTORY / method)
    scoreRoot = output / SCORE_DIRECTORY / method
    scores = {
        path.stem: readScoreFile(path) for path in sorted(scoreRoot.glob("*.txt"))
    }
    if not scores:
        raise ValueError(f"no score files in {scoreRoot}; rerun the benchmark to write them")
    names = [name for name in results if name in scores and (only is None or name in only)]
    groundTruth = {}
    frameWidthPx = 0
    for name in names:
        sequence = dataset.sequence(name)
        groundTruth[name] = sequence.groundTruth("bbox")
        frameWidthPx = sequence.frameSize[0]
        sequence.close()
    table = buildFrameTable(
        groundTruth, {name: results[name] for name in names}, scores, frameWidthPx
    )
    return summarize(table)


def formatSummary(summary: dict) -> str:
    share, correlation, area = summary["share"], summary["correlation"], summary["auroc"]
    lines = [
        f"frames: {summary['frames']} in {summary['sequences']} sequences "
        "(target present, first frame excluded)",
        f"  good (IoU >= {GOOD_IOU_THRESHOLD}): {share['good']:.3f}   "
        f"weak: {share['weak']:.3f}   lost: {share['lost']:.3f}",
        "",
        f"score vs IoU:  Pearson {_number(correlation['pearson'])}   "
        f"Spearman {_number(correlation['spearman'])}",
        "AUROC (0.5 = chance, 1.0 = a threshold separates perfectly):",
        f"  lost vs rest {_number(area['lostVsRest'])}   lost vs good {_number(area['lostVsGood'])}"
        f"   good vs rest {_number(area['goodVsRest'])}",
        "",
        "score percentiles",
        f"  {'group':<8}" + "".join(f"{name:>8}" for name in ALL_PERCENTILES),
    ]
    for group in ("all", "good", "weak", "lost"):
        lines.append(f"  {group:<8}" + _percentileCells(summary["scorePercentiles"][group]))
    lines += [
        "",
        "score by IoU bin",
        f"  {'IoU':<12}{'frames':>8}" + "".join(f"{name:>8}" for name in BIN_PERCENTILES),
    ]
    for item in summary["scoreByIou"]:
        label = f"{item['iouFrom']:.1f}-{item['iouTo']:.1f}"
        score = item["score"]
        cells = (
            "".join(f"{score[key]:>8.3f}" for key in BIN_PERCENTILES)
            if score
            else f"{'-':>8}" * 5
        )
        lines.append(f"  {label:<12}{item['frames']:>8}{cells}")
    lines += [
        "",
        "flagging frames with score < threshold",
        f"  {'threshold':>9}{'lost caught':>13}{'good flagged':>14}"
        f"{'flagged lost':>14}{'flagged':>9}",
    ]
    for row in summary["thresholds"]:
        lines.append(_thresholdLine(row))
    best = summary["bestThreshold"]
    if best is not None:
        lines += ["  best separation of lost from good:", _thresholdLine(best, digits=4)]
    lines += ["", "median score around the first frame of a loss run (offset in frames)"]
    onset = summary["lossOnset"]
    offsets = sorted(int(key) for key in onset)
    lines.append("  " + "".join(f"{offset:>7d}" for offset in offsets))
    lines.append("  " + "".join(f"{onset[offset]['median']:>7.3f}" for offset in offsets))
    lines += [
        "",
        "per sequence",
        f"  {'seq':<6}{'frames':>7}{'lost':>7}{'median':>8}{'good':>8}{'lost':>8}"
        f"{'AUROC':>8}{'Spearman':>10}",
    ]
    for name, item in summary["perSequence"].items():
        lines.append(
            f"  {name:<6}{item['frames']:>7}{item['lostShare']:>7.2f}{item['medianScore']:>8.3f}"
            f"{_number(item['medianScoreGood']):>8}{_number(item['medianScoreLost']):>8}"
            f"{_number(item['aurocLost']):>8}{_number(item['spearman']):>10}"
        )
    return "\n".join(lines)


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _percentileCells(values: dict | None) -> str:
    if values is None:
        return f"{'-':>8}" * 7
    return "".join(f"{values[key]:>8.3f}" for key in ALL_PERCENTILES)


def _thresholdLine(row: dict, digits: int = 2) -> str:
    return (
        f"  {row['threshold']:>9.{digits}f}{row['lostFlagged']:>13.3f}{row['goodFlagged']:>14.3f}"
        f"{row['flaggedLost']:>14.3f}{row['flagged']:>9.3f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
