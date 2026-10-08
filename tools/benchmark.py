"""Run and score tracking methods on a 360VOT-style dataset.

    python tools/benchmark.py run  --dataset-root <dir> --output-root <dir> --method ours
    python tools/benchmark.py eval --dataset-root <dir> --output-root <dir>
    python tools/benchmark.py compare --dataset-root <dir> \\
        --baseline <output-root>:<method> --candidate <output-root>:<method>

``run`` writes result files in the layout the official toolkit evaluates and resumes
an interrupted run.  ``eval`` scores every method found under the output root.
``compare`` gives the difference between two runs with bootstrap confidence intervals
and, with ``--hard-file``, fails when a hard-regression sequence got worse.

The tune set lives in the 360VOS training archives, which carry no labels of their
own; pass ``--label-root`` (see ``tools/prepare_tune_set.py``) and
``--sequence-file configs/splits/360vos_tune.txt``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from track360.core.config import loadConfig
from track360.core.errors import Track360Error
from track360.datasets.tune_split import readSequenceFile
from track360.datasets.vots_info import loadVotsInfo
from track360.evaluation.comparison import (
    CHANGE_THRESHOLD,
    HARD_REGRESSION_TOLERANCE,
    Comparison,
    GroupChange,
    SequenceChange,
    compareScores,
    groupChange,
    hardRegressions,
)
from track360.evaluation.vot360_metrics import Vot360Scores
from track360.runtime.benchmark import (
    METHODS,
    Efficiency,
    evaluateResults,
    loadEfficiency,
    runBenchmark,
)
from track360.runtime.run_archive import TIMING_STATES, archiveRun

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
DEFAULT_ARCHIVE = Path(__file__).resolve().parents[1] / "docs" / "runs"
ATTRIBUTE_SETS = ("360vot", "360vos-train")


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="track sequences and write result files")
    _addCommon(run)
    run.add_argument(
        "--method",
        required=True,
        choices=sorted(METHODS),
        help="; ".join(f"{name}: {text}" for name, text in METHODS.items()),
    )
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--max-frames", type=int, default=None, help="smoke runs only")
    run.add_argument(
        "--shard",
        default="0/1",
        help="i/n: run every n-th sequence starting at i, to split a run across processes",
    )
    run.add_argument("--no-resume", action="store_true", help="rerun finished sequences")

    evaluate = commands.add_parser("eval", help="score the result files of every method")
    _addCommon(evaluate)
    evaluate.add_argument("--methods", default=None, help="comma-separated; default: all found")
    evaluate.add_argument("--json", type=Path, default=None, help="also write scores as JSON")
    evaluate.add_argument(
        "--info",
        type=Path,
        default=None,
        help="360vots-info.csv; adds scores per challenge attribute",
    )
    evaluate.add_argument(
        "--attribute-set",
        choices=ATTRIBUTE_SETS,
        default="360vot",
        help="which sequence IDs the dataset uses: 360VOT test or 360VOS training",
    )
    evaluate.add_argument(
        "--allow-partial",
        action="store_true",
        help="score result files shorter than their sequence (smoke runs only)",
    )

    archive = commands.add_parser(
        "archive", help="write the versioned record of one run (scores, cost, configuration)"
    )
    _addCommon(archive)
    archive.add_argument("--method", required=True)
    archive.add_argument("--name", required=True, help="record file name, without extension")
    archive.add_argument("--experiment", default="", help="evaluation-log entry, e.g. E011")
    archive.add_argument("--split", default="", help="tune / hold-out / test")
    archive.add_argument("--note", default="")
    archive.add_argument(
        "--timing",
        choices=TIMING_STATES,
        default="unknown",
        help="solo: nothing else was running, so the latencies are comparable",
    )
    archive.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)

    compare = commands.add_parser(
        "compare", help="difference between two runs with bootstrap confidence intervals"
    )
    _addCommon(compare, outputRoot=False)
    compare.add_argument("--baseline", required=True, help="<output-root>:<method>")
    compare.add_argument("--candidate", required=True, help="<output-root>:<method>")
    compare.add_argument(
        "--hard-file",
        type=Path,
        default=None,
        help="hard-regression sequences; exit 1 when one drops by more than "
        f"{HARD_REGRESSION_TOLERANCE} S_dual",
    )
    compare.add_argument(
        "--fragile-file",
        type=Path,
        default=None,
        help="unstable sequences judged as a group; exit 1 when the interval of their "
        "mean S_dual difference lies below 0",
    )
    compare.add_argument("--samples", type=int, default=10_000, help="bootstrap resamples")
    compare.add_argument("--json", type=Path, default=None, help="also write the result as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = buildParser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        sequences = _sequences(args)
        if args.command == "run":
            summary = runBenchmark(
                datasetRoot=args.dataset_root,
                outputRoot=args.output_root,
                labelRoot=args.label_root,
                method=args.method,
                config=loadConfig(args.config),
                sequences=sequences,
                maxFrames=args.max_frames,
                resume=not args.no_resume,
                shard=_parseShard(args.shard),
            )
            return 1 if summary.failures else 0
        if args.command == "archive":
            path = archiveRun(
                datasetRoot=args.dataset_root,
                outputRoot=args.output_root,
                labelRoot=args.label_root,
                method=args.method,
                name=args.name,
                archiveRoot=args.archive_root,
                sequences=sequences,
                split=args.split,
                experiment=args.experiment,
                note=args.note,
                timing=args.timing,
            )
            print(f"record written: {path}")
            return 0
        if args.command == "compare":
            return _compare(args, sequences)

        def score(only: list[str] | None) -> dict[str, dict[str, Vot360Scores]]:
            return evaluateResults(
                datasetRoot=args.dataset_root,
                outputRoot=args.output_root,
                labelRoot=args.label_root,
                methods=_split(args.methods),
                allowPartial=args.allow_partial,
                only=only,
            )

        scores = score(sequences)
        print(formatScores(scores))
        efficiency = {
            method: loadEfficiency(args.output_root, method, sequences) for method in scores
        }
        print()
        print(formatEfficiency(efficiency))
        payload: dict[str, object] = {
            "overall": _payload(scores),
            "efficiency": {
                method: _efficiencyPayload(item) for method, item in efficiency.items()
            },
        }
        if args.info is not None:
            byAttribute = {}
            for name, members in _attributeSequences(args.info, args.attribute_set).items():
                wanted = [m for m in members if sequences is None or m in sequences]
                if wanted:
                    try:
                        byAttribute[name] = score(wanted)
                    except Track360Error:
                        continue  # no method has results for this attribute yet
            print()
            print(formatAttributeScores(byAttribute))
            payload["byAttribute"] = {name: _payload(item) for name, item in byAttribute.items()}
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 0
    except (Track360Error, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


def formatScores(scores: dict[str, dict[str, Vot360Scores]]) -> str:
    """Render the two tables the official toolkit prints, best success first."""
    lines: list[str] = []
    layouts = (
        ("bbox", "BBox", ("S_dual", "P_dual", "norm_P_dual", "P_angle", "loss_rate")),
        ("bfov", "BFoV", ("S_sphere", "P_angle")),
    )
    for kind, title, columns in layouts:
        rows = [(method, byKind[kind]) for method, byKind in scores.items() if kind in byKind]
        if not rows:
            continue
        rows.sort(key=lambda row: row[1].success, reverse=True)
        lines.append(f"360VOT metrics in {title}")
        lines.append(
            f"  {'method':<16}" + "".join(f"{name:>13}" for name in columns) + f"{'sequences':>11}"
        )
        for method, item in rows:
            summary = item.summary()
            values = "".join(f"{summary[name]:>13.3f}" for name in columns)
            lines.append(f"  {method:<16}{values}{item.sequenceCount:>11}")
        lines.append("")
    return "\n".join(lines).rstrip()


def formatEfficiency(efficiency: dict[str, Efficiency | None]) -> str:
    """Cost of each run.  Timings are a reference: the machine load is not controlled."""
    lines = ["efficiency (timings are a reference only: machine load is not controlled)"]
    lines.append(
        f"  {'run':<16}{'forwards/frame':>16}{'P50 ms':>10}{'P95 ms':>10}"
        f"{'FPS':>8}{'sequences':>11}"
    )
    for name, item in efficiency.items():
        if item is None:
            lines.append(f"  {name:<16}{'no run reports':>16}")
            continue
        forwards = (
            f"{'-':>16}"
            if item.forwardsPerFrame is None
            else f"{item.forwardsPerFrame:>16.2f}"
        )
        lines.append(
            f"  {name:<16}{forwards}{item.p50LatencyMs:>10.0f}{item.p95LatencyMs:>10.0f}"
            f"{item.fps:>8.1f}{item.sequences:>11}"
        )
    return "\n".join(lines)


def _efficiencyPayload(item: Efficiency | None) -> dict[str, object] | None:
    if item is None:
        return None
    return {
        "sequences": item.sequences,
        "frames": item.frames,
        "fps": item.fps,
        "p50LatencyMs": item.p50LatencyMs,
        "p95LatencyMs": item.p95LatencyMs,
        "forwardsPerFrame": item.forwardsPerFrame,
    }


def formatAttributeScores(byAttribute: dict[str, dict[str, dict[str, Vot360Scores]]]) -> str:
    """One row per attribute: S_dual / P_angle of each method on the BBox results."""
    methods = sorted({method for scores in byAttribute.values() for method in scores})
    lines = ["S_dual / P_angle per attribute (BBox results)"]
    lines.append(f"  {'attribute':<10}{'seqs':>5}" + "".join(f"{name:>16}" for name in methods))
    for name, scores in byAttribute.items():
        cells = []
        count = 0
        for method in methods:
            item = scores.get(method, {}).get("bbox")
            if item is None:
                cells.append(f"{'-':>16}")
            else:
                count = max(count, item.sequenceCount)
                cells.append(f"{item.success:>9.3f} /{item.anglePrecision:>5.3f}")
        lines.append(f"  {name:<10}{count:>5}" + "".join(cells))
    return "\n".join(lines)


def _compare(args: argparse.Namespace, sequences: list[str] | None) -> int:
    hard = None if args.hard_file is None else readSequenceFile(args.hard_file)
    scores = []
    efficiency: dict[str, Efficiency | None] = {}
    for label, reference in (("baseline", args.baseline), ("candidate", args.candidate)):
        root, method = _parseRun(reference)
        efficiency[label] = loadEfficiency(root, method, sequences)
        scores.append(
            evaluateResults(
                datasetRoot=args.dataset_root,
                outputRoot=root,
                labelRoot=args.label_root,
                methods=[method],
                only=sequences,
            )[method]["bbox"]
        )
    comparison = compareScores(scores[0], scores[1], samples=args.samples)
    failed = () if hard is None else hardRegressions(comparison, hard)
    fragile = (
        None
        if args.fragile_file is None
        else groupChange(comparison, readSequenceFile(args.fragile_file), samples=args.samples)
    )
    print(formatComparison(comparison, args.baseline, args.candidate))
    print()
    print(formatEfficiency(efficiency))
    if hard is not None:
        print()
        print(formatHardRegressions(comparison, hard, failed))
    if fragile is not None:
        print()
        print(formatGroupChange(fragile))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    **_comparisonPayload(comparison, args, failed),
                    "fragileGroup": None
                    if fragile is None
                    else {
                        "sequences": list(fragile.sequences),
                        "baseline": fragile.baseline,
                        "candidate": fragile.candidate,
                        "difference": {
                            "value": fragile.difference.value,
                            "low": fragile.difference.low,
                            "high": fragile.difference.high,
                        },
                        "regressed": fragile.regressed,
                    },
                    "efficiency": {
                        label: _efficiencyPayload(item) for label, item in efficiency.items()
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 1 if failed or (fragile is not None and fragile.regressed) else 0


def formatComparison(comparison: Comparison, baseline: str, candidate: str) -> str:
    """Scores of both runs and their difference, each with a 95% bootstrap interval."""
    lines = [
        f"baseline:  {baseline}",
        f"candidate: {candidate}",
        f"sequences: {len(comparison.sequences)} (those both runs have results for)",
        "",
        f"  {'metric':<10}{'baseline':>24}{'candidate':>24}{'difference':>26}",
    ]
    for metric in comparison.metrics:
        verdict = "" if metric.difference.excludesZero else "  (interval includes 0)"
        lines.append(
            f"  {metric.name:<10}"
            f"{_interval(metric.baseline):>24}"
            f"{_interval(metric.candidate):>24}"
            f"{_interval(metric.difference, signed=True):>26}{verdict}"
        )
    up = sum(change.difference > CHANGE_THRESHOLD for change in comparison.changes)
    down = sum(change.difference < -CHANGE_THRESHOLD for change in comparison.changes)
    lines += ["", f"S_dual per sequence: {up} up, {down} down (by more than 0.02)"]
    lines += [_changeLine(change) for change in comparison.changes[:5]]
    if len(comparison.changes) > 10:
        lines.append("    ...")
    lines += [_changeLine(change) for change in comparison.changes[5:][-5:]]
    return "\n".join(lines)


def formatHardRegressions(
    comparison: Comparison, hard: list[str], failed: tuple[SequenceChange, ...]
) -> str:
    byName = {change.sequence: change for change in comparison.changes}
    failedNames = {change.sequence for change in failed}
    lines = [f"hard-regression sequences (fail: S_dual drops by over {HARD_REGRESSION_TOLERANCE})"]
    for name in hard:
        status = "FAIL" if name in failedNames else "ok"
        lines.append(f"{_changeLine(byName[name])}  {status}")
    lines.append("result: " + (f"FAILED ({len(failed)} of {len(hard)})" if failed else "passed"))
    return "\n".join(lines)


def formatGroupChange(group: GroupChange) -> str:
    verdict = (
        "REGRESSED (interval below 0)"
        if group.regressed
        else "improved (interval above 0)"
        if group.difference.low > 0.0
        else "no change beyond noise (interval includes 0)"
    )
    return "\n".join(
        [
            f"fragile sequences, judged as a group ({len(group.sequences)} sequences)",
            f"    mean S_dual  {group.baseline:.3f} -> {group.candidate:.3f}  "
            f"difference {_interval(group.difference, signed=True)}",
            f"result: {verdict}",
        ]
    )


def _interval(interval, *, signed: bool = False) -> str:
    sign = "+" if signed else ""
    return f"{interval.value:{sign}.3f} [{interval.low:{sign}.3f}, {interval.high:{sign}.3f}]"


def _changeLine(change: SequenceChange) -> str:
    return (
        f"    {change.sequence:<8}{change.baseline:.3f} -> {change.candidate:.3f}"
        f"  ({change.difference:+.3f})"
    )


def _comparisonPayload(
    comparison: Comparison, args: argparse.Namespace, failed: tuple[SequenceChange, ...]
) -> dict[str, object]:
    return {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "sequences": list(comparison.sequences),
        "bootstrapSamples": args.samples,
        "metrics": {
            metric.name: {
                part: {"value": item.value, "low": item.low, "high": item.high}
                for part, item in (
                    ("baseline", metric.baseline),
                    ("candidate", metric.candidate),
                    ("difference", metric.difference),
                )
            }
            for metric in comparison.metrics
        },
        "perSequence": {
            change.sequence: {"baseline": change.baseline, "candidate": change.candidate}
            for change in comparison.changes
        },
        "hardRegressionFailures": [change.sequence for change in failed],
    }


def _parseRun(reference: str) -> tuple[str, str]:
    root, separator, method = reference.rpartition(":")
    if not separator or not root or not method:
        raise ValueError(f"expected <output-root>:<method>, got '{reference}'")
    return root, method


def _addCommon(parser: argparse.ArgumentParser, *, outputRoot: bool = True) -> None:
    parser.add_argument(
        "--dataset-root", required=True, help="directory of sequence folders or zips"
    )
    if outputRoot:
        parser.add_argument(
            "--output-root", required=True, help="where bbox/, bfov/, reports/ live"
        )
    parser.add_argument(
        "--label-root", default=None, help="labels for sequences shipped without label.json"
    )
    parser.add_argument("--sequences", default=None, help="comma-separated names; default: all")
    parser.add_argument(
        "--sequence-file", type=Path, default=None, help="file with one sequence name per line"
    )


def _sequences(args: argparse.Namespace) -> list[str] | None:
    if args.sequences is not None and args.sequence_file is not None:
        raise ValueError("give --sequences or --sequence-file, not both")
    if args.sequence_file is not None:
        return readSequenceFile(args.sequence_file)
    return _split(args.sequences)


def _attributeSequences(infoPath: Path, attributeSet: str) -> dict[str, list[str]]:
    info = loadVotsInfo(infoPath)
    attributes = info.votAttributes() if attributeSet == "360vot" else info.vosAttributes("train")
    return {
        name: sorted(sequence for sequence, names in attributes.items() if name in names)
        for name in info.attributeNames
    }


def _split(text: str | None) -> list[str] | None:
    if text is None:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def _parseShard(text: str) -> tuple[int, int]:
    try:
        index, count = (int(part) for part in text.split("/"))
    except ValueError as error:
        raise ValueError("--shard must look like i/n, for example 0/2") from error
    return index, count


def _payload(scores: dict[str, dict[str, Vot360Scores]]) -> dict[str, object]:
    return {
        method: {kind: _scorePayload(item) for kind, item in byKind.items()}
        for method, byKind in scores.items()
    }


def _scorePayload(scores: Vot360Scores) -> dict[str, object]:
    return {
        **scores.summary(),
        "successCurve": scores.successCurve.tolist(),
        "anglePrecisionCurve": scores.anglePrecisionCurve.tolist(),
        "perSequence": {
            name: {
                "success": item.success,
                "precision": item.precision,
                "normPrecision": item.normPrecision,
                "anglePrecision": item.anglePrecision,
                "frames": item.frames,
                "lostFrames": item.lostFrames,
                "firstLostFrame": item.firstLostFrame,
            }
            for name, item in scores.perSequence.items()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
