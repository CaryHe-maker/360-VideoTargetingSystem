"""Run and score tracking methods on a 360VOT dataset.

    python tools/benchmark.py run  --dataset-root <dir> --output-root <dir> --method ours
    python tools/benchmark.py eval --dataset-root <dir> --output-root <dir>

``run`` writes result files in the layout the official toolkit evaluates and resumes
an interrupted run.  ``eval`` scores every method found under the output root.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from track360.core.config import loadConfig
from track360.core.errors import Track360Error
from track360.evaluation.vot360_metrics import Vot360Scores
from track360.runtime.benchmark import METHODS, evaluateResults, runBenchmark

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="track sequences and write result files")
    _addRoots(run)
    run.add_argument(
        "--method",
        required=True,
        choices=sorted(METHODS),
        help="; ".join(f"{name}: {text}" for name, text in METHODS.items()),
    )
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--sequences", default=None, help="comma-separated names; default: all")
    run.add_argument("--max-frames", type=int, default=None, help="smoke runs only")
    run.add_argument(
        "--shard",
        default="0/1",
        help="i/n: run every n-th sequence starting at i, to split a run across processes",
    )
    run.add_argument("--no-resume", action="store_true", help="rerun finished sequences")

    evaluate = commands.add_parser("eval", help="score the result files of every method")
    _addRoots(evaluate)
    evaluate.add_argument("--methods", default=None, help="comma-separated; default: all found")
    evaluate.add_argument("--json", type=Path, default=None, help="also write scores as JSON")
    evaluate.add_argument(
        "--allow-partial",
        action="store_true",
        help="score result files shorter than their sequence (smoke runs only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = buildParser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        if args.command == "run":
            summary = runBenchmark(
                datasetRoot=args.dataset_root,
                outputRoot=args.output_root,
                method=args.method,
                config=loadConfig(args.config),
                sequences=_split(args.sequences),
                maxFrames=args.max_frames,
                resume=not args.no_resume,
                shard=_parseShard(args.shard),
            )
            return 1 if summary.failures else 0
        scores = evaluateResults(
            datasetRoot=args.dataset_root,
            outputRoot=args.output_root,
            methods=_split(args.methods),
            allowPartial=args.allow_partial,
        )
        print(formatScores(scores))
        if args.json is not None:
            payload = {
                method: {kind: _scorePayload(item) for kind, item in byKind.items()}
                for method, byKind in scores.items()
            }
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
        ("bbox", "BBox", ("S_dual", "P_dual", "norm_P_dual", "P_angle")),
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


def _addRoots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root", required=True, help="directory of 360VOT sequence folders or zips"
    )
    parser.add_argument("--output-root", required=True, help="where bbox/, bfov/, reports/ live")


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
            }
            for name, item in scores.perSequence.items()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
