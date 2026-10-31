"""Check that the tracker judges candidates as the offline data does (evaluation log E051).

    python tools/benchmark.py run ... --candidate-judge J --judge-threshold T  (no noise)
    python tools/backend_signals.py trace --run <run> --out <dir> ...
    python tools/candidate_judge_check.py --run <run> --traced <dir> --judge J

The tracker writes the judge score of each candidate into its state trace.  The trace
tool reruns, outside the tracker, the stateless pass on each candidate of that trace
and stores the pass's inputs; this tool scores those with the same judge and compares
the two scores candidate by candidate.  The trace tool works from the rounded places
of the state trace, so the two passes see views that differ by up to 0.0005 degrees.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from track360.runtime.candidate_judge import CandidateJudge  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--traced", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--method", default="ours")
    args = parser.parse_args(argv)
    online: dict[tuple[str, int, int], float] = {}
    for path in sorted((args.run / "trace" / args.method).glob("*.csv")):
        with path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                for order, candidate in enumerate(json.loads(row["candidates"] or "[]")):
                    if candidate.get("judge") is not None:
                        online[(path.stem, int(row["frame"]), order)] = float(candidate["judge"])
    d = np.load(args.traced / "trace_candidates.npz")
    scan = d["scan"]
    x = np.column_stack(
        [
            d["originalScore"],
            d["score"],
            d["similarity"],
            (scan == "1x").astype(float),
            (scan == "4x").astype(float),
            d["coordEntropy"].mean(1),
            d["coordEntropy"].max(1),
            d["coordPeak"].min(1),
            d["hardSoftGap"].max(1),
            d["scoreFeature"].astype(np.float32),
        ]
    )
    judge = CandidateJudge(args.judge, threshold=0.0)
    pairs = []
    for i in range(len(x)):
        key = (str(d["sequence"][i]), int(d["frame"][i]), int(d["order"][i]))
        if key not in online:
            continue
        judge.select(key[0])
        pairs.append((online[key], float(judge.score(x[i])[0])))
    if not pairs:
        raise SystemExit("no candidate is in both the run and the traced passes")
    a = np.asarray(pairs)
    gap = np.abs(a[:, 0] - a[:, 1])
    print(
        f"{len(a)} candidates in both ({len(online)} judged in the run, {len(x)} traced);"
        f" score difference: median {np.median(gap):.2e}, 99th percentile"
        f" {np.percentile(gap, 99):.2e}, largest {gap.max():.2e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
