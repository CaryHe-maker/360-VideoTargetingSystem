"""Event-level comparison of candidate-judge runs with base 2 (evaluation log E051).

    python tools/judge_report.py --dataset-root <dir> --label-root <dir> \\
        --base outputs/E043_base2 --candidate outputs/E051_T1 --seeds 1,2,3,4,5

For every enlarged-view search made with the target in the picture and inside the
view, whether any returned candidate was on the target (gate 2, any candidate) and,
of those, whether the search ended in a jump that held (gate 3: mean IoU of the next
10 frames >= 0.1).  The gates of ``loss_report.py`` judge only the candidate with the
highest search score; a judge may take another one, so here any candidate counts.
Also the precision of the taken candidates: the share of jumps whose candidate was on
the target (all searches, the look in place included).  Each side pools its seeds;
the 95% interval of the difference comes from resampling sequences.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_outcomes  # noqa: E402

from track360.datasets.vot360 import Vot360Dataset  # noqa: E402

SAMPLES = 10_000


def _direction(yawDeg: float, pitchDeg: float) -> np.ndarray:
    yaw, pitch = np.radians(float(yawDeg)), np.radians(float(pitchDeg))
    return np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])


def _onTarget(candidate: dict, truth: np.ndarray, size: float) -> bool:
    place = _direction(candidate["yawDeg"], candidate["pitchDeg"])
    angle = float(np.degrees(np.arccos(np.clip(float(truth @ place), -1.0, 1.0))))
    return angle < 0.5 * max(size, float(candidate["sizeDeg"]))


def counts(root: Path, args: argparse.Namespace, dataset: Vot360Dataset) -> dict[str, np.ndarray]:
    """Per sequence: in view, any on target, held among those, jumps, jumps on target."""
    ns = argparse.Namespace(**{**vars(args), "output_root": root})
    attempts = {(a["sequence"], a["frame"]): a for a in search_outcomes.analyse(ns)}
    table: dict[str, list[int]] = {}
    import csv

    for path in sorted((root / "trace" / args.method).glob("*.csv")):
        name = path.stem
        sequence = dataset.sequence(name)
        truthBfov, present = sequence.groundTruth("bfov")
        sequence.close()
        present = np.asarray(present, dtype=bool)
        row_ = table.setdefault(name, [0, 0, 0, 0, 0, 0, 0])
        with path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                found = json.loads(row["candidates"] or "[]")
                if not found:
                    continue
                index = int(row["frame"])
                attempt = attempts.get((name, index))
                taken = [c for c in found if c["verdict"] in ("accepted", "same_place")]
                onTarget = []
                if present[index]:
                    truth = _direction(truthBfov[index, 0], truthBfov[index, 1])
                    size = float(np.sqrt(max(truthBfov[index, 2] * truthBfov[index, 3], 1e-6)))
                    onTarget = [_onTarget(c, truth, size) for c in found]
                if row["action"] == "jump" and taken:
                    row_[3] += 1
                    chosen = found.index(taken[0])
                    row_[4] += int(bool(onTarget) and onTarget[chosen])
                if attempt is None or row["scan"] == "1x" or not present[index]:
                    continue
                if float(attempt.get("reach", 9.0)) > 1.0:
                    continue
                row_[0] += 1
                if any(onTarget):
                    row_[1] += 1
                    held = attempt["jumped"] and (attempt.get("iouAfter") or 0.0) >= 0.1
                    row_[2] += int(bool(held))
                # The gate of loss_report: the candidate with the best search score.
                if attempt.get("bestOnTarget"):
                    row_[5] += 1
                    row_[6] += int(
                        bool(attempt["jumped"] and (attempt.get("iouAfter") or 0.0) >= 0.1)
                    )
    return {name: np.asarray(values) for name, values in table.items()}


def pooled(table: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    names = sorted(set().union(*table))
    return {name: sum(t.get(name, np.zeros(7, int)) for t in table) for name in names}


def ratio(totals: np.ndarray, top: int, bottom: int) -> float:
    return float(totals[top] / totals[bottom]) if totals[bottom] else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--base", required=True, help="output root before _s<seed>")
    parser.add_argument("--candidate", required=True, help="output root before _s<seed>")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--method", default="ours")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    seeds = [int(s) for s in args.seeds.split(",")]
    sides = {}
    for label, root in (("base", args.base), ("candidate", args.candidate)):
        sides[label] = pooled([counts(Path(f"{root}_s{s}"), args, dataset) for s in seeds])
    names = sorted(set(sides["base"]) & set(sides["candidate"]))
    a = np.stack([sides["base"][n] for n in names])
    b = np.stack([sides["candidate"][n] for n in names])
    metrics = {
        "gate2 (any candidate on target)": (1, 0),
        "gate3 (held jump | any on target)": (2, 1),
        "precision of taken candidates": (4, 3),
        "gate3, loss_report (best search score)": (6, 5),
    }
    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(names), size=(SAMPLES, len(names)))
    result: dict[str, object] = {"sequences": len(names), "seeds": seeds, "metrics": {}}
    print(f"{len(names)} sequences, seeds {seeds}; totals over seeds")
    for label, (top, bottom) in metrics.items():
        ta, tb = a.sum(0), b.sum(0)
        va, vb = ratio(ta, top, bottom), ratio(tb, top, bottom)
        sa, sb = a[picks].sum(1), b[picks].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            diff = sb[:, top] / sb[:, bottom] - sa[:, top] / sa[:, bottom]
        low, high = np.nanpercentile(diff, [2.5, 97.5])
        result["metrics"][label] = {
            "base": va,
            "candidate": vb,
            "baseCounts": [int(ta[top]), int(ta[bottom])],
            "candidateCounts": [int(tb[top]), int(tb[bottom])],
            "difference": vb - va,
            "low": float(low),
            "high": float(high),
        }
        print(
            f"  {label:<40} base {va:6.1%} ({ta[top]}/{ta[bottom]})"
            f"  candidate {vb:6.1%} ({tb[top]}/{tb[bottom]})"
            f"  difference {vb - va:+.3f} [{low:+.3f}, {high:+.3f}]"
        )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
