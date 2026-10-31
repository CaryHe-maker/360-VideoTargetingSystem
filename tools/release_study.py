"""What tells a right release from a wrong one? (E071, offline, dev train)

    python tools/release_study.py --dataset-root <dir> --label-root <dir> \\
        --runs outputs/E066_s1 ... --out outputs/E071

A release ends a doubt without a jump (the state machine trusts the track again after
``releaseFrames`` calm frames).  For every release of the recorded runs: whether the
track is on the target at that frame (IoU >= 0.5; wrong: target absent or IoU < 0.1),
and what the system knew then: how long the doubt lasted, the relative signal on the
release frame and its lowest value over the calm frames, how far the released box is
from the last box trusted before the doubt (in that box's sizes, turned with the
camera when ``--camera`` is given), and the best scan candidate seen during the doubt.
Reported: the AUROC of each for wrong against right, and rules that refuse a release.
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
from size_study import angle, sequenceData, unit  # noqa: E402

from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402

FEATURES = (
    "duration",
    "distance",
    "bestCandidate",
    "relative",
    "calmMin",
    "backend",
    "appearance",
)


def auroc(pos: list[float], neg: list[float]) -> float:
    values = np.concatenate([pos, neg])
    ranks = values.argsort().argsort() + 1.0
    return float(
        (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    )


def events(args) -> list[dict]:
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    out = []
    for root in args.runs:
        results = loadTrackerResults(root / "bbox" / "ours")
        for name in readSequenceFile(args.sequence_file):
            d = sequenceData(dataset, root, name, results)
            if d is None:
                continue
            orientation = None
            if args.camera is not None and (args.camera / f"{name}.npz").exists():
                orientation = np.load(args.camera / f"{name}.npz")["orientation"]
            with (root / "trace" / "ours" / f"{name}.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            start = None
            trusted = None  # (frame, direction, size) of the last trusted box
            relatives: list[float] = []
            best = 0.0
            for r in rows:
                f = int(r["frame"])
                before, after = r["modeBefore"], r["modeAfter"]
                if before == "TRACKING" and after != "TRACKING" and r["action"] != "jump":
                    start, relatives, best = f, [], 0.0
                if start is not None and r["candidates"] not in ("", "[]"):
                    for c in json.loads(r["candidates"]):
                        best = max(best, float(c["score"]))
                if start is not None and r["relative"] not in ("", "None"):
                    relatives.append(float(r["relative"]))
                if start is not None and r["action"] == "jump":
                    start = None
                if start is not None and r["reason"] == "RELEASED":
                    here = unit(float(r["yawDeg"]), float(r["pitchDeg"]))
                    distance = np.nan
                    if trusted is not None:
                        place = trusted[1]
                        if orientation is not None and f < len(orientation):
                            place = (orientation[f] @ orientation[trusted[0]].T) @ place
                        distance = float(angle(here, place)) / max(trusted[2], 1e-3)
                    present = f < len(d["present"]) and bool(d["present"][f])
                    iou = float(d["iou"][f]) if present else 0.0
                    out.append(
                        {
                            "run": root.name,
                            "sequence": name,
                            "frame": f,
                            "right": present and iou >= 0.5,
                            "wrong": (not present) or iou < 0.1,
                            "duration": f - start,
                            "relative": relatives[-1] if relatives else 0.0,
                            "calmMin": min(relatives[-3:]) if relatives else 0.0,
                            "distance": distance,
                            "bestCandidate": best,
                            "backend": float(r["backend"] or 0.0),
                            "appearance": float(r["appearance"] or 0.0),
                        }
                    )
                    start = None
                if after == "TRACKING" and start is None:
                    trusted = (
                        f,
                        unit(float(r["yawDeg"]), float(r["pitchDeg"])),
                        float(r["sizeDeg"]),
                    )
    return out


def report(rows: list[dict]) -> str:
    right = [r for r in rows if r["right"]]
    wrong = [r for r in rows if r["wrong"]]
    lines = [
        f"{len(rows)} releases: right {len(right)} ({len(right) / max(1, len(rows)):.1%}),"
        f" wrong {len(wrong)}, between {len(rows) - len(right) - len(wrong)}",
        "  AUROC (wrong vs right; > 0.5 means larger values are wrong):",
    ]
    for key in FEATURES:
        pos = [r[key] for r in wrong if np.isfinite(r[key])]
        neg = [r[key] for r in right if np.isfinite(r[key])]
        if pos and neg:
            lines.append(
                f"    {key:<14} {auroc(pos, neg):.3f}   median wrong {np.median(pos):.3f}"
                f" right {np.median(neg):.3f}"
            )
    lines.append("  refuse the release when ...: refused right / refused wrong")
    rules = [
        *(("duration", ">", v) for v in (10, 20, 40)),
        *(("distance", ">", v) for v in (1.0, 2.0, 4.0)),
        *(("relative", "<", v) for v in (-0.2, -0.1, 0.0)),
    ]
    for key, op, v in rules:
        hit = [
            r for r in rows if np.isfinite(r[key]) and (r[key] > v if op == ">" else r[key] < v)
        ]
        lines.append(
            f"    {key} {op} {v:<5}  {sum(r['right'] for r in hit):>4} /"
            f" {sum(r['wrong'] for r in hit):>4}   (right {len(right)}, wrong {len(wrong)})"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument(
        "--sequence-file", type=Path, default=Path("configs/splits/360vos_dev_train.txt")
    )
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--camera", type=Path, default=Path("outputs/E053/camera512"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    rows = events(args)
    text = report(rows)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.txt").write_text(text + "\n", encoding="utf-8")
    (args.out / "rows.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
