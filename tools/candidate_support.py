"""How often is a search candidate on the target, by its score and by how many views agreed?

    python tools/candidate_support.py --dataset-root <dir> --label-root <dir> \\
        --output-root outputs/E033_c_train --json outputs/E033/c_support.json

Reads the state trace of a run with ``scanMode: zoom``.  Every box a search returned is
one row: the kind of scan (``1x``: the look in place; ``2x`` / ``4x``: one enlarged view
and a view of the normal size; ``4x2`` / ``4x4``: two or four half-overlapping enlarged
views), its score without memory, and its support (2 or more: a second view pointed at
the same place).  A box is on the target when its centre is within half a target size
of the true centre.  Boxes that are the tracked box found again are counted apart, as
jumping to them moves nothing.

For every jump the mean IoU of the following frames is given as well, split the same
way, so a threshold can be read off the table instead of being guessed.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from track360.datasets.vot360 import Vot360Dataset
from track360.evaluation.loss_rate import dualIou
from track360.evaluation.vot360_metrics import loadTrackerResults

BANDS = ((0.0, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01))
AFTER_FRAMES = 30


def _direction(yawDeg: float, pitchDeg: float) -> np.ndarray:
    yaw, pitch = np.radians(float(yawDeg)), np.radians(float(pitchDeg))
    return np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(first @ second), -1.0, 1.0))))


def collect(args: argparse.Namespace) -> list[dict[str, object]]:
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    results = loadTrackerResults(args.output_root / "bbox" / args.method)
    rowsOut: list[dict[str, object]] = []
    for path in sorted((args.output_root / "trace" / args.method).glob("*.csv")):
        name = path.stem
        sequence = dataset.sequence(name)
        truthBox, present = sequence.groundTruth("bbox")
        truthBfov, _ = sequence.groundTruth("bfov")
        width = sequence.frameSize[0]
        sequence.close()
        present = np.asarray(present, dtype=bool)
        iou = np.zeros(len(truthBox))
        iou[present] = dualIou(truthBox[present], results[name][present], width)
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        previous = None
        for row in rows:
            index = int(row["frame"])
            if row["scan"] and previous is not None:
                for candidate in json.loads(row["candidates"] or "[]"):
                    place = _direction(candidate["yawDeg"], candidate["pitchDeg"])
                    size = float(candidate["sizeDeg"])
                    tracked = _direction(previous["yawDeg"], previous["pitchDeg"])
                    trackedSize = max(float(previous["sizeDeg"]), 1e-3)
                    onTarget = False
                    if present[index]:
                        truth = _direction(truthBfov[index, 0], truthBfov[index, 1])
                        targetSize = float(
                            np.sqrt(max(truthBfov[index, 2] * truthBfov[index, 3], 1e-6))
                        )
                        onTarget = _angle(truth, place) < 0.5 * max(targetSize, size)
                    after = iou[index : index + AFTER_FRAMES][present[index : index + AFTER_FRAMES]]
                    rowsOut.append(
                        {
                            "sequence": name,
                            "frame": index,
                            "scan": row["scan"],
                            "score": float(candidate["score"]),
                            "support": int(candidate.get("support", 1)),
                            "onTarget": bool(onTarget),
                            "onTracked": bool(
                                _angle(tracked, place) < 0.5 * max(trackedSize, size)
                                and max(trackedSize, size) < 1.5 * min(trackedSize, size)
                            ),
                            "jumped": candidate["verdict"] == "accepted",
                            "iouAfter": float(after.mean()) if after.size else None,
                        }
                    )
            previous = row
    return rowsOut


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", default="ours")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    rows = collect(args)
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        kind = "1x" if row["scan"] == "1x" else "enlarged"
        support = "2+" if int(row["support"]) >= 2 else "1"
        groups[(kind, support)].append(row)
        if kind != "1x":
            groups[(str(row["scan"]), support)].append(row)
    payload: dict[str, object] = {}
    for (kind, support), part in sorted(groups.items()):
        print(f"\nscan {kind}, support {support}: {len(part)} boxes")
        print(
            f"  {'score':<11}{'boxes':>7}{'tracked box':>13}{'elsewhere':>11}"
            f"{'on target':>11}{'precision':>11}{'jumps':>7}{'IoU after':>11}"
        )
        table = {}
        for low, high in BANDS:
            band = [r for r in part if low <= float(r["score"]) < high]
            moved = [r for r in band if not r["onTracked"]]
            right = [r for r in moved if r["onTarget"]]
            jumps = [r for r in band if r["jumped"] and r["iouAfter"] is not None]
            precision = len(right) / len(moved) if moved else float("nan")
            iouAfter = float(np.mean([r["iouAfter"] for r in jumps])) if jumps else float("nan")
            table[f"{low:g}-{min(high, 1.0):g}"] = {
                "boxes": len(band),
                "onTrackedBox": len(band) - len(moved),
                "elsewhere": len(moved),
                "onTarget": len(right),
                "precision": precision,
                "jumps": len(jumps),
                "iouAfterJump": iouAfter,
            }
            print(
                f"  {f'{low:g}-{min(high, 1.0):g}':<11}{len(band):>7}{len(band) - len(moved):>13}"
                f"{len(moved):>11}{len(right):>11}{precision:>11.1%}{len(jumps):>7}{iouAfter:>11.3f}"
            )
        payload[f"{kind}|{support}"] = table
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
