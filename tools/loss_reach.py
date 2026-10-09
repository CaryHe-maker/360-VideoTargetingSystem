"""How far from the search view is a lost target?

    python tools/loss_reach.py --dataset-root <dir> --label-root <dir> \\
        --run outputs/E015_tune --run outputs/E015_holdout --json outputs/E024/reach.json

For every lost frame with the target present (IoU < 0.1 for at least five frames), the
distance from the true target to two places is measured, in half-widths of the normal
search view (the view is four target sizes wide, so 1 means: at the edge of the view):

    current   where the tracker's own box is: the centre of the search view
    trusted   where the tracker was before the loss began (its last frame with
              IoU >= 0.5 before the lost run)

A view ``s`` times as wide still contains the target's centre when the distance is at
most ``s``.  The table gives the share of lost frames within each reach, by how long
the track has been lost, and how many views of the normal size it takes to tile the same
area (``(2s - 1)^2`` with views half a view apart, as the scan does).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from track360.core.errors import Track360Error
from track360.datasets.vot360 import Vot360Dataset
from track360.evaluation.loss_rate import dualIou, lostFrameMask
from track360.evaluation.vot360_metrics import loadTrackerResults

REACHES = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
AGES = ((0, 5), (5, 20), (20, 100), (100, 10**9))
VIEW_FACTOR, MIN_VIEW_DEG = 4.0, 2.0


def _directions(rows: np.ndarray) -> np.ndarray:
    longitude, latitude = np.radians(rows[:, 0]), np.radians(rows[:, 1])
    return np.column_stack(
        [
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
            np.cos(latitude) * np.cos(longitude),
        ]
    )


def collect(args: argparse.Namespace) -> list[tuple[int, float, float, float]]:
    """(frames since the loss began, distance to current, to trusted, target size) per frame."""
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    rows = []
    for run in args.run:
        boxes = loadTrackerResults(run / "bbox" / args.method)
        spheres = loadTrackerResults(run / "bfov" / args.method)
        for name in sorted(boxes):
            sequence = dataset.sequence(name)
            truthBox, present = sequence.groundTruth("bbox")
            truthBfov, _ = sequence.groundTruth("bfov")
            frameWidthPx = sequence.frameSize[0]
            sequence.close()
            present = np.asarray(present, dtype=bool)
            iou = np.zeros(len(truthBox))
            iou[present] = dualIou(truthBox[present], boxes[name][present], frameWidthPx)
            lost = lostFrameMask(iou, present)
            truth, tracked = _directions(truthBfov), _directions(spheres[name])
            trusted, trustedSize, age = None, 0.0, 0
            for index in range(len(lost)):
                if present[index] and iou[index] >= 0.5:
                    trusted = tracked[index]
                    trustedSize = float(np.sqrt(spheres[name][index, 2] * spheres[name][index, 3]))
                    age = 0
                if not (lost[index] and present[index]) or trusted is None:
                    if not lost[index] and present[index]:
                        age = 0
                    continue
                # Half the width of the view the tracker would use around the trusted box.
                half = 0.5 * max(MIN_VIEW_DEG, VIEW_FACTOR * trustedSize)
                toCurrent = np.degrees(np.arccos(np.clip(truth[index] @ tracked[index], -1, 1)))
                toTrusted = np.degrees(np.arccos(np.clip(truth[index] @ trusted, -1, 1)))
                rows.append((age, toCurrent / half, toTrusted / half, trustedSize))
                age += 1
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--method", default="ours")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    try:
        rows = np.asarray(collect(args))
    except (Track360Error, ValueError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    payload: dict[str, object] = {"frames": len(rows)}
    print(f"{len(rows)} lost frames with the target present and a trusted place before them")
    for column, label in ((1, "current"), (2, "trusted")):
        print(f"\nshare of lost frames with the target within reach of the {label} place")
        header = "".join(f"{f'<= {reach:g}':>9}" for reach in REACHES)
        print(f"  {'frames lost':<14}{'n':>7}{header}")
        payload[label] = {}
        for low, high in (*AGES, (0, 10**9)):
            chosen = rows[(rows[:, 0] >= low) & (rows[:, 0] < high)]
            everything = low == 0 and high >= 10**9
            name = "all" if everything else f"{low}-{high if high < 10**9 else ''}"
            shares = [
                float((chosen[:, column] <= reach).mean()) if len(chosen) else 0.0
                for reach in REACHES
            ]
            payload[label][name] = {
                "frames": len(chosen),
                "within": dict(zip(map(str, REACHES), shares, strict=True)),
            }
            cells = "".join(f"{share:>9.1%}" for share in shares)
            print(f"  {name:<14}{len(chosen):>7}{cells}")
    print("\nviews of the normal size needed to tile each reach (half a view apart):")
    print("  " + ", ".join(f"{reach:g}: {round((2 * reach - 1) ** 2)}" for reach in REACHES))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
