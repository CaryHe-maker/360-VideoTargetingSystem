"""Where should the tracking view stay while the track is in doubt? (E059, offline proxy)

    python tools/freeze_study.py --dataset-root <dir> --label-root <dir> \\
        --runs outputs/E053d_cam_s1 ... --camera outputs/E053/camera512 --out outputs/E059

On every untrusted frame of recorded baseline-3 runs (the state after the frame before
was not TRACKING) with the target in the picture: would a normal tracking view (half
width two target sizes, the size when last trusted) hold the target, centred on

    follow    the tracker's box of the frame before (what the motion model does now:
              its samples keep following the tracker while in doubt);
    frozen    the place last trusted;
    turned    the place last trusted turned with the camera's rotation since then.

Reported for all such frames, for those where the tracker's box was off the target
(IoU < 0.1), and for those where the target had come back near the last trusted place
(within the view of ``turned``) while the tracker was off it: there a frozen or
turned view would have let the tracker see it again.
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
from size_study import angle, unit  # noqa: E402

from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.loss_rate import dualIou  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402


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
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    groups = {"all": [], "tracker off target": [], "target back near": []}
    durations = {"all": [], "tracker off target": [], "target back near": []}
    records: list[dict] = []
    for root in args.runs:
        results = loadTrackerResults(root / "bbox" / "ours")
        for name in readSequenceFile(args.sequence_file):
            path = root / "trace" / "ours" / f"{name}.csv"
            if not path.exists():
                continue
            sequence = dataset.sequence(name)
            box, present = sequence.groundTruth("bbox")
            bfov, _ = sequence.groundTruth("bfov")
            width = sequence.frameSize[0]
            sequence.close()
            present = np.asarray(present, bool)
            iou = np.zeros(len(present))
            iou[present] = dualIou(box[present], results[name][present], width)
            truth = unit(bfov[:, 0], bfov[:, 1])
            orientation = np.load(args.camera / f"{name}.npz")["orientation"]
            with path.open(encoding="utf-8") as stream:
                rows = {int(r["frame"]): r for r in csv.DictReader(stream)}
            trusted = None
            previous = None
            for f in sorted(rows):
                row = rows[f]
                if (
                    previous is not None
                    and trusted is not None
                    and previous["modeAfter"] != "TRACKING"
                    and f < len(present)
                    and present[f]
                ):
                    half = 2.0 * float(trusted["sizeDeg"])
                    follow = unit(previous["yawDeg"], previous["pitchDeg"])
                    frozen = unit(trusted["yawDeg"], trusted["pitchDeg"])
                    turned = (orientation[f - 1] @ orientation[int(trusted["frame"])].T) @ frozen
                    inside = {
                        "follow": float(angle(truth[f], follow)) <= half,
                        "frozen": float(angle(truth[f], frozen)) <= half,
                        "turned": float(angle(truth[f], turned)) <= half,
                    }
                    off = iou[f - 1] < 0.1
                    since = f - int(trusted["frame"])
                    records.append({**inside, "since": since, "off": bool(off)})
                    groups["all"].append(inside)
                    durations["all"].append(since)
                    if off:
                        groups["tracker off target"].append(inside)
                        durations["tracker off target"].append(since)
                        if inside["turned"] and not inside["follow"]:
                            groups["target back near"].append(inside)
                            durations["target back near"].append(since)
                if (
                    row["modeAfter"] == "TRACKING"
                    and row["action"] != "jump"
                    or row["action"] == "jump"
                ):
                    trusted = row
                previous = row
    lines = ["untrusted frames, target in the picture: share with the target inside the view"]
    for label, part in groups.items():
        if not part:
            continue
        cells = "  ".join(
            f"{k} {np.mean([p[k] for p in part]):6.1%}" for k in ("follow", "frozen", "turned")
        )
        lines.append(
            f"  {label:<20} n={len(part):>6}  {cells}   frames since trusted median {np.median(durations[label]):.0f}"
        )
    lines.append("all untrusted frames by frames since the place was trusted")
    for low, high in ((1, 4), (4, 10), (10, 30), (30, 100), (100, 10**6)):
        part = [r for r in records if low <= r["since"] < high]
        if part:
            cells = "  ".join(
                f"{k} {np.mean([p[k] for p in part]):6.1%}" for k in ("follow", "frozen", "turned")
            )
            off = np.mean([p["off"] for p in part])
            lines.append(f"  {low:>3}-{high:<7} n={len(part):>6}  {cells}   tracker off {off:5.1%}")
    text = "\n".join(lines)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.txt").write_text(text + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                k: {
                    "n": len(v),
                    **{
                        m: float(np.mean([p[m] for p in v])) for m in ("follow", "frozen", "turned")
                    },
                }
                for k, v in groups.items()
                if v
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
