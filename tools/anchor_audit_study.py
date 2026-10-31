"""Would a look back at where the track was earlier reveal that it has gone wrong?

    python tools/anchor_audit_study.py --dataset-root <dir> --label-root <dir> \\
        --sequence-file configs/splits/360vos_dev_train.txt \\
        --run outputs/E015_train --out outputs/E044

Only forward passes on the frames of a finished run: nothing is tracked.  Every
``--step`` frames, for each lag L, a view of the normal size is laid around the place
the run's own box had L frames earlier and the tracker is asked for the target with a
stateless pass (frame-0 template only).  Lag 0 is the look in place, at the current box.

A stateless pass centred on the current box says little once the track sits on another
object (evaluation log, E016).  A pass centred on an earlier place is a different view:
if the target is still there, or has come back there, it finds it and scores high while
the track is elsewhere.  The table shows how often that happens on frames where the run
had lost the target, and how often such a pass raises a false alarm on frames where the
run was right.

One row per pass goes to ``<out>/audits.csv``.  A returned box is "on the target" when
its dual IoU with the ground truth is at least 0.5, and "the tracked box" when its
centre is within half a size of the run's current box and the sizes agree within 1.5.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from math import pi, sqrt
from pathlib import Path

import numpy as np

from track360.backends import createArtrackSession
from track360.controller.view_planner import ViewPlanner
from track360.core.config import loadConfig
from track360.core.types import FrameIndex, FramePacket, SequenceId
from track360.datasets.tune_split import readSequenceFile
from track360.datasets.vot360 import Vot360Dataset
from track360.evaluation.loss_rate import dualIou, lostFrameMask
from track360.evaluation.vot360_metrics import loadTrackerResults
from track360.geometry.projection_math import makeSphericalPoint
from track360.runtime.benchmark import _SharedSession
from track360.runtime.driver import buildRuntime, closeRuntime
from track360.runtime.gpu_lock import exclusiveGpuRun

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
GOOD_IOU, LOST_IOU = 0.5, 0.1
THRESHOLDS = (0.5, 0.6, 0.7, 0.8)


def _direction(yawDeg: float, pitchDeg: float) -> np.ndarray:
    yaw, pitch = np.radians(float(yawDeg)), np.radians(float(pitchDeg))
    return np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])


def _angleDeg(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(first @ second), -1.0, 1.0))))


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    names = readSequenceFile(args.sequence_file)
    boxes = loadTrackerResults(args.run / "bbox" / args.method)
    spheres = loadTrackerResults(args.run / "bfov" / args.method)
    lags = [int(item) for item in args.lags.split(",")]
    session = createArtrackSession(config)
    planner = ViewPlanner(config.geometry, config.tracking, config.backendTuning)
    rows: list[dict[str, object]] = []
    try:
        for number, name in enumerate(names, 1):
            if name not in boxes:
                continue
            sequence = dataset.sequence(name)
            truthBox, present = sequence.groundTruth("bbox")
            truthBfov, _ = sequence.groundTruth("bfov")
            present = np.asarray(present, dtype=bool)
            width = sequence.frameSize[0]
            iou = np.zeros(len(present))
            iou[present] = dualIou(truthBox[present], boxes[name][present], width)
            lost = lostFrameMask(iou, present)
            tracked = spheres[name]
            runtime = buildRuntime(config, artrackSessionFactory=lambda _: _SharedSession(session))
            try:
                first = FramePacket(SequenceId(name), FrameIndex(0), 0, sequence.readRgb(0))
                plan = runtime.controller.buildInitialization(
                    first, initialBfov=sequence.initialBfov()
                )
                template = runtime.geometry.cropViews(first, [plan.templateView])[0]
                runtime.backend.initialize(template, plan.templateBox)
                for index in range(args.start, sequence.frameCount, args.step):
                    frame = FramePacket(
                        SequenceId(name), FrameIndex(index), index, sequence.readRgb(index)
                    )
                    height = frame.rgb.shape[0]
                    here = _direction(tracked[index, 0], tracked[index, 1])
                    hereSize = sqrt(max(tracked[index, 2] * tracked[index, 3], 1e-6))
                    for lag in lags:
                        source = index - lag
                        if source < 0:
                            continue
                        wide = min(max(np.radians(tracked[source, 2]), 1e-4), 2 * pi - 1e-3)
                        tall = min(max(np.radians(tracked[source, 3]), 1e-4), pi - 1e-3)
                        centre = makeSphericalPoint(
                            np.radians(tracked[source, 0]), np.radians(tracked[source, 1])
                        )
                        spec = planner._searchView(centre, wide, tall)
                        found = runtime.backend.inferDetached(
                            (runtime.geometry.cropViews(frame, [spec])[0],)
                        )
                        row: dict[str, object] = {
                            "sequence": name,
                            "frame": index,
                            "lag": lag,
                            "present": int(present[index]),
                            "trackedIou": round(float(iou[index]), 4),
                            "lostRun": int(lost[index]),
                            "anchorMovedSizes": round(
                                _angleDeg(
                                    here, _direction(tracked[source, 0], tracked[source, 1])
                                )
                                / hereSize,
                                3,
                            ),
                            "found": 0,
                            "score": "",
                            "iou": "",
                            "onTrackedBox": "",
                        }
                        projected = None
                        if found:
                            try:
                                projected = runtime.geometry.projectLocalBoxBoundary(
                                    found[0].bbox, spec, width, height
                                )
                            except Exception:
                                projected = None
                        if projected is not None:
                            box = projected.bbox
                            place = np.array(
                                [
                                    projected.bfov.center.x,
                                    projected.bfov.center.y,
                                    projected.bfov.center.z,
                                ]
                            )
                            size = float(
                                np.degrees(
                                    sqrt(
                                        projected.bfov.horizontalFovRad
                                        * projected.bfov.verticalFovRad
                                    )
                                )
                            )
                            larger, smaller = max(size, hereSize), min(size, hereSize)
                            row.update(
                                found=1,
                                score=round(float(found[0].fusedScore), 4),
                                onTrackedBox=int(
                                    _angleDeg(here, place) < 0.5 * larger
                                    and larger < 1.5 * max(smaller, 1e-6)
                                ),
                            )
                            if present[index]:
                                row["iou"] = round(
                                    float(
                                        dualIou(
                                            truthBox[index : index + 1],
                                            np.asarray(
                                                [[box.xPx, box.yPx, box.widthPx, box.heightPx]]
                                            ),
                                            width,
                                        )[0]
                                    ),
                                    4,
                                )
                        rows.append(row)
                    releaseFrame = getattr(runtime.geometry, "releaseFrame", None)
                    if callable(releaseFrame):
                        releaseFrame()
            finally:
                closeRuntime(runtime)
                sequence.close()
            print(f"[{number}/{len(names)}] {name}: {len(rows)} passes so far", flush=True)
    finally:
        session.close()
    return rows


def report(rows: list[dict[str, object]]) -> dict[str, object]:
    """Per lag and score threshold: what a pass returns on lost, good and absent frames."""
    result: dict[str, object] = {}
    lags = sorted({int(row["lag"]) for row in rows})

    def table(part: list[dict[str, object]], label: str) -> None:
        groups = {
            "lost": [r for r in part if int(r["present"]) and int(r["lostRun"])],
            "good": [r for r in part if int(r["present"]) and float(r["trackedIou"]) >= GOOD_IOU],
            "absent": [r for r in part if not int(r["present"])],
        }
        print(
            f"\n{label}: passes on lost frames {len(groups['lost'])}, on good frames"
            f" {len(groups['good'])}, with the target absent {len(groups['absent'])}"
        )
        print(
            f"  {'score >=':<9}{'lost: on target':>17}{'lost: elsewhere':>17}"
            f"{'good: elsewhere':>17}{'good: on target':>17}{'absent: any':>13}"
        )
        entry = {}
        for threshold in THRESHOLDS:

            def share(group: str, kind: str, threshold: float = threshold) -> float:
                items = groups[group]
                if not items:
                    return float("nan")
                hits = 0
                for r in items:
                    if not int(r["found"]) or float(r["score"]) < threshold:
                        continue
                    onTarget = r["iou"] != "" and float(r["iou"]) >= GOOD_IOU
                    moved = not int(r["onTrackedBox"])
                    if kind == "onTarget":
                        hits += onTarget
                    elif kind == "elsewhere":
                        hits += moved and not onTarget
                    else:
                        hits += 1
                return hits / len(items)

            cells = (
                share("lost", "onTarget"),
                share("lost", "elsewhere"),
                share("good", "elsewhere"),
                share("good", "onTarget"),
                share("absent", "any"),
            )
            entry[f"{threshold:g}"] = dict(
                zip(
                    ("lostOnTarget", "lostElsewhere", "goodElsewhere", "goodOnTarget", "absent"),
                    cells,
                    strict=True,
                )
            )
            print(
                f"  {threshold:<9g}"
                + "".join(f"{value:>17.1%}" for value in cells[:4])
                + f"{cells[4]:>13.1%}"
            )
        result[label] = {"counts": {key: len(value) for key, value in groups.items()}, **entry}

    for lag in lags:
        table([row for row in rows if int(row["lag"]) == lag], f"lag {lag}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=None, help="output root of the finished run")
    parser.add_argument("--method", default="ours")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--start", type=int, default=10)
    parser.add_argument("--lags", default="0,10,30,60,100")
    parser.add_argument("--report-only", action="store_true", help="re-read <out>/audits.csv")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    path = args.out / "audits.csv"
    if args.report_only:
        with path.open(encoding="utf-8") as stream:
            rows: list[dict[str, object]] = list(csv.DictReader(stream))
    else:
        if args.run is None:
            print("--run is required unless --report-only", file=sys.stderr)
            return 2
        with exclusiveGpuRun():
            rows = run(args)
        args.out.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = report(rows)
    (args.out / "audits.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
