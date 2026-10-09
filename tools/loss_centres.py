"""Where is a lost target, measured from the places a search could start at?

    python tools/loss_centres.py --dataset-root <dir> --label-root <dir> \\
        --run outputs/E015_tune --run outputs/E015_holdout \\
        --table outputs/E018/tune_freerun.csv --table outputs/E018/holdout_freerun.csv \\
        --sequence-file configs/splits/360vos_dev_train.txt --json outputs/E032/centres.json

Works on runs without loss handling and their per-frame tables; the tracker is not run.
For every lost frame with the target present (IoU < 0.1 for at least five frames) the
distance from the true target to several candidate centres is measured, in units of the
half-width of the normal search view (four target sizes wide, so one unit is two target
sizes, and "within 1" means inside a view of the normal size around that centre):

    trusted (state machine)   the tracker's box at the last frame the state machine,
                              replayed over the recorded scores, called TRACKING: what
                              the system has
    trusted (ground truth)    the tracker's box at its last frame with IoU >= 0.5: what
                              a perfect judge would have
    motion, running           the motion model's prediction for the frame when it is
                              fed the tracker's own boxes all along, right or wrong:
                              where the search view is today
    motion, from trusted      the target's motion over the last frames before the
                              trusted one, continued at constant speed up to the frame
    motion, from trusted,     the same, but continued for at most ``--horizon`` frames
      capped                  and held there
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from state_replay import loadTable, replay  # noqa: E402

from track360.controller.motion_estimator import SphericalMotionEstimator  # noqa: E402
from track360.core.config import loadConfig  # noqa: E402
from track360.datasets.vot360 import FRAME_INTERVAL_NS, Vot360Dataset  # noqa: E402
from track360.evaluation.loss_rate import dualIou, lostFrameMask  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402
from track360.geometry import makeSphericalPoint  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
REACHES = (1.0, 2.0, 4.0, 7.0, 8.0)
AGES = ((0, 5), (5, 20), (20, 100), (100, 10**9))
VIEW_FACTOR, MIN_VIEW_DEG = 4.0, 2.0
VELOCITY_FRAMES = 5
CENTRES = (
    "trusted (state machine)",
    "trusted (ground truth)",
    "motion, running",
    "motion, from trusted",
    "motion, from trusted, capped",
)


def _directions(rows: np.ndarray) -> np.ndarray:
    longitude, latitude = np.radians(rows[:, 0]), np.radians(rows[:, 1])
    return np.column_stack(
        [
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
            np.cos(latitude) * np.cos(longitude),
        ]
    )


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(first @ second), -1.0, 1.0))))


def _continued(older: np.ndarray, newer: np.ndarray, steps: float) -> np.ndarray:
    """``newer`` moved on along the great circle from ``older``, ``steps`` times as far."""
    axis = np.cross(older, newer)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return newer
    axis /= norm
    angle = math.acos(max(-1.0, min(1.0, float(older @ newer)))) * steps
    return (
        newer * math.cos(angle)
        + np.cross(axis, newer) * math.sin(angle)
        + axis * float(axis @ newer) * (1.0 - math.cos(angle))
    )


def collect(args: argparse.Namespace) -> tuple[list[dict[str, float]], dict[str, int]]:
    config = loadConfig(args.config)
    tuning = replace(config.backendTuning, lossHandling=True, stateRule="relative")
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    wanted = None
    if args.sequence_file is not None:
        wanted = {
            line.strip()
            for line in args.sequence_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
    tables = {}
    for path in args.table:
        tables.update(loadTable(path, tuning))
    rows: list[dict[str, float]] = []
    totals = {"frames": 0, "present": 0, "lost": 0, "sequences": 0}
    for run in args.run:
        boxes = loadTrackerResults(run / "bbox" / args.method)
        spheres = loadTrackerResults(run / "bfov" / args.method)
        for name in sorted(boxes):
            if (wanted is not None and name not in wanted) or name not in tables:
                continue
            sequence = dataset.sequence(name)
            truthBox, present = sequence.groundTruth("bbox")
            truthBfov, _ = sequence.groundTruth("bfov")
            width = sequence.frameSize[0]
            sequence.close()
            present = np.asarray(present, dtype=bool)
            iou = np.zeros(len(truthBox))
            iou[present] = dualIou(truthBox[present], boxes[name][present], width)
            lost = lostFrameMask(iou, present)
            totals["frames"] += len(lost)
            totals["present"] += int(present.sum())
            totals["lost"] += int(lost.sum())
            totals["sequences"] += 1
            truth, tracked = _directions(truthBfov), _directions(spheres[name])
            sizes = np.sqrt(np.clip(spheres[name][:, 2] * spheres[name][:, 3], 1e-6, None))
            # States of the recorded scores; the table starts at frame 1.
            states = np.zeros(len(lost), dtype=int)
            states[tables[name]["frame"]] = replay(tables[name], tuning)
            estimator = SphericalMotionEstimator(
                windowLength=config.tracking.windowLength,
                maxPredictionHorizon=config.tracking.maxPredictionHorizon,
                minSamplesForVelocity=config.motion.minSamplesForVelocity,
                maxTangentSpanRad=config.motion.maxTangentSpanRad,
                huberDeltaRad=config.motion.huberDeltaRad,
                processNoiseRadPerSec=config.motion.processNoiseRadPerSec,
                maxAngularSpeedRadPerSec=config.motion.maxAngularSpeedRadPerSec,
                maxLogScaleRatePerSec=config.motion.maxLogScaleRatePerSec,
            )

            def point(index: int, spheres=spheres, name=name):
                row = spheres[name][index]
                return makeSphericalPoint(math.radians(row[0]), math.radians(row[1]))

            def extent(index: int, spheres=spheres, name=name) -> tuple[float, float]:
                row = spheres[name][index]
                return (
                    min(max(math.radians(row[2]), 1e-4), 2.0 * math.pi - 1e-3),
                    min(max(math.radians(row[3]), 1e-4), math.pi - 1e-3),
                )

            estimator.resetFromMeasurement(point(0), 0, 0, 1.0, *extent(0))
            machine = truthful = 0
            age = 0
            for index in range(1, len(lost)):
                predicted = estimator.predictDetailed(index * FRAME_INTERVAL_NS, 1).center
                running = np.array([predicted.x, predicted.y, predicted.z])
                if lost[index] and present[index]:
                    centres = {
                        "trusted (state machine)": (tracked[machine], machine),
                        "trusted (ground truth)": (tracked[truthful], truthful),
                        "motion, running": (running, machine),
                    }
                    older = tracked[max(0, machine - VELOCITY_FRAMES)]
                    span = max(1, machine - max(0, machine - VELOCITY_FRAMES))
                    gap = index - machine
                    centres["motion, from trusted"] = (
                        _continued(older, tracked[machine], gap / span),
                        machine,
                    )
                    centres["motion, from trusted, capped"] = (
                        _continued(older, tracked[machine], min(gap, args.horizon) / span),
                        machine,
                    )
                    row = {"age": float(age)}
                    for label, (centre, anchor) in centres.items():
                        half = 0.5 * max(MIN_VIEW_DEG, VIEW_FACTOR * float(sizes[anchor]))
                        row[label] = _angle(truth[index], centre) / half
                    rows.append(row)
                    age += 1
                elif present[index]:
                    age = 0
                estimator.recordMeasurement(
                    frameIndex=index,
                    timestampNs=index * FRAME_INTERVAL_NS,
                    point=point(index),
                    confidence=1.0,
                    horizontalSizeRad=extent(index)[0],
                    verticalSizeRad=extent(index)[1],
                )
                if states[index] == 0:
                    machine = index
                if present[index] and iou[index] >= 0.5:
                    truthful = index
    return rows, totals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--table", type=Path, action="append", required=True)
    parser.add_argument("--sequence-file", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--method", default="ours")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    rows, totals = collect(args)
    payload: dict[str, object] = {"totals": totals}
    print(
        f"{totals['sequences']} sequences, {totals['frames']} frames, {totals['present']} with "
        f"the target present; lost frames {totals['lost']}: "
        f"{totals['lost'] / totals['frames']:.1%} of all frames, "
        f"{totals['lost'] / totals['present']:.1%} of the frames with the target"
    )
    for label in CENTRES:
        print(f"\nshare of lost frames with the target within reach of: {label}")
        header = "".join(f"{f'<= {reach:g}':>9}" for reach in REACHES)
        print(f"  {'frames lost':<14}{'n':>7}{header}")
        payload[label] = {}
        for low, high in (*AGES, (0, 10**9)):
            chosen = [row[label] for row in rows if low <= row["age"] < high]
            everything = low == 0 and high >= 10**9
            name = "all" if everything else f"{low}-{high if high < 10**9 else ''}"
            shares = [float(np.mean(np.asarray(chosen) <= reach)) for reach in REACHES]
            payload[label][name] = {
                "frames": len(chosen),
                "within": dict(zip(map(str, REACHES), shares, strict=True)),
            }
            cells = "".join(f"{share:>9.1%}" for share in shares)
            print(f"  {name:<14}{len(chosen):>7}{cells}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
