"""Export the per-frame fusion table of a normal (free-running) probed run.

    python tools/fusion_freerun.py --dataset-root <dir> --label-root <dir> \\
        --output-root outputs/E016f_tune --output outputs/E018/tune_freerun.csv

The run must have been made with ``tools/benchmark.py run --probe dinov2``.  One row per
frame after the first, with the same columns as ``tools/fusion_dataset.py`` so both
tables can be analysed together.  Here nothing is given to the tracker: the trajectory,
the appearance memory and the motion model all come from its own earlier outputs, as in
deployment.  The motion residuals are recomputed by replaying the motion model over the
result file.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path

import numpy as np

from track360.controller.motion_estimator import SphericalMotionEstimator
from track360.core.config import loadConfig
from track360.core.errors import Track360Error
from track360.core.types import BFoV
from track360.datasets.vot360 import FRAME_INTERVAL_NS, Vot360Dataset
from track360.evaluation.loss_rate import dualIou, lostFrameMask
from track360.evaluation.vot360_metrics import loadTrackerResults
from track360.geometry import makeSphericalPoint
from track360.io.vot360_results import readScoreFile

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
COLUMNS = (
    "sequence",
    "frame",
    "variant",
    "present",
    "iou",
    "lostRun",
    "score",
    "simTemplate",
    "simRecent",
    "offset",
    "logScale",
    "targetSizeDeg",
)
# The verifier's running reference: update rate and the gates a frame must pass.
MEMORY_RATE = 0.05
TRUST_SIMILARITY = 0.40
TRUST_SCORE = 0.50


def _bfov(row: np.ndarray) -> BFoV:
    return BFoV(
        makeSphericalPoint(math.radians(row[0]), math.radians(row[1])),
        min(max(math.radians(row[2]), 1e-4), 2.0 * math.pi - 1e-3),
        min(max(math.radians(row[3]), 1e-4), math.pi - 1e-3),
    )


def export(args: argparse.Namespace) -> int:
    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    root = args.output_root
    boxes = loadTrackerResults(root / "bbox" / args.method)
    spheres = loadTrackerResults(root / "bfov" / args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for name in sorted(boxes):
            sequence = dataset.sequence(name)
            truth, present = sequence.groundTruth("bbox")
            frameWidthPx = sequence.frameSize[0]
            sequence.close()
            present = np.asarray(present, dtype=bool)
            iou = np.zeros(len(truth))
            iou[present] = dualIou(truth[present], boxes[name][present], frameWidthPx)
            lost = lostFrameMask(iou, present)
            score = readScoreFile(root / "score" / args.method / f"{name}.txt")
            features = np.load(root / "probe" / args.method / "features" / f"{name}.npy").astype(
                np.float32
            )
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
            first = _bfov(spheres[name][0])
            estimator.resetFromMeasurement(
                first.center, 0, 0, 1.0, first.horizontalFovRad, first.verticalFovRad
            )
            previous = first
            reference = features[0].copy()
            for index in range(1, len(truth)):
                timestampNs = index * FRAME_INTERVAL_NS
                prediction = estimator.predictDetailed(timestampNs, 1)
                horizontal = prediction.horizontalSizeRad or previous.horizontalFovRad
                vertical = prediction.verticalSizeRad or previous.verticalFovRad
                size = math.sqrt(horizontal * vertical)
                result = _bfov(spheres[name][index])
                centerCos = (
                    result.center.x * prediction.center.x
                    + result.center.y * prediction.center.y
                    + result.center.z * prediction.center.z
                )
                feature = features[index]
                toTemplate = float(feature @ features[0])
                toRecent = float(feature @ reference) / (float(np.linalg.norm(reference)) + 1e-9)
                value = max(toTemplate, toRecent)
                writer.writerow(
                    {
                        "sequence": name,
                        "frame": index,
                        "variant": "freerun",
                        "present": int(present[index]),
                        "iou": round(float(iou[index]), 6),
                        "lostRun": int(lost[index]),
                        "score": round(float(score[index]), 6),
                        "simTemplate": round(toTemplate, 6),
                        "simRecent": round(toRecent, 6),
                        "offset": round(math.acos(max(-1.0, min(1.0, centerCos))) / size, 6),
                        "logScale": round(
                            0.5
                            * math.log(
                                (result.horizontalFovRad * result.verticalFovRad)
                                / (horizontal * vertical)
                            ),
                            6,
                        ),
                        "targetSizeDeg": round(math.degrees(size), 3),
                    }
                )
                written += 1
                if feature.any() and value >= TRUST_SIMILARITY and score[index] >= TRUST_SCORE:
                    reference = (1.0 - MEMORY_RATE) * reference + MEMORY_RATE * feature
                estimator.recordMeasurement(
                    frameIndex=index,
                    timestampNs=timestampNs,
                    point=result.center,
                    confidence=float(min(1.0, max(0.0, score[index]))),
                    horizontalSizeRad=result.horizontalFovRad,
                    verticalSizeRad=result.verticalFovRad,
                )
                previous = result
    print(f"written {written} frames to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="ours")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    try:
        return export(args)
    except (Track360Error, ValueError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
