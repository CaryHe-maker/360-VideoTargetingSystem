"""Collect per-frame data for fusing the three confidence signals.

    python tools/fusion_dataset.py --dataset-root <dir> --label-root <dir> \\
        --sequence-file configs/splits/360vos_tune.txt --output outputs/E018/tune_episodes.csv

Every sampled frame is one independent episode.  The frames before it are given to the
tracker with their ground truth: the trajectory, the motion model and the views of the
warm-up frames all use the true target, so the tracker reaches the sampled frame in a
correct state.  The sampled frame itself is then tracked normally, with the view placed
by the motion prediction, and one row is written:

    iou          IoU of the returned box with the ground truth (0 when the target is absent)
    score        the backend's confidence
    simTemplate  DINOv2 similarity of the returned box to the frame-0 template
    simRecent    DINOv2 similarity to the mean of the true boxes of the warm-up frames
    offset       angle between the returned box and the predicted position, in units of
                 the predicted target size
    logScale     log of the ratio of the returned box size to the predicted size

Frames where the target is absent are sampled too: there any returned box is wrong.
Every other sampled frame is tracked a second time with the view moved off the target
by one to two and a half target sizes (``variant`` = displaced), which is what a
drifted tracker looks at; the motion prediction it is compared with stays the true one.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np

from track360.backends import ARTrackBackend, TrackerBackendImpl
from track360.backends.appearance import AppearanceModels, targetCrop
from track360.backends.artrack_seq_session import createArtrackSession
from track360.controller.motion_estimator import SphericalMotionEstimator
from track360.controller.view_planner import TRAJECTORY_LENGTH, ViewPlanner
from track360.core.config import loadConfig
from track360.core.errors import Track360Error
from track360.core.types import BFoV, TemplateCommand, TemplateCommandKind
from track360.datasets.tune_split import readSequenceFile
from track360.datasets.vot360 import Vot360Dataset, Vot360DataSource
from track360.evaluation.loss_rate import dualIou
from track360.geometry import SphericalGeometryImpl, makeSphericalPoint
from track360.geometry.projection_math import unitVectorToYawPitch, viewAxes
from track360.runtime.benchmark import _SharedSession

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
COLUMNS = (
    "sequence",
    "frame",
    "variant",
    "displacement",
    "present",
    "warmFrames",
    "iou",
    "score",
    "simTemplate",
    "simRecent",
    "offset",
    "logScale",
    "targetSizeDeg",
    "projection",
)
# A frame without a box returned: the worst value of every signal.
MISSING = {"score": 0.0, "simTemplate": -1.0, "simRecent": -1.0, "offset": 10.0, "logScale": 3.0}


def _bfov(row: np.ndarray) -> BFoV:
    return BFoV(
        makeSphericalPoint(math.radians(row[0]), math.radians(row[1])),
        min(math.radians(row[2]), 2.0 * math.pi - 1e-3),
        min(math.radians(row[3]), math.pi - 1e-3),
    )


def collect(args: argparse.Namespace) -> int:
    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    names = readSequenceFile(args.sequence_file)
    session = createArtrackSession(config)
    geometry = SphericalGeometryImpl(useRemap=config.geometry.resampler == "opencv")
    planner = ViewPlanner(config.geometry, config.tracking, config.backendTuning)
    models = AppearanceModels(Path(config.model.weights).parent / "hub", ("dinov2",))
    torch = models.torch
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for name in names:
            sequence = dataset.sequence(name)
            truthBox, present = sequence.groundTruth("bbox")
            truthBfov, _ = sequence.groundTruth("bfov")
            frameWidthPx = sequence.frameSize[0]
            initial = sequence.initialBfov()
            sequence.close()
            present = np.asarray(present, dtype=bool)
            source = Vot360DataSource(labelRoot=dataset.labelRoot)
            source.open(str(dataset.root), name)
            backend = TrackerBackendImpl(ARTrackBackend(_SharedSession(session)))
            first = source.read()
            templateSpec = planner.templateView(initial)
            templateView = geometry.cropViews(first, [_withoutPrior(templateSpec)])[0]
            backend.initialize(templateView, templateSpec.priorBox)
            templateFeature = models.embed(targetCrop(templateView.rgb, templateSpec.priorBox))[
                "dinov2"
            ]
            revision = 0
            rng = np.random.default_rng(int(name))
            window: deque = deque(maxlen=args.warm_frames + 1)
            window.append(first)
            index = 0
            while (frame := source.read()) is not None:
                index += 1
                window.append(frame)
                if (index + args.offset) % args.step:
                    continue
                # The frames just before, oldest first, that have a true target.
                warm = [item for item in list(window)[:-1] if present[int(item.frameIndex)]]
                if not warm:
                    continue
                backend.resetState()
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
                history: deque[BFoV] = deque(maxlen=TRAJECTORY_LENGTH)
                recent = []
                for position, item in enumerate(warm):
                    frameIndex = int(item.frameIndex)
                    target = _bfov(truthBfov[frameIndex])
                    if position == 0:
                        history.extend([target] * TRAJECTORY_LENGTH)
                        estimator.resetFromMeasurement(
                            target.center,
                            item.timestampNs,
                            frameIndex,
                            1.0,
                            target.horizontalFovRad,
                            target.verticalFovRad,
                        )
                    else:
                        estimator.recordMeasurement(
                            frameIndex=frameIndex,
                            timestampNs=item.timestampNs,
                            point=target.center,
                            confidence=1.0,
                            horizontalSizeRad=target.horizontalFovRad,
                            verticalSizeRad=target.verticalFovRad,
                        )
                    # The warm-up frames pass through the tracker with true inputs, so
                    # its appearance memory is that of a tracker that has not failed.
                    if frameIndex > 0:
                        spec = planner.searchView(
                            target.center,
                            target.horizontalFovRad,
                            target.verticalFovRad,
                            tuple(history),
                        )
                        view = geometry.cropViews(item, [spec])[0]
                        revision += 1
                        command = TemplateCommand(
                            TemplateCommandKind.KEEP, item.frameIndex, None, None, revision
                        )
                        try:
                            backend.infer((view,), command)
                        except Track360Error:
                            pass
                        recent.append(models.embed(targetCrop(view.rgb, spec.priorBox))["dinov2"])
                        history.append(target)
                prediction = estimator.predictDetailed(frame.timestampNs, 1)
                last = history[-1]
                horizontal = prediction.horizontalSizeRad or last.horizontalFovRad
                vertical = prediction.verticalSizeRad or last.verticalFovRad
                horizontal = min(horizontal, 2.0 * math.pi - 1e-3)
                vertical = min(vertical, math.pi - 1e-3)
                size = math.sqrt(horizontal * vertical)
                saved = backend.saveState()
                variants = [("normal", 0.0)]
                if present[index] and (index // args.step) % 2 == 0:
                    variants.append(("displaced", float(rng.uniform(1.0, 2.5))))
                for variant, displacement in variants:
                    center = prediction.center
                    if displacement:
                        # Put the view where a drifted tracker would look: off the target by a few
                        # target sizes in a random direction.
                        forward, right, up = viewAxes(prediction.center)
                        bearing = float(rng.uniform(0.0, 2.0 * math.pi))
                        angle = min(displacement * size, math.pi - 1e-3)
                        direction = math.cos(angle) * forward + math.sin(angle) * (
                            math.cos(bearing) * right + math.sin(bearing) * up
                        )
                        center = makeSphericalPoint(*unitVectorToYawPitch(tuple(direction)))
                    backend.restoreState(saved)
                    spec = planner.searchView(center, horizontal, vertical, tuple(history))
                    view = geometry.cropViews(frame, [spec])[0]
                    revision += 1
                    row = {
                        "sequence": name,
                        "frame": index,
                        "variant": variant,
                        "displacement": round(displacement, 3),
                        "present": int(present[index]),
                        "warmFrames": len(warm),
                        "iou": 0.0,
                        "targetSizeDeg": round(math.degrees(size), 3),
                        "projection": spec.projection.name.lower(),
                        **MISSING,
                    }
                    try:
                        observation = backend.infer(
                            (view,),
                            TemplateCommand(
                                TemplateCommandKind.KEEP, frame.frameIndex, None, None, revision
                            ),
                        )[0]
                        projected = geometry.projectLocalBoxBoundary(
                            observation.bbox, spec, frame.rgb.shape[1], frame.rgb.shape[0]
                        )
                    except Track360Error:
                        observation = None
                    if observation is not None:
                        box, prior = observation.bbox, spec.priorBox
                        feature = models.embed(targetCrop(view.rgb, box))["dinov2"]
                        reference = (
                            torch.nn.functional.normalize(torch.stack(recent).mean(dim=0), dim=1)
                            if recent
                            else templateFeature
                        )
                        centerCos = (
                            projected.bfov.center.x * prediction.center.x
                            + projected.bfov.center.y * prediction.center.y
                            + projected.bfov.center.z * prediction.center.z
                        )
                        row.update(
                            score=round(float(observation.modelScore), 6),
                            simTemplate=round(float((feature * templateFeature).sum().item()), 6),
                            simRecent=round(float((feature * reference).sum().item()), 6),
                            offset=round(math.acos(max(-1.0, min(1.0, centerCos))) / size, 6),
                            logScale=round(
                                0.5
                                * math.log(
                                    max(box.widthPx * box.heightPx, 1e-6)
                                    / (prior.widthPx * prior.heightPx)
                                ),
                                6,
                            ),
                        )
                        if present[index]:
                            result = np.asarray(
                                [
                                    [
                                        projected.bbox.xPx,
                                        projected.bbox.yPx,
                                        projected.bbox.widthPx,
                                        projected.bbox.heightPx,
                                    ]
                                ]
                            )
                            row["iou"] = round(
                                float(
                                    dualIou(truthBox[index : index + 1], result, frameWidthPx)[0]
                                ),
                                6,
                            )
                    writer.writerow(row)
                    written += 1
            source.close()
            backend.close()
            stream.flush()
            print(f"{name}: {written} episodes so far", flush=True)
    session.close()
    print(f"written {written} episodes to {args.output}")
    return 0


def _withoutPrior(spec):
    from dataclasses import replace

    return replace(spec, priorBox=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--step", type=int, default=4, help="sample every step-th frame")
    parser.add_argument("--offset", type=int, default=0, help="shift which frames are sampled")
    parser.add_argument(
        "--warm-frames", type=int, default=7, help="true frames given before each sampled frame"
    )
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    try:
        return collect(args)
    except (Track360Error, ValueError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
