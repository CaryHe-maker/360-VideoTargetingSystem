"""Can the tracker still box the target when the search view is enlarged?

    python tools/view_scale_study.py --dataset-root <dir> --label-root <dir> \\
        --sequence-file configs/splits/360vos_dev_train.txt --out outputs/E028

Only forward passes: nothing is tracked.  For sampled frames with the target present, a
view is laid around the true target, shifted by a random share of the view's half-width
so the target is not in the middle, and the tracker is asked for the target with a
stateless pass (frame-0 template only).  The view is made 1, 4 and 8 times the normal
search view (which is four target sizes wide), and the whole sphere.  The same shift,
as a share of the view, is used at every scale.

The tracker also takes the previous boxes of the target as input.  Three choices of
what it is told, since a lost track does not know where the target is:

    view      a box the size of the enlarged search region's centre: what a view of
              this size gets today
    centre    a box of the target's real size in the middle of the view
    oracle    a box of the target's real size where the target really is: an upper
              bound for this resolution, not available in practice

A hit is a returned box with a dual IoU of at least 0.5 with the ground truth.  One row
per forward pass goes to ``<out>/forwards.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import replace
from math import cos, pi, sin, sqrt
from pathlib import Path

import numpy as np

from track360.backends import createArtrackSession
from track360.controller.view_planner import (
    ALIGNED_SEARCH_FACTOR,
    TRAJECTORY_LENGTH,
    ViewPlanner,
    localBoxOfBfov,
)
from track360.core.config import loadConfig
from track360.core.types import BFoV, FrameIndex, FramePacket, SequenceId
from track360.datasets.vot360 import Vot360Dataset
from track360.evaluation.loss_rate import dualIou
from track360.geometry.projection_math import makeSphericalPoint, unitVectorToYawPitch, viewAxes
from track360.runtime.benchmark import _SharedSession
from track360.runtime.driver import buildRuntime, closeRuntime

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
SCALES = (1.0, 4.0, 8.0, 0.0)  # 0: the whole sphere
PRIORS = ("view", "centre", "oracle")
MAX_SHIFT = 0.7
HIT_IOU = 0.5
COLUMNS = (
    "sequence",
    "frame",
    "scale",
    "prior",
    "targetSizeDeg",
    "viewWidthDeg",
    "viewHeightDeg",
    "projection",
    "targetPx",
    "shift",
    "found",
    "iou",
    "centreErrorSizes",
    "score",
)


def _label(scale: float) -> str:
    return "sphere" if scale == 0.0 else f"{scale:g}x"


def shifted(center, angle: float, bearing: float):
    forward, right, up = viewAxes(center)
    direction = (
        cos(angle) * np.asarray(forward)
        + sin(angle) * (cos(bearing) * np.asarray(right) + sin(bearing) * np.asarray(up))
    )
    return makeSphericalPoint(*unitVectorToYawPitch(tuple(direction)))


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    names = [
        line.strip()
        for line in args.sequence_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    session = createArtrackSession(config)
    planner = ViewPlanner(config.geometry, config.tracking, config.backendTuning)
    side = config.geometry.viewWidthPx
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    try:
        for number, name in enumerate(names, 1):
            sequence = dataset.sequence(name)
            truthBox, present = sequence.groundTruth("bbox")
            truthBfov, _ = sequence.groundTruth("bfov")
            present = np.asarray(present, dtype=bool)
            runtime = buildRuntime(config, artrackSessionFactory=lambda _: _SharedSession(session))
            try:
                first = FramePacket(SequenceId(name), FrameIndex(0), 0, sequence.readRgb(0))
                plan = runtime.controller.buildInitialization(
                    first, initialBfov=sequence.initialBfov()
                )
                template = runtime.geometry.cropViews(first, [plan.templateView])[0]
                runtime.backend.initialize(template, plan.templateBox)
                for index in range(args.start, sequence.frameCount, args.step):
                    if not present[index]:
                        continue
                    frame = FramePacket(
                        SequenceId(name), FrameIndex(index), index, sequence.readRgb(index)
                    )
                    height, width = frame.rgb.shape[:2]
                    target = BFoV(
                        makeSphericalPoint(
                            np.radians(truthBfov[index, 0]), np.radians(truthBfov[index, 1])
                        ),
                        min(max(np.radians(truthBfov[index, 2]), 1e-4), 2 * pi - 1e-3),
                        min(max(np.radians(truthBfov[index, 3]), 1e-4), pi - 1e-3),
                    )
                    size = sqrt(target.horizontalFovRad * target.verticalFovRad)
                    share, bearing = rng.uniform(0.0, MAX_SHIFT), rng.uniform(0.0, 2 * pi)
                    for scale in SCALES:
                        # A view for a target ``factor`` times as large is that much wider.
                        factor = scale or 2 * pi / (ALIGNED_SEARCH_FACTOR * size)
                        wide = factor * target.horizontalFovRad
                        tall = factor * target.verticalFovRad
                        if scale == 0.0:
                            base = planner._sphericalView(
                                target.center, wide, tall, ALIGNED_SEARCH_FACTOR, side
                            )
                        else:
                            base = planner._searchView(target.center, wide, tall)
                        half = 0.5 * min(base.bfov.horizontalFovRad, base.bfov.verticalFovRad)
                        center = shifted(target.center, share * half, bearing)
                        if scale == 0.0:
                            spec = planner._sphericalView(
                                center, wide, tall, ALIGNED_SEARCH_FACTOR, side
                            )
                        else:
                            spec = planner._searchView(center, wide, tall)
                        centred = BFoV(center, target.horizontalFovRad, target.verticalFovRad)
                        priors = {
                            "view": (),
                            "centre": (localBoxOfBfov(spec, centred),) * TRAJECTORY_LENGTH,
                            "oracle": (localBoxOfBfov(spec, target),) * TRAJECTORY_LENGTH,
                        }
                        view = runtime.geometry.cropViews(frame, [spec])[0]
                        targetPx = localBoxOfBfov(spec, target)
                        for prior, trajectory in priors.items():
                            asked = replace(view, spec=replace(spec, trajectory=trajectory))
                            found = runtime.backend.inferDetached((asked,))
                            row = {
                                "sequence": name,
                                "frame": index,
                                "scale": _label(scale),
                                "prior": prior,
                                "targetSizeDeg": round(float(np.degrees(size)), 3),
                                "viewWidthDeg": round(
                                    float(np.degrees(spec.bfov.horizontalFovRad)), 2
                                ),
                                "viewHeightDeg": round(
                                    float(np.degrees(spec.bfov.verticalFovRad)), 2
                                ),
                                "projection": spec.projection.name,
                                "targetPx": round(sqrt(targetPx.widthPx * targetPx.heightPx), 2),
                                "shift": round(float(share), 3),
                                "found": 0,
                                "iou": 0.0,
                                "centreErrorSizes": "",
                                "score": "",
                            }
                            if found:
                                try:
                                    projected = runtime.geometry.projectLocalBoxBoundary(
                                        found[0].bbox, spec, width, height
                                    )
                                except Exception:
                                    projected = None
                                if projected is not None:
                                    box = projected.bbox
                                    iou = float(
                                        dualIou(
                                            truthBox[index : index + 1],
                                            np.asarray(
                                                [[box.xPx, box.yPx, box.widthPx, box.heightPx]]
                                            ),
                                            width,
                                        )[0]
                                    )
                                    cosine = (
                                        projected.bfov.center.x * target.center.x
                                        + projected.bfov.center.y * target.center.y
                                        + projected.bfov.center.z * target.center.z
                                    )
                                    error = float(np.arccos(np.clip(cosine, -1.0, 1.0))) / size
                                    row.update(
                                        found=1,
                                        iou=round(iou, 4),
                                        centreErrorSizes=round(error, 3),
                                        score=round(float(found[0].fusedScore), 4),
                                    )
                            rows.append(row)
                    releaseFrame = getattr(runtime.geometry, "releaseFrame", None)
                    if callable(releaseFrame):
                        releaseFrame()
            finally:
                closeRuntime(runtime)
                sequence.close()
            print(f"[{number}/{len(names)}] {name}: {len(rows)} forwards so far", flush=True)
    finally:
        session.close()
    return rows


def report(rows: list[dict[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}

    def table(title: str, chosen: list[dict[str, object]]) -> None:
        print(f"\n{title}")
        print(
            f"  {'view':<8}{'told':<8}{'n':>6}{'hit':>8}{'IoU>=.1':>9}{'target px':>11}"
            f"{'score hit':>11}{'score miss':>12}"
        )
        payload[title] = {}
        for scale in map(_label, SCALES):
            for prior in PRIORS:
                part = [r for r in chosen if r["scale"] == scale and r["prior"] == prior]
                if not part:
                    continue
                iou = np.asarray([r["iou"] for r in part])
                hit = iou >= HIT_IOU
                scores = np.asarray(
                    [float(r["score"]) if r["score"] != "" else np.nan for r in part]
                )
                item = {
                    "n": len(part),
                    "hit": float(hit.mean()),
                    "loose": float((iou >= 0.1).mean()),
                    "targetPx": float(np.median([r["targetPx"] for r in part])),
                    "scoreHit": float(np.nanmedian(scores[hit])) if hit.any() else None,
                    "scoreMiss": float(np.nanmedian(scores[~hit])) if (~hit).any() else None,
                }
                payload[title][f"{scale} / {prior}"] = item
                cells = [
                    "" if item[key] is None else f"{item[key]:.2f}"
                    for key in ("scoreHit", "scoreMiss")
                ]
                print(
                    f"  {scale:<8}{prior:<8}{item['n']:>6}{item['hit']:>8.1%}{item['loose']:>9.1%}"
                    f"{item['targetPx']:>11.1f}{cells[0]:>11}{cells[1]:>12}"
                )

    table("all sampled frames", rows)
    for low, high in ((0.0, 3.0), (3.0, 10.0), (10.0, 400.0)):
        chosen = [r for r in rows if low <= r["targetSizeDeg"] < high]
        table(f"target size {low:g} to {high:g} degrees", chosen)
    for low, high in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.71)):
        chosen = [r for r in rows if low <= r["shift"] < high]
        table(f"target shifted {low:g} to {high:g} of the view's half-width", chosen)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, default=5)
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    rows = run(args)
    if not rows:
        print("no frames sampled", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "forwards.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    payload = report(rows)
    text = json.dumps(payload, indent=1) + "\n"
    (args.out / "view_scale.json").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
