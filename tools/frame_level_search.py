"""Does the search for a lost target work better without the trajectory input?

    python tools/frame_level_search.py --dataset-root <dir> --label-root <dir> \\
        --run outputs/E033_a_train --config configs/_e033_a.yaml --out outputs/E034

Only forward passes: nothing is tracked.  The search attempts of a finished run with
``scanMode: zoom`` are taken from its state trace, and at each of them the target is
looked for again, around the place the track was last trusted, in two ways:

    1x        one view of the normal size
    4x>1x     a view four times as large, then a view of the normal size around the
              box found there

each with two forward passes that use the same weights:

    S (sequence-level)   what the system runs: the pass is told the target's last
                         seven boxes; a search does not know them, so it is told a box
                         in the middle of the view seven times
    F (frame-level)      the model built without the trajectory input and the
                         appearance decoder (the parameters for them are left out of
                         the checkpoint); it sees the template and the view only

and, for the enlarged view, a mixed one (``F:4x>S:1x``): located by F, boxed by S.

One row per returned box goes to ``<out>/boxes.csv`` with the score of the pass, the
similarity to the frame-0 template, a motion score (agreement with the place and size
the target would have if it had kept moving as before the trusted frame, at most ten
frames on; the same fall-offs as the state score's) and the true IoU.  The report
gives the three gates of a search (target in the view / box on the target / box
accepted) and how the scores separate boxes on the target from the others.

The frame-level model code must be present in the vendor tree
(``lib/models/artrackv2``, ``lib/config/artrackv2``, ``artrackv2_256_full.yaml``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

from track360.backends import createArtrackSession
from track360.backends.artrack_model import (
    _SEARCH_FACTOR,
    _SEARCH_SIZE,
    ARTrackPrediction,
    _clipBox,
    _sampleTarget,
)
from track360.backends.observation import buildRgbObservation
from track360.controller.view_planner import ViewPlanner
from track360.core.config import loadConfig
from track360.core.errors import GeometryError, ModelError
from track360.core.types import BBoxXYWH, FrameIndex, FramePacket, SequenceId
from track360.datasets.vot360 import Vot360Dataset
from track360.evaluation.loss_rate import dualIou
from track360.evaluation.vot360_metrics import loadTrackerResults
from track360.geometry.projection_math import makeSphericalPoint
from track360.runtime.benchmark import _SharedSession
from track360.runtime.driver import buildRuntime, closeRuntime

KINDS = ("1x", "2x", "3x", "4x")
BINS = 400
VELOCITY_FRAMES = 5
HORIZON = 10
OFFSET_SCALE, SIZE_SCALE = 0.5, 0.1
ACCEPT = 0.70
COLUMNS = (
    "sequence",
    "frame",
    "scan",
    "lostFrames",
    "present",
    "targetSizeDeg",
    "trustedSizeDeg",
    "reach1",
    "reach4",
    "method",
    "found",
    "score",
    "coarseScore",
    "similarity",
    "motion",
    "motionOffset",
    "motionLogScale",
    "iou",
    "centreErrorSizes",
    "onTarget",
    "onTracked",
    "trackedScore",
    "trackedSimilarity",
)
METHODS = ("S:1x", "F:1x", "S:4x>1x", "F:4x>1x", "F:4x>S:1x", "S:4x", "F:4x")


class FrameLevel:
    """ARTrackV2 without the trajectory input, on the weights of the session."""

    def __init__(self, session) -> None:
        self._session = session
        torch = session._torch
        from lib.config.artrackv2.config import cfg, update_config_from_file
        from lib.models.artrackv2 import build_artrackv2

        update_config_from_file(str(session._root / "artrackv2_256_full.yaml"))
        model = build_artrackv2(cfg, training=False)
        checkpoint = torch.load(session._weights, map_location="cpu", weights_only=False)
        state = checkpoint.get("net", checkpoint.get("model", checkpoint))
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            raise ModelError(f"missing parameters: {incompatible.missing_keys[:8]}")
        self.unused = len(incompatible.unexpected_keys)
        self._model = model.to(session._device).eval()

    def infer(self, view, templateTensor):
        """The box of the target in ``view``; ``None`` when it leaves the view."""
        torch = self._session._torch
        prior = view.spec.priorBox
        crop, resizeFactor, _ = _sampleTarget(view.rgb, prior, _SEARCH_FACTOR, _SEARCH_SIZE)
        search = self._session._preprocess(crop).unsqueeze(0)
        template = torch.stack((templateTensor, templateTensor), dim=0)[:, None]
        with torch.inference_mode():
            output = self._model(template=template, search=search)
        x0, y0, x1, y1 = (
            float(value) for value in (output["seqs"][0, :4].float() / (BINS - 1) - 0.5).tolist()
        )
        scale = _SEARCH_SIZE / resizeFactor
        width, height = max(1.0, (x1 - x0) * scale), max(1.0, (y1 - y0) * scale)
        half = 0.5 * scale
        mapped = BBoxXYWH(
            xPx=(x0 + x1) * 0.5 * scale + prior.xPx + 0.5 * prior.widthPx - half - 0.5 * width,
            yPx=(y0 + y1) * 0.5 * scale + prior.yPx + 0.5 * prior.heightPx - half - 0.5 * height,
            widthPx=width,
            heightPx=height,
        )
        mapped = _clipBox(mapped, view.rgb.shape[1], view.rgb.shape[0])
        score = float(torch.sigmoid(output["score"].reshape(-1)[0]).item())
        try:
            return buildRgbObservation(view, ARTrackPrediction(mapped, score, score, score), 0)
        except ModelError:
            return None


def _vector(point) -> np.ndarray:
    return np.array([point.x, point.y, point.z])


def _point(vector: np.ndarray):
    return makeSphericalPoint(
        math.atan2(vector[0], vector[2]), math.asin(max(-1.0, min(1.0, float(vector[1]))))
    )


def _rowPoint(row: np.ndarray):
    return makeSphericalPoint(math.radians(row[0]), math.radians(row[1]))


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.arccos(np.clip(float(first @ second), -1.0, 1.0)))


def _continued(older: np.ndarray, newer: np.ndarray, steps: float) -> np.ndarray:
    axis = np.cross(older, newer)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return newer
    axis /= norm
    angle = _angle(older, newer) * steps
    return newer * math.cos(angle) + np.cross(axis, newer) * math.sin(angle)


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    session = createArtrackSession(config)
    frameLevel = FrameLevel(session)
    print(f"frame-level model: {frameLevel.unused} parameter groups of the checkpoint unused")
    planner = ViewPlanner(config.geometry, config.tracking, config.backendTuning)
    spheres = loadTrackerResults(args.run / "bfov" / args.method)
    rows: list[dict[str, object]] = []
    paths = sorted((args.run / "trace" / args.method).glob("*.csv"))
    try:
        for number, path in enumerate(paths, 1):
            name = path.stem
            sequence = dataset.sequence(name)
            truthBox, present = sequence.groundTruth("bbox")
            truthBfov, _ = sequence.groundTruth("bfov")
            present = np.asarray(present, dtype=bool)
            with path.open(encoding="utf-8") as stream:
                trace = list(csv.DictReader(stream))
            runtime = buildRuntime(config, artrackSessionFactory=lambda _: _SharedSession(session))
            try:
                first = FramePacket(SequenceId(name), FrameIndex(0), 0, sequence.readRgb(0))
                plan = runtime.controller.buildInitialization(
                    first, initialBfov=sequence.initialBfov()
                )
                template = runtime.geometry.cropViews(first, [plan.templateView])[0]
                runtime.backend.initialize(template, plan.templateBox)
                runtime.verifier.setTemplate(template, plan.templateBox)
                templateTensor = runtime.backend._template.tensor
                tracked = spheres[name]

                def sequenceLevel(view, runtime=runtime):
                    found = runtime.backend.inferDetached((view,))
                    return found[0] if found else None

                def frameLevelPass(view, templateTensor=templateTensor):
                    return frameLevel.infer(view, templateTensor)

                passes = {"S": sequenceLevel, "F": frameLevelPass}
                trusted = previous = lostSince = None
                for row in trace:
                    index = int(row["frame"])
                    if row["modeBefore"] == "LOST" and lostSince is None:
                        lostSince = index
                    if row["modeBefore"] != "LOST":
                        lostSince = None
                    if row["scan"] in KINDS and previous is not None and trusted is not None:
                        rows.extend(
                            _attempt(
                                args,
                                runtime,
                                planner,
                                passes,
                                sequence,
                                name,
                                index,
                                row,
                                int(trusted["frame"]),
                                previous,
                                lostSince,
                                tracked,
                                truthBox,
                                truthBfov,
                                present,
                            )
                        )
                    if row["modeAfter"] == "TRACKING" and row["action"] != "jump":
                        trusted = row
                    previous = row
            finally:
                closeRuntime(runtime)
                sequence.close()
            print(f"[{number}/{len(paths)}] {name}: {len(rows)} boxes so far", flush=True)
    finally:
        session.close()
    return rows


def _attempt(
    args,
    runtime,
    planner,
    passes,
    sequence,
    name,
    index,
    row,
    trustedFrame,
    previous,
    lostSince,
    tracked,
    truthBox,
    truthBfov,
    present,
) -> list[dict[str, object]]:
    frame = FramePacket(SequenceId(name), FrameIndex(index), index, sequence.readRgb(index))
    height, width = frame.rgb.shape[:2]
    anchor = tracked[trustedFrame]
    centre = _rowPoint(anchor)
    wide = min(max(math.radians(anchor[2]), 1e-4), 2 * math.pi - 1e-3)
    tall = min(max(math.radians(anchor[3]), 1e-4), math.pi - 1e-3)
    size = math.sqrt(wide * tall)
    # Where the target would be had it kept moving as before the trusted frame.
    older = _vector(_rowPoint(tracked[max(0, trustedFrame - VELOCITY_FRAMES)]))
    span = max(1, trustedFrame - max(0, trustedFrame - VELOCITY_FRAMES))
    expected = _continued(older, _vector(centre), min(index - trustedFrame, HORIZON) / span)
    one = planner.probeView(centre, wide, tall, 1.0, 1)
    four = planner.probeView(centre, wide, tall, 4.0, 2)
    isPresent = bool(present[index])
    truth = _vector(_rowPoint(truthBfov[index])) if isPresent else None
    targetSize = (
        math.radians(math.sqrt(max(truthBfov[index, 2] * truthBfov[index, 3], 1e-6)))
        if isPresent
        else float("nan")
    )
    trackedPlace = _vector(
        makeSphericalPoint(
            math.radians(float(previous["yawDeg"])), math.radians(float(previous["pitchDeg"]))
        )
    )
    trackedSize = max(math.radians(float(previous["sizeDeg"])), 1e-5)
    base = {
        "sequence": name,
        "frame": index,
        "scan": row["scan"],
        "lostFrames": 0 if lostSince is None else index - lostSince,
        "present": int(isPresent),
        "targetSizeDeg": round(math.degrees(targetSize), 3) if isPresent else "",
        "trustedSizeDeg": round(math.degrees(size), 3),
        "reach1": round(_angle(truth, _vector(centre)) / (0.5 * one.bfov.horizontalFovRad), 3)
        if isPresent
        else "",
        "reach4": round(_angle(truth, _vector(centre)) / (0.5 * four.bfov.horizontalFovRad), 3)
        if isPresent
        else "",
        "trackedScore": row["backend"] if row["hasBox"] == "1" else "",
        "trackedSimilarity": row["appearance"],
    }

    def look(spec, model: str):
        view = runtime.geometry.cropViews(frame, [spec])[0]
        local = passes[model](view)
        if local is None:
            return None
        try:
            projected = runtime.geometry.projectLocalBoxBoundary(local.bbox, spec, width, height)
        except GeometryError:
            return None
        return view, local, projected

    def record(method: str, result, coarseScore: object = "") -> dict[str, object]:
        item = dict(base, method=method, found=0, coarseScore=coarseScore)
        if result is None:
            return item
        view, local, projected = result
        place = _vector(projected.bfov.center)
        boxSize = math.sqrt(projected.bfov.horizontalFovRad * projected.bfov.verticalFovRad)
        offset = _angle(place, expected) / size
        logScale = math.log(boxSize / size)
        similarity = runtime.verifier.similarity(view, local.bbox)
        item.update(
            found=1,
            score=round(float(local.fusedScore), 4),
            similarity="" if similarity is None else round(float(similarity), 4),
            motion=round(
                math.exp(-0.5 * (offset / OFFSET_SCALE) ** 2)
                * math.exp(-0.5 * (logScale / SIZE_SCALE) ** 2),
                4,
            ),
            motionOffset=round(offset, 3),
            motionLogScale=round(logScale, 3),
            iou=0.0,
            onTarget=0,
            onTracked=int(
                _angle(place, trackedPlace) < 0.5 * max(trackedSize, boxSize)
                and max(trackedSize, boxSize) < 1.5 * min(trackedSize, boxSize)
            ),
        )
        if isPresent:
            box = projected.bbox
            iou = float(
                dualIou(
                    truthBox[index : index + 1],
                    np.asarray([[box.xPx, box.yPx, box.widthPx, box.heightPx]]),
                    width,
                )[0]
            )
            error = _angle(place, truth)
            item.update(
                iou=round(iou, 4),
                centreErrorSizes=round(error / targetSize, 3),
                onTarget=int(error < 0.5 * max(targetSize, boxSize)),
            )
        return item

    out = []
    coarse = {}
    for model in ("S", "F"):
        out.append(record(f"{model}:1x", look(one, model)))
        coarse[model] = look(four, model)
        out.append(record(f"{model}:4x", coarse[model]))
    for locate, box in (("S", "S"), ("F", "F"), ("F", "S")):
        method = f"{locate}:4x>1x" if locate == box else f"{locate}:4x>{box}:1x"
        if coarse[locate] is None:
            out.append(record(method, None))
            continue
        fine = planner.probeView(coarse[locate][2].bfov.center, wide, tall, 1.0, 64)
        out.append(record(method, look(fine, box), round(float(coarse[locate][1].fusedScore), 4)))
    release = getattr(runtime.geometry, "releaseFrame", None)
    if callable(release):
        release()
    return out


def _auroc(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        return float("nan")
    values = np.asarray(positive + negative)
    ranks = values.argsort().argsort().astype(float) + 1
    # Ties: average ranks.
    for value in np.unique(values):
        same = values == value
        if same.sum() > 1:
            ranks[same] = ranks[same].mean()
    total = ranks[: len(positive)].sum()
    return float(
        (total - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative))
    )


def _number(row: dict[str, object], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def report(rows: list[dict[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    print("\nthe three gates (attempts with the target in the picture)")
    print(
        f"  {'method':<12}{'attempts':>9}{'in view':>9}{'gate 1':>8}{'on target':>10}{'gate 2':>8}"
        f"{'accepted':>9}{'gate 3':>8}{'all':>7}{'IoU on target':>15}"
    )
    for method in METHODS:
        part = [r for r in rows if r["method"] == method and int(r["present"])]
        reach = "reach1" if method.endswith(":1x") and ">" not in method else "reach4"
        inView = [r for r in part if _number(r, reach) <= 1.0]
        on = [r for r in inView if int(r["found"]) and int(r["onTarget"])]
        taken = [r for r in on if _number(r, "score") >= ACCEPT]
        item = {
            "attempts": len(part),
            "inView": len(inView),
            "onTarget": len(on),
            "accepted": len(taken),
            "meanIouOnTarget": float(np.mean([_number(r, "iou") for r in on])) if on else None,
        }
        payload[method] = item
        print(
            f"  {method:<12}{len(part):>9}{len(inView):>9}{len(inView) / max(1, len(part)):>8.1%}"
            f"{len(on):>10}{len(on) / max(1, len(inView)):>8.1%}{len(taken):>9}"
            f"{len(taken) / max(1, len(on)):>8.1%}{len(taken) / max(1, len(part)):>7.1%}"
            f"{(item['meanIouOnTarget'] or 0.0):>15.3f}"
        )
    print("\ngate 2 by how far the target is from the middle of the view (share of its half-width)")
    for method in METHODS:
        part = [r for r in rows if r["method"] == method and int(r["present"])]
        reach = "reach1" if method.endswith(":1x") and ">" not in method else "reach4"
        cells = []
        for low, high in ((0, 0.125), (0.125, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
            band = [r for r in part if low <= _number(r, reach) < high]
            on = [r for r in band if int(r["found"]) and int(r["onTarget"])]
            cells.append(f"{len(on)}/{len(band)} {len(on) / max(1, len(band)):.0%}")
            payload[method][f"reach {low}-{high}"] = [len(on), len(band)]
        print(f"  {method:<12}" + "".join(f"{cell:>16}" for cell in cells))
    print("\nscores of the returned boxes that moved away from the tracked box")
    print(
        f"  {'method':<12}{'':<11}{'n':>6}{'score':>18}{'similarity':>18}{'motion':>18}{'IoU':>18}"
    )
    for method in METHODS:
        if method.endswith(":4x"):
            continue
        part = [r for r in rows if r["method"] == method and int(r["found"])]
        moved = [r for r in part if not int(r["onTracked"])]
        groups = {
            "on target": [r for r in moved if int(r["onTarget"])],
            "elsewhere": [r for r in moved if not int(r["onTarget"])],
        }
        payload[method]["scores"] = {}
        for label, group in groups.items():
            cells = []
            payload[method]["scores"][label] = {"n": len(group)}
            for key in ("score", "similarity", "motion", "iou"):
                values = np.asarray([_number(r, key) for r in group])
                values = values[~np.isnan(values)]
                if values.size:
                    quartiles = np.percentile(values, [25, 50, 75])
                    cells.append(f"{quartiles[1]:.2f} ({quartiles[0]:.2f}-{quartiles[2]:.2f})")
                    payload[method]["scores"][label][key] = [float(q) for q in quartiles]
                else:
                    cells.append("")
            print(f"  {method:<12}{label:<11}{len(group):>6}" + "".join(f"{c:>18}" for c in cells))
        aurocs = {}
        for key in ("score", "similarity", "motion"):
            aurocs[key] = _auroc(
                [_number(r, key) for r in groups["on target"] if not math.isnan(_number(r, key))],
                [_number(r, key) for r in groups["elsewhere"] if not math.isnan(_number(r, key))],
            )
        payload[method]["auroc"] = aurocs
        print(
            f"  {method:<12}{'AUROC':<11}{'':>6}{aurocs['score']:>18.3f}"
            f"{aurocs['similarity']:>18.3f}{aurocs['motion']:>18.3f}"
        )
    print("\naccepting by the score of the pass (boxes away from the tracked box)")
    print(
        f"  {'method':<12}{'threshold':>10}{'accepted':>9}{'on target':>10}"
        f"{'precision':>10}{'recall':>8}"
    )
    for method in METHODS:
        if method.endswith(":4x"):
            continue
        moved = [
            r for r in rows if r["method"] == method and int(r["found"]) and not int(r["onTracked"])
        ]
        right = [r for r in moved if int(r["onTarget"])]
        payload[method]["thresholds"] = {}
        for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            taken = [r for r in moved if _number(r, "score") >= threshold]
            good = [r for r in taken if int(r["onTarget"])]
            payload[method]["thresholds"][str(threshold)] = [len(taken), len(good)]
            print(
                f"  {method:<12}{threshold:>10.2f}{len(taken):>9}{len(good):>10}"
                f"{len(good) / max(1, len(taken)):>10.1%}{len(good) / max(1, len(right)):>8.1%}"
            )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--method", default="ours")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    table = args.out / "boxes.csv"
    if args.report_only:
        with table.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = run(args)
        if not rows:
            print("no search attempts in the trace", file=sys.stderr)
            return 2
        args.out.mkdir(parents=True, exist_ok=True)
        with table.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, restval="")
            writer.writeheader()
            writer.writerows(rows)
    payload = report(rows)
    (args.out / "frame_level_search.json").write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
