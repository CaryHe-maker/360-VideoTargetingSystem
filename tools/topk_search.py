"""Can a cheap similarity map point at a lost target, and can the scores tell which point is it?

    python tools/topk_search.py peaks    --dataset-root <dir> --label-root <dir> \\
        --run outputs/E033_a_train --config configs/loss_handling.yaml --out outputs/E036
    python tools/topk_search.py forwards --dataset-root <dir> --label-root <dir> \\
        --run outputs/E033_a_train --config configs/loss_handling.yaml --out outputs/E036 \\
        --variant mean
    python tools/topk_search.py report --out outputs/E036

Only forward passes: nothing is tracked.  The search attempts of a finished run with
``scanMode: zoom`` are taken from its state trace.  At each of them a region three
search views wide (twelve target sizes) around the place the track was last trusted is
rendered at 448 x 448 pixels and passed through DINOv2 ViT-S/14 once, giving one
feature per 14-pixel patch (32 x 32).  Each patch is compared with the frame-0 template
and the seven best separated peaks of the map are kept (``peaks``):

    mean     cosine with the mean patch feature of the template crop (224 pixels)
    scaled   the same, the template first shrunk to the size the target has in the region
    dense    the best cosine with any single patch of the template crop
    grid     no model: the middle and six points two target sizes around it (a control)

A peak is a hit when it is within one target size of the true centre, the range in
which a search view of the normal size boxes the target four times in five.

``forwards`` takes the seven points of one variant and, around each, a view of the
normal size and a stateless pass of the tracker.  Every returned box is one row with
the score of the pass, the similarity of the box to the template, a motion score (as in
``frame_level_search.py``) and the true IoU.  ``report`` reads the tables back.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

VARIANTS = ("mean", "scaled", "dense", "grid", "apart2", "apart3", "kmeans")
REGION_SCALE = 3.0
REGION_PX = 448
PATCH = 14
TOP = 7
HIT_SIZES = 1.0
ACCEPT = 0.70
PEAK_COLUMNS = (
    "sequence",
    "frame",
    "scan",
    "lostFrames",
    "present",
    "targetSizeDeg",
    "trustedSizeDeg",
    "reach",
    "regionDeg",
    "targetPatches",
    "variant",
    "rank",
    "value",
    "yawDeg",
    "pitchDeg",
    "distanceSizes",
    "dinoMs",
)
BOX_COLUMNS = (
    "sequence",
    "frame",
    "present",
    "variant",
    "rank",
    "value",
    "pointDistanceSizes",
    "found",
    "score",
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


def _attempts(args):
    """The search attempts of the run: one dict per frame, with what is needed to look."""
    from frame_level_search import HORIZON, KINDS, VELOCITY_FRAMES, _continued, _rowPoint, _vector

    from track360.datasets.vot360 import Vot360Dataset
    from track360.evaluation.vot360_metrics import loadTrackerResults

    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    spheres = loadTrackerResults(args.run / "bfov" / args.method)
    for path in sorted((args.run / "trace" / args.method).glob("*.csv")):
        name = path.stem
        sequence = dataset.sequence(name)
        truthBox, present = sequence.groundTruth("bbox")
        truthBfov, _ = sequence.groundTruth("bfov")
        present = np.asarray(present, dtype=bool)
        with path.open(encoding="utf-8") as stream:
            trace = list(csv.DictReader(stream))
        tracked = spheres[name]
        chosen = []
        trusted = previous = lostSince = None
        for row in trace:
            index = int(row["frame"])
            if row["modeBefore"] == "LOST" and lostSince is None:
                lostSince = index
            if row["modeBefore"] != "LOST":
                lostSince = None
            if row["scan"] in KINDS and previous is not None and trusted is not None:
                anchorFrame = int(trusted["frame"])
                anchor = tracked[anchorFrame]
                centre = _rowPoint(anchor)
                wide = min(max(math.radians(anchor[2]), 1e-4), 2 * math.pi - 1e-3)
                tall = min(max(math.radians(anchor[3]), 1e-4), math.pi - 1e-3)
                older = _vector(_rowPoint(tracked[max(0, anchorFrame - VELOCITY_FRAMES)]))
                span = max(1, anchorFrame - max(0, anchorFrame - VELOCITY_FRAMES))
                chosen.append(
                    {
                        "frame": index,
                        "scan": row["scan"],
                        "lostFrames": 0 if lostSince is None else index - lostSince,
                        "centre": centre,
                        "wide": wide,
                        "tall": tall,
                        "expected": _continued(
                            older, _vector(centre), min(index - anchorFrame, HORIZON) / span
                        ),
                        "present": bool(present[index]),
                        "truth": _vector(_rowPoint(truthBfov[index])) if present[index] else None,
                        "truthBox": truthBox[index],
                        "targetSize": math.radians(
                            math.sqrt(max(truthBfov[index, 2] * truthBfov[index, 3], 1e-6))
                        ),
                        "trackedPlace": _vector(
                            _rowPoint(
                                np.array([float(previous["yawDeg"]), float(previous["pitchDeg"])])
                            )
                        ),
                        "trackedSize": max(math.radians(float(previous["sizeDeg"])), 1e-5),
                        "trackedScore": row["backend"] if row["hasBox"] == "1" else "",
                        "trackedSimilarity": row["appearance"],
                    }
                )
            if row["modeAfter"] == "TRACKING" and row["action"] != "jump":
                trusted = row
            previous = row
        yield name, sequence, chosen


def _eventAttempts(args):
    """The frames at which the target came back or arrived after a jump while the run's
    tracker was off it, looked at from the last true position: the ideal starting point."""
    from frame_level_search import _rowPoint, _vector
    from oracle_restart import events

    from track360.datasets.vot360 import Vot360Dataset
    from track360.evaluation.loss_rate import dualIou
    from track360.evaluation.vot360_metrics import loadTrackerResults

    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    results = loadTrackerResults(args.run / "bbox" / args.method)
    for name in sorted(results):
        sequence = dataset.sequence(name)
        truthBox, present = sequence.groundTruth("bbox")
        truthBfov, _ = sequence.groundTruth("bfov")
        present = np.asarray(present, dtype=bool)
        iou = np.zeros(len(truthBox))
        iou[present] = dualIou(truthBox[present], results[name][present], sequence.frameSize[0])
        chosen = []
        for index, kind in sorted(events(truthBfov, present, 1.0, 10).items()):
            last = index - 1
            while last >= 0 and not present[last]:
                last -= 1
            if last < 0 or iou[index] >= 0.1:
                continue
            anchor = truthBfov[last]
            chosen.append(
                {
                    "frame": index,
                    "scan": kind,
                    "lostFrames": index - last - 1,
                    "centre": _rowPoint(anchor),
                    "wide": min(max(math.radians(anchor[2]), 1e-4), 2 * math.pi - 1e-3),
                    "tall": min(max(math.radians(anchor[3]), 1e-4), math.pi - 1e-3),
                    "present": True,
                    "truth": _vector(_rowPoint(truthBfov[index])),
                    "targetSize": math.radians(
                        math.sqrt(max(truthBfov[index, 2] * truthBfov[index, 3], 1e-6))
                    ),
                }
            )
        yield name, sequence, chosen


def _runtime(args, name, sequence, session):
    from track360.core.types import FrameIndex, FramePacket, SequenceId
    from track360.runtime.benchmark import _SharedSession
    from track360.runtime.driver import buildRuntime

    runtime = buildRuntime(args.loaded, artrackSessionFactory=lambda _: _SharedSession(session))
    first = FramePacket(SequenceId(name), FrameIndex(0), 0, sequence.readRgb(0))
    plan = runtime.controller.buildInitialization(first, initialBfov=sequence.initialBfov())
    template = runtime.geometry.cropViews(first, [plan.templateView])[0]
    runtime.backend.initialize(template, plan.templateBox)
    runtime.verifier.setTemplate(template, plan.templateBox)
    return runtime, template, plan.templateBox


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.arccos(np.clip(float(first @ second), -1.0, 1.0)))


def _tokens(models, image: np.ndarray):
    """Unit-length DINOv2 patch features of an RGB image, as (rows, columns, channels)."""
    torch = models.torch
    tensor = torch.from_numpy(np.ascontiguousarray(image)).to(models._device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = (tensor - models._mean) / models._std
    with torch.inference_mode():
        tokens = models._models["dinov2"].forward_features(tensor)["x_norm_patchtokens"][0]
    side = image.shape[0] // PATCH
    return torch.nn.functional.normalize(tokens, dim=1).reshape(side, image.shape[1] // PATCH, -1)


def _peaks(
    score: np.ndarray, targetPatches: float, apart: float = 1.0
) -> list[tuple[int, int, float]]:
    """The best peaks of a map smoothed over one target size, ``apart`` sizes apart."""
    import cv2

    kernel = int(max(1, round(targetPatches)))
    smooth = cv2.blur(score.astype(np.float32), (kernel, kernel)) if kernel > 1 else score.copy()
    radius = max(1.5, apart * targetPatches)
    rows, columns = np.mgrid[0 : smooth.shape[0], 0 : smooth.shape[1]]
    picked = []
    work = smooth.copy()
    for _ in range(TOP):
        flat = int(np.argmax(work))
        row, column = divmod(flat, work.shape[1])
        if not np.isfinite(work[row, column]):
            break
        picked.append((row, column, float(smooth[row, column])))
        work[(rows - row) ** 2 + (columns - column) ** 2 <= radius**2] = -np.inf
    return picked


def _clusters(score: np.ndarray, targetPatches: float) -> list[tuple[int, int, float]]:
    """Weighted k-means over the most similar fifth of the patches, heaviest cluster first."""
    import cv2

    kernel = int(max(1, round(targetPatches)))
    smooth = cv2.blur(score.astype(np.float32), (kernel, kernel)) if kernel > 1 else score.copy()
    floor = float(np.percentile(smooth, 80))
    rows, columns = np.nonzero(smooth >= floor)
    weights = (smooth[rows, columns] - floor + 1e-6).astype(np.float64)
    places = np.column_stack([rows, columns]).astype(np.float64)
    centres = np.asarray(
        [(row, column) for row, column, _ in _peaks(score, targetPatches, 2.0)], dtype=np.float64
    )
    if len(centres) == 0:
        return []
    for _ in range(15):
        nearest = ((places[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        for k in range(len(centres)):
            chosen = nearest == k
            if chosen.any():
                centres[k] = (places[chosen] * weights[chosen, None]).sum(axis=0) / weights[
                    chosen
                ].sum()
    mass = [float(weights[nearest == k].sum()) for k in range(len(centres))]
    order = np.argsort(mass)[::-1]
    return [(float(centres[k][0]), float(centres[k][1]), mass[k]) for k in order]


def peaks(args: argparse.Namespace) -> int:
    import cv2
    from frame_level_search import _vector

    from track360.backends import createArtrackSession
    from track360.backends.appearance import targetCrop
    from track360.controller.track_controller import _shifted
    from track360.controller.view_planner import ViewPlanner
    from track360.core.types import BBoxXYWH, FrameIndex, FramePacket, SequenceId
    from track360.runtime.driver import closeRuntime

    global REGION_SCALE, REGION_PX
    REGION_SCALE, REGION_PX = args.region_scale, args.region_px
    config = args.loaded
    session = createArtrackSession(config)
    planner = ViewPlanner(config.geometry, config.tracking, config.backendTuning)
    rows: list[dict[str, object]] = []
    try:
        source = _eventAttempts(args) if args.centre == "truth" else _attempts(args)
        for number, (name, sequence, chosen) in enumerate(source, 1):
            runtime, template, templateBox = _runtime(args, name, sequence, session)
            models = runtime.verifier._models
            torch = models.torch
            try:
                crop = targetCrop(template.rgb, templateBox)
                templateTokens = _tokens(models, crop).reshape(-1, 384)
                prototype = torch.nn.functional.normalize(templateTokens.mean(dim=0), dim=0)
                for item in chosen:
                    index = item["frame"]
                    frame = FramePacket(
                        SequenceId(name), FrameIndex(index), index, sequence.readRgb(index)
                    )
                    height, width = frame.rgb.shape[:2]
                    size = math.sqrt(item["wide"] * item["tall"])
                    region = planner.probeView(
                        item["centre"], item["wide"], item["tall"], REGION_SCALE, 2
                    )
                    region = replace(
                        region, outputWidthPx=REGION_PX, outputHeightPx=REGION_PX, priorBox=None
                    )
                    view = runtime.geometry.cropViews(frame, [region])[0]
                    regionFov = region.bfov.horizontalFovRad
                    targetPatches = REGION_PX * size / regionFov / PATCH
                    torch.cuda.synchronize()
                    started = perf_counter()
                    tokens = _tokens(models, view.rgb)
                    torch.cuda.synchronize()
                    elapsed = (perf_counter() - started) * 1000.0
                    maps = {"mean": (tokens @ prototype).cpu().numpy()}
                    small = int(max(1, round(targetPatches))) * PATCH
                    smallTokens = _tokens(
                        models, cv2.resize(crop, (small, small), interpolation=cv2.INTER_AREA)
                    ).reshape(-1, 384)
                    maps["scaled"] = (
                        (tokens @ torch.nn.functional.normalize(smallTokens.mean(dim=0), dim=0))
                        .cpu()
                        .numpy()
                    )
                    maps["dense"] = (
                        (tokens.reshape(-1, 384) @ templateTokens.T)
                        .max(dim=1)
                        .values.reshape(tokens.shape[0], tokens.shape[1])
                        .cpu()
                        .numpy()
                    )
                    points: dict[str, list[tuple[object, float]]] = {}
                    box = max(4.0, targetPatches * PATCH)
                    found = {
                        variant: _peaks(score, targetPatches) for variant, score in maps.items()
                    }
                    found["apart2"] = _peaks(maps["mean"], targetPatches, 2.0)
                    found["apart3"] = _peaks(maps["mean"], targetPatches, 3.0)
                    found["kmeans"] = _clusters(maps["mean"], targetPatches)
                    for variant, picked in found.items():
                        points[variant] = []
                        for row, column, value in picked:
                            x = min(max((column + 0.5) * PATCH, box / 2), REGION_PX - box / 2)
                            y = min(max((row + 0.5) * PATCH, box / 2), REGION_PX - box / 2)
                            try:
                                projected = runtime.geometry.projectLocalBoxBoundary(
                                    BBoxXYWH(x - box / 2, y - box / 2, box, box),
                                    region,
                                    width,
                                    height,
                                )
                            except Exception:
                                continue
                            points[variant].append((projected.bfov.center, value))
                    step = 2.0 * size
                    ring = [(0.0, 0.0)] + [
                        (step * math.cos(k * math.pi / 3), step * math.sin(k * math.pi / 3))
                        for k in range(6)
                    ]
                    points["grid"] = [(_shifted(item["centre"], dx, dy), 0.0) for dx, dy in ring]
                    reach = (
                        _angle(item["truth"], _vector(item["centre"])) / (0.5 * regionFov)
                        if item["present"]
                        else ""
                    )
                    for variant, located in points.items():
                        for rank, (point, value) in enumerate(located, 1):
                            distance = (
                                _angle(_vector(point), item["truth"]) / item["targetSize"]
                                if item["present"]
                                else ""
                            )
                            rows.append(
                                {
                                    "sequence": name,
                                    "frame": index,
                                    "scan": item["scan"],
                                    "lostFrames": item["lostFrames"],
                                    "present": int(item["present"]),
                                    "targetSizeDeg": round(math.degrees(item["targetSize"]), 3),
                                    "trustedSizeDeg": round(math.degrees(size), 3),
                                    "reach": reach if reach == "" else round(reach, 3),
                                    "regionDeg": round(math.degrees(regionFov), 2),
                                    "targetPatches": round(targetPatches, 2),
                                    "variant": variant,
                                    "rank": rank,
                                    "value": round(value, 4),
                                    "yawDeg": round(math.degrees(point.yawRad), 4),
                                    "pitchDeg": round(math.degrees(point.pitchRad), 4),
                                    "distanceSizes": distance
                                    if distance == ""
                                    else round(distance, 3),
                                    "dinoMs": round(elapsed, 2),
                                }
                            )
                    release = getattr(runtime.geometry, "releaseFrame", None)
                    if callable(release):
                        release()
            finally:
                closeRuntime(runtime)
                sequence.close()
            print(f"[{number}] {name}: {len(rows)} points so far", flush=True)
    finally:
        session.close()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "peaks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PEAK_COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return 0


def forwards(args: argparse.Namespace) -> int:
    from frame_level_search import OFFSET_SCALE, SIZE_SCALE, _vector

    from track360.backends import createArtrackSession
    from track360.controller.view_planner import ViewPlanner
    from track360.core.errors import GeometryError
    from track360.core.types import FrameIndex, FramePacket, SequenceId
    from track360.evaluation.loss_rate import dualIou
    from track360.geometry.projection_math import makeSphericalPoint
    from track360.runtime.driver import closeRuntime

    with (args.out / "peaks.csv").open(encoding="utf-8") as stream:
        table = [row for row in csv.DictReader(stream) if row["variant"] == args.variant]
    wanted: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in table:
        wanted.setdefault((row["sequence"], int(row["frame"])), []).append(row)
    config = args.loaded
    session = createArtrackSession(config)
    planner = ViewPlanner(config.geometry, config.tracking, config.backendTuning)
    rows: list[dict[str, object]] = []
    try:
        for number, (name, sequence, chosen) in enumerate(_attempts(args), 1):
            runtime, _, _ = _runtime(args, name, sequence, session)
            try:
                for item in chosen:
                    index = item["frame"]
                    points = wanted.get((name, index), [])
                    if not points:
                        continue
                    frame = FramePacket(
                        SequenceId(name), FrameIndex(index), index, sequence.readRgb(index)
                    )
                    height, width = frame.rgb.shape[:2]
                    size = math.sqrt(item["wide"] * item["tall"])
                    for point in points:
                        centre = makeSphericalPoint(
                            math.radians(float(point["yawDeg"])),
                            math.radians(float(point["pitchDeg"])),
                        )
                        spec = planner.probeView(centre, item["wide"], item["tall"], 1.0, 64)
                        view = runtime.geometry.cropViews(frame, [spec])[0]
                        found = runtime.backend.inferDetached((view,))
                        record: dict[str, object] = {
                            "sequence": name,
                            "frame": index,
                            "present": int(item["present"]),
                            "variant": args.variant,
                            "rank": int(point["rank"]),
                            "value": point["value"],
                            "pointDistanceSizes": point["distanceSizes"],
                            "found": 0,
                            "trackedScore": item["trackedScore"],
                            "trackedSimilarity": item["trackedSimilarity"],
                        }
                        projected = None
                        if found:
                            try:
                                projected = runtime.geometry.projectLocalBoxBoundary(
                                    found[0].bbox, spec, width, height
                                )
                            except GeometryError:
                                projected = None
                        if projected is not None:
                            place = _vector(projected.bfov.center)
                            boxSize = math.sqrt(
                                projected.bfov.horizontalFovRad * projected.bfov.verticalFovRad
                            )
                            offset = _angle(place, item["expected"]) / size
                            logScale = math.log(boxSize / size)
                            similarity = runtime.verifier.similarity(view, found[0].bbox)
                            record.update(
                                found=1,
                                score=round(float(found[0].fusedScore), 4),
                                similarity=""
                                if similarity is None
                                else round(float(similarity), 4),
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
                                    _angle(place, item["trackedPlace"])
                                    < 0.5 * max(item["trackedSize"], boxSize)
                                    and max(item["trackedSize"], boxSize)
                                    < 1.5 * min(item["trackedSize"], boxSize)
                                ),
                            )
                            if item["present"]:
                                box = projected.bbox
                                iou = float(
                                    dualIou(
                                        item["truthBox"][None, :],
                                        np.asarray([[box.xPx, box.yPx, box.widthPx, box.heightPx]]),
                                        width,
                                    )[0]
                                )
                                error = _angle(place, item["truth"])
                                record.update(
                                    iou=round(iou, 4),
                                    centreErrorSizes=round(error / item["targetSize"], 3),
                                    onTarget=int(error < 0.5 * max(item["targetSize"], boxSize)),
                                )
                        rows.append(record)
                    release = getattr(runtime.geometry, "releaseFrame", None)
                    if callable(release):
                        release()
            finally:
                closeRuntime(runtime)
                sequence.close()
            print(f"[{number}] {name}: {len(rows)} forwards so far", flush=True)
    finally:
        session.close()
    with (args.out / f"boxes_{args.variant}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=BOX_COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return 0


def _number(row: dict[str, object], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def _auroc(positive: list[float], negative: list[float]) -> float:
    positive = [v for v in positive if not math.isnan(v)]
    negative = [v for v in negative if not math.isnan(v)]
    if not positive or not negative:
        return float("nan")
    values = np.asarray(positive + negative)
    order = values.argsort()
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    for value in np.unique(values):
        same = values == value
        if same.sum() > 1:
            ranks[same] = ranks[same].mean()
    total = ranks[: len(positive)].sum()
    return float(
        (total - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative))
    )


def report(args: argparse.Namespace) -> int:
    payload: dict[str, object] = {}
    with (args.out / "peaks.csv").open(encoding="utf-8") as stream:
        peakRows = list(csv.DictReader(stream))
    attempts: dict[str, dict[tuple[str, str], list[dict[str, str]]]] = {}
    for row in peakRows:
        attempts.setdefault(row["variant"], {}).setdefault(
            (row["sequence"], row["frame"]), []
        ).append(row)
    times = [float(r["dinoMs"]) for r in peakRows if r["variant"] == "mean" and r["rank"] == "1"]
    print(
        f"DINOv2 ViT-S/14 on the {REGION_PX}-pixel region: median {np.median(times):.1f} ms "
        f"(quartiles {np.percentile(times, 25):.1f}-{np.percentile(times, 75):.1f})"
    )
    payload["dinoMs"] = float(np.median(times))
    print(f"\nis the true target within {HIT_SIZES:g} target size of one of the first k points?")
    for label, keep in (
        ("attempts with the target in the picture", lambda rows: rows[0]["present"] == "1"),
        (
            "... and inside the region",
            lambda rows: rows[0]["present"] == "1" and float(rows[0]["reach"]) <= 1.0,
        ),
    ):
        print(f"  {label}")
        print(f"    {'points':<9}{'n':>6}{'top 1':>8}{'top 3':>8}{'top 5':>8}{'top 7':>8}")
        payload[label] = {}
        for variant in VARIANTS:
            groups = [rows for rows in attempts.get(variant, {}).values() if keep(rows)]
            if not groups:
                continue
            shares = []
            for k in (1, 3, 5, 7):
                hit = [
                    any(
                        int(r["rank"]) <= k and float(r["distanceSizes"]) <= HIT_SIZES for r in rows
                    )
                    for rows in groups
                ]
                shares.append(float(np.mean(hit)))
            payload[label][variant] = {"n": len(groups), "hit": shares}
            print(f"    {variant:<9}{len(groups):>6}" + "".join(f"{s:>8.1%}" for s in shares))
    print("\n  inside the region, by target size (top 7)")
    for low, high in ((0, 3), (3, 6), (6, 12), (12, 400)):
        cells = []
        for variant in VARIANTS:
            groups = [
                rows
                for rows in attempts.get(variant, {}).values()
                if rows[0]["present"] == "1"
                and float(rows[0]["reach"]) <= 1.0
                and low <= float(rows[0]["targetSizeDeg"]) < high
            ]
            hit = [any(float(r["distanceSizes"]) <= HIT_SIZES for r in rows) for rows in groups]
            cells.append(f"{variant} {np.mean(hit) if hit else float('nan'):.0%} (n={len(groups)})")
        print(f"    {low}-{high} deg: " + ", ".join(cells))

    for path in sorted(args.out.glob("boxes_*.csv")):
        variant = path.stem.split("_", 1)[1]
        with path.open(encoding="utf-8") as stream:
            boxes = list(csv.DictReader(stream))
        byAttempt: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in boxes:
            byAttempt.setdefault((row["sequence"], row["frame"]), []).append(row)
        present = [rows for rows in byAttempt.values() if rows[0]["present"] == "1"]
        print(f"\n=== forwards around the points of '{variant}': {len(byAttempt)} attempts")
        item: dict[str, object] = {}
        print("  gates (attempts with the target in the picture)")
        print(
            f"    {'first k':<9}{'attempts':>9}{'a point near':>14}{'a box on target':>17}"
            f"{'... scoring 0.70':>18}{'all':>7}"
        )
        for k in (1, 3, 5, 7):
            near = [
                any(
                    int(r["rank"]) <= k and _number(r, "pointDistanceSizes") <= HIT_SIZES
                    for r in rows
                )
                for rows in present
            ]
            on = [
                any(int(r["rank"]) <= k and r["onTarget"] == "1" for r in rows) for rows in present
            ]
            taken = [
                any(
                    int(r["rank"]) <= k and r["onTarget"] == "1" and _number(r, "score") >= ACCEPT
                    for r in rows
                )
                for rows in present
            ]
            item[f"top{k}"] = {
                "near": float(np.mean(near)),
                "onTarget": float(np.mean(on)),
                "accepted": float(np.mean(taken)),
            }
            print(
                f"    {k:<9}{len(present):>9}{np.mean(near):>14.1%}{np.mean(on):>17.1%}"
                f"{(sum(taken) / max(1, sum(on))):>18.1%}{np.mean(taken):>7.1%}"
            )
        found = [r for r in boxes if r["found"] == "1"]
        groups = {
            "on target": [r for r in found if r["onTarget"] == "1"],
            "elsewhere": [r for r in found if r["onTarget"] != "1"],
        }
        print("  scores of the boxes (median, quartiles)")
        print(
            f"    {'':<11}{'n':>6}{'pass score':>18}{'similarity':>18}{'motion':>18}"
            f"{'offset (sizes)':>18}{'peak value':>18}{'IoU':>18}"
        )
        for label, group in groups.items():
            cells = []
            for key in ("score", "similarity", "motion", "motionOffset", "value", "iou"):
                values = np.asarray([_number(r, key) for r in group])
                values = values[~np.isnan(values)]
                if values.size:
                    q = np.percentile(values, [25, 50, 75])
                    cells.append(f"{q[1]:.2f} ({q[0]:.2f}-{q[2]:.2f})")
                else:
                    cells.append("")
            print(f"    {label:<11}{len(group):>6}" + "".join(f"{c:>18}" for c in cells))
        aurocs = {}
        for key, sign in (
            ("score", 1),
            ("similarity", 1),
            ("motion", 1),
            ("motionOffset", -1),
            ("value", 1),
        ):
            aurocs[key] = _auroc(
                [sign * _number(r, key) for r in groups["on target"]],
                [sign * _number(r, key) for r in groups["elsewhere"]],
            )
        item["auroc"] = aurocs
        print(
            "    AUROC (on target against elsewhere): "
            + ", ".join(f"{key} {value:.3f}" for key, value in aurocs.items())
        )
        print("  picking one box per attempt (attempts where a box is on the target)")
        withTarget = [
            rows for rows in byAttempt.values() if any(r["onTarget"] == "1" for r in rows)
        ]
        rules = {
            "highest pass score": lambda r: _number(r, "score"),
            "highest similarity": lambda r: _number(r, "similarity"),
            "nearest to the expected place": lambda r: -_number(r, "motionOffset"),
            "highest peak value": lambda r: _number(r, "value"),
            "score + similarity": lambda r: _number(r, "score") + _number(r, "similarity"),
            "score + similarity + motion": lambda r: (
                _number(r, "score") + _number(r, "similarity") + _number(r, "motion")
            ),
        }
        item["pick"] = {}
        for label, key in rules.items():
            right = 0
            for rows in withTarget:
                usable = [r for r in rows if r["found"] == "1" and not math.isnan(key(r))]
                if usable and max(usable, key=key)["onTarget"] == "1":
                    right += 1
            item["pick"][label] = right / max(1, len(withTarget))
            print(
                f"    {label:<32}{right:>5} / {len(withTarget)}  "
                f"{right / max(1, len(withTarget)):.1%}"
            )
        print(
            "  taking the box with the highest pass score when it reaches a threshold "
            "(all attempts)"
        )
        print(f"    {'threshold':<11}{'taken':>7}{'on target':>11}{'precision':>11}{'recall':>9}")
        item["thresholds"] = {}
        for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            taken = right = 0
            for rows in byAttempt.values():
                usable = [r for r in rows if r["found"] == "1"]
                if not usable:
                    continue
                best = max(usable, key=lambda r: _number(r, "score"))
                if _number(best, "score") >= threshold:
                    taken += 1
                    right += best["onTarget"] == "1"
            item["thresholds"][str(threshold)] = [taken, right]
            print(
                f"    {threshold:<11.2f}{taken:>7}{right:>11}{right / max(1, taken):>11.1%}"
                f"{right / max(1, len(present)):>9.1%}"
            )
        payload[f"forwards {variant}"] = item
    (args.out / "topk_search.json").write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("peaks", "forwards", "report"))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--method", default="ours")
    parser.add_argument("--variant", default="mean", choices=VARIANTS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--centre", default="trusted", choices=("trusted", "truth"))
    parser.add_argument("--region-scale", type=float, default=REGION_SCALE)
    parser.add_argument("--region-px", type=int, default=REGION_PX)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    if args.command == "report":
        return report(args)
    from track360.core.config import loadConfig

    args.loaded = loadConfig(args.config)
    return peaks(args) if args.command == "peaks" else forwards(args)


if __name__ == "__main__":
    raise SystemExit(main())
