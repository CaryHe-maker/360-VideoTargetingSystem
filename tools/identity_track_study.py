"""Keep the target's identity by tracking it and its look-alikes as objects (E079, offline).

    python tools/identity_track_study.py --dataset-root <dir> --label-root <dir> \\
        --run outputs/E066_s1 --out outputs/E079

Independent of the tracker, on consecutive frames of every dev-train sequence: YOLOE
(the frame-0 target box as its visual prompt) detects objects in a view ``--scale``
times the normal search view of the target, centred where the target's track is
predicted to be.  Detections become tracks by association (ByteTrack-style, one stage):
the predicted place of each track (constant angular velocity), gated at ``--gate``
sizes, and the DINOv2 similarity of the crops, matched with the Hungarian method.  The
target's track starts from the ground-truth box of frame 0 and is never dropped; while
unmatched it coasts on its velocity.  Other tracks are dropped after ``--max-age``
missed frames.

Reported: how often the target's track is on the target (centre within half the
target's size) over all frames with the target in the picture, on the frames the
recorded tracker run has wrong, and 0 / 10 / 30 frames after each onset (the tracker on
the target the frame before, off it now); identity switches of the target's track; and
the time per frame.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, replace
from math import sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from negative_gallery_study import _bfov  # noqa: E402
from proposal_study import VisualPromptYoloe  # noqa: E402
from size_study import angle, sequenceData  # noqa: E402

from track360.backends.appearance import AppearanceModels, targetCrop  # noqa: E402
from track360.controller import TrackControllerImpl  # noqa: E402
from track360.controller.view_planner import localBoxOfBfov  # noqa: E402
from track360.core.config import loadConfig  # noqa: E402
from track360.core.types import BBoxXYWH, FrameIndex, FramePacket, SequenceId  # noqa: E402
from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402
from track360.geometry import SphericalGeometryImpl  # noqa: E402


@dataclass
class Track:
    ident: int
    pos: np.ndarray  # unit vector
    vel: np.ndarray  # per frame
    size: float  # degrees
    feature: np.ndarray
    missed: int = 0

    def predicted(self) -> np.ndarray:
        p = self.pos + self.vel
        return p / np.linalg.norm(p)


def _unitOf(point) -> np.ndarray:
    return np.array([point.x, point.y, point.z], dtype=np.float64)


def _toBfov(direction: np.ndarray, sizeDeg: float):
    from math import asin, atan2, degrees

    yaw = degrees(atan2(direction[0], direction[2]))
    pitch = degrees(asin(max(-1.0, min(1.0, direction[1]))))
    return _bfov(yaw, pitch, sizeDeg, sizeDeg)


def runSequence(name, args, dataset, config, models, yoloe, recorded) -> dict:
    from scipy.optimize import linear_sum_assignment

    sequence = dataset.sequence(name)
    truth, present = sequence.groundTruth("bfov")
    present = np.asarray(present, bool)
    width, height = sequence.frameSize
    geometry = SphericalGeometryImpl(boundarySamplesPerEdge=config.geometry.boundarySamplesPerEdge)
    planner = TrackControllerImpl(geometry, config)._planner

    def view(frame, centre: np.ndarray, sizeDeg: float):
        target = _toBfov(centre, sizeDeg)
        spec = planner.probeView(
            target.center, target.horizontalFovRad, target.verticalFovRad, args.scale, 0
        )
        spec = replace(spec, outputWidthPx=args.width, outputHeightPx=args.width, priorBox=None)
        return spec, geometry.cropViews(frame, [spec])[0].rgb

    def feature(rgb, box: BBoxXYWH) -> np.ndarray:
        return models.embed(targetCrop(rgb, box))["dinov2"].cpu().numpy().reshape(-1)

    onTarget = np.zeros(len(present), bool)
    ms = []
    try:
        first = FramePacket(SequenceId(name), FrameIndex(0), 0, sequence.readRgb(0))
        initial = sequence.initialBfov()
        start = _unitOf(initial.center)
        size0 = float(np.degrees(sqrt(initial.horizontalFovRad * initial.verticalFovRad)))
        spec0, rgb0 = view(first, start, size0)
        box0 = localBoxOfBfov(spec0, initial)
        yoloe.setTarget(rgb0, box0)
        target = Track(0, start, np.zeros(3), size0, feature(rgb0, box0))
        tracks: list[Track] = [target]
        nextId = 1
        switches = 0
        lastOn = True
        for t in range(1, len(present)):
            frame = FramePacket(SequenceId(name), FrameIndex(t), t, sequence.readRgb(t))
            started = time.perf_counter()
            spec, rgb = view(frame, target.predicted(), target.size)
            found = yoloe.boxes(rgb, args.max_det)
            dets = []
            for x0, y0, x1, y1, _ in found:
                local = BBoxXYWH(
                    xPx=x0, yPx=y0, widthPx=max(x1 - x0, 1.0), heightPx=max(y1 - y0, 1.0)
                )
                try:
                    place = geometry.projectLocalBoxBoundary(local, spec, width, height).bfov
                except Exception:
                    continue
                dets.append(
                    (
                        _unitOf(place.center),
                        float(np.degrees(sqrt(place.horizontalFovRad * place.verticalFovRad))),
                        feature(rgb, local),
                    )
                )
            # One object often comes back as several overlapping boxes: keep the most
            # confident of those whose centres lie within half a size of each other.
            unique = []
            for det in dets:
                if all(
                    float(angle(det[0], other[0])) >= 0.5 * max(det[1], other[1])
                    for other in unique
                ):
                    unique.append(det)
            dets = unique
            cost = np.full((len(tracks), len(dets)), 9.0)
            for i, tr in enumerate(tracks):
                p = tr.predicted()
                for j, (pos, size, feat) in enumerate(dets):
                    distance = float(angle(p, pos)) / max(tr.size, size, 1e-3)
                    if distance > args.gate:
                        continue
                    cost[i, j] = 0.5 * distance / args.gate + 0.5 * (1.0 - float(tr.feature @ feat))
            pairs = []
            free = list(range(len(dets)))
            # The target's track chooses first; the others share what is left.
            if dets and float(cost[0].min()) < args.max_cost:
                best = int(np.argmin(cost[0]))
                pairs.append((0, best))
                free.remove(best)
            if len(tracks) > 1 and free:
                rest = cost[1:][:, free]
                pairs += [
                    (i + 1, free[j])
                    for i, j in zip(*linear_sum_assignment(rest), strict=True)
                    if rest[i, j] < args.max_cost
                ]
            matchedTracks = {i for i, _ in pairs}
            matchedDets = {j for _, j in pairs}
            for i, j in pairs:
                tr = tracks[i]
                pos, size, feat = dets[j]
                tr.vel = 0.7 * tr.vel + 0.3 * (pos - tr.pos)
                tr.pos, tr.size, tr.missed = pos, 0.8 * tr.size + 0.2 * size, 0
                f = 0.9 * tr.feature + 0.1 * feat
                tr.feature = f / np.linalg.norm(f)
            # Unmatched tracks coast on their velocity; the target's never ends.
            kept = []
            for i, tr in enumerate(tracks):
                if i not in matchedTracks:
                    tr.missed += 1
                    tr.pos = tr.predicted()
                    tr.vel = 0.9 * tr.vel
                if tr is target or tr.missed <= args.max_age:
                    kept.append(tr)
            for j, (pos, size, feat) in enumerate(dets):
                if j not in matchedDets:
                    kept.append(Track(nextId, pos, np.zeros(3), size, feat))
                    nextId += 1
            tracks = kept
            ms.append(1000.0 * (time.perf_counter() - started))
            if present[t]:
                gt = _unitOf(_bfov(*truth[t][:4]).center)
                gsize = float(np.sqrt(max(truth[t, 2] * truth[t, 3], 1e-6)))
                onTarget[t] = float(angle(target.pos, gt)) < 0.5 * max(gsize, 1e-3)
                if args.debug and t <= args.debug:
                    print(
                        f"    t={t} dets={len(dets)} missed={target.missed}"
                        f" dist={float(angle(target.pos, gt)) / max(gsize, 1e-3):.2f}"
                        f" size={target.size:.2f}/{gsize:.2f}"
                        f" detdist={[round(float(angle(d[0], gt)) / max(gsize, 1e-3), 2) for d in dets][:5]}"  # noqa: E501
                        f" cost0={[round(float(c), 2) for c in cost[0][:5]] if len(dets) else []}"
                    )
                if lastOn and not onTarget[t]:
                    switches += 1
                lastOn = bool(onTarget[t])
            release = getattr(geometry, "releaseFrame", None)
            if callable(release):
                release()
    finally:
        sequence.close()
    iou = recorded["iou"]
    trackerRight = present & (iou >= 0.5)
    trackerWrong = present & (iou < 0.1)
    onsets = [t for t in range(1, len(present)) if trackerWrong[t] and trackerRight[t - 1]]
    return {
        "sequence": name,
        "present": int(present[1:].sum()),
        "identityOn": int(onTarget[1:][present[1:]].sum()),
        "trackerRight": int(trackerRight[1:].sum()),
        "trackerWrong": int(trackerWrong[1:].sum()),
        "identityOnTrackerWrong": int((onTarget & trackerWrong)[1:].sum()),
        "onsets": [
            [
                bool(onTarget[t + k]) if t + k < len(present) and present[t + k] else None
                for k in (0, 10, 30)
            ]
            for t in onsets
        ],
        "switches": switches,
        "ms": float(np.median(ms)) if ms else 0.0,
    }


def report(rows: list[dict]) -> str:
    present = sum(r["present"] for r in rows)
    lines = [
        f"{len(rows)} sequences, {present} frames with the target in the picture",
        f"  identity track on the target {sum(r['identityOn'] for r in rows) / max(1, present):.1%}"
        f"; recorded tracker IoU >= 0.5 {sum(r['trackerRight'] for r in rows) / max(1, present):.1%}",  # noqa: E501
        f"  on the {sum(r['trackerWrong'] for r in rows)} frames the tracker has wrong: identity"
        f" track on the target {sum(r['identityOnTrackerWrong'] for r in rows) / max(1, sum(r['trackerWrong'] for r in rows)):.1%}",  # noqa: E501
    ]
    onsets = [o for r in rows for o in r["onsets"]]
    for k, label in enumerate(("at the onset", "10 frames later", "30 frames later")):
        known = [o[k] for o in onsets if o[k] is not None]
        if known:
            lines.append(
                f"  {len(onsets)} onsets, {label}: identity on the target {np.mean(known):.1%} (n={len(known)})"  # noqa: E501
            )
    lines.append(
        f"  identity lost {sum(r['switches'] for r in rows)} times;"
        f" median ms per frame {np.median([r['ms'] for r in rows]):.1f}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument(
        "--sequence-file", type=Path, default=Path("configs/splits/360vos_dev_train.txt")
    )
    parser.add_argument("--sequences", default=None)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "second_look.yaml",
    )
    parser.add_argument(
        "--yoloe",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "yoloe" / "yoloe-11s-seg.pt",
    )
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=10)
    parser.add_argument("--gate", type=float, default=1.5)
    parser.add_argument("--max-cost", type=float, default=0.8)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--debug", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--rows", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    if args.rows is not None:
        rows = json.loads(args.rows.read_text(encoding="utf-8"))
    else:
        config = loadConfig(args.config)
        dataset = Vot360Dataset(args.dataset_root, args.label_root)
        models = AppearanceModels(Path(config.model.weights).parent / "hub", ("dinov2",))
        yoloe = VisualPromptYoloe(args.yoloe)
        results = loadTrackerResults(args.run / "bbox" / "ours")
        names = readSequenceFile(args.sequence_file)
        if args.sequences:
            names = [n for n in names if n in args.sequences.split(",")]
        rows = []
        args.out.mkdir(parents=True, exist_ok=True)
        for name in names:
            recorded = sequenceData(dataset, args.run, name, results)
            if recorded is None:
                continue
            rows.append(runSequence(name, args, dataset, config, models, yoloe, recorded))
            print(f"  {name}: identity {rows[-1]['identityOn']}/{rows[-1]['present']}", flush=True)
            (args.out / "rows.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    text = report(rows)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
