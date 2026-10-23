"""Small targets: where they get lost, and would another view size keep them (E055).

    python tools/size_study.py onsets --dataset-root <dir> --label-root <dir> \\
        --runs outputs/E053d_cam_s1 ... --out outputs/E055/onsets
    python tools/size_study.py views --dataset-root <dir> --label-root <dir> \\
        --run outputs/E053d_cam_s1 --max-size-deg 3 --out outputs/E055/views

``onsets``: every stretch of lost frames that began right after good tracking (mean IoU
of the five frames before >= 0.5), by the target's stable size (the median of its size
over the ten frames before; frame-0 size for the first ten frames).  Reported per size
band: onsets per 1000 good frames, and the share whose target was already outside the
normal view (half width two target sizes) centred on the tracker's last box.

``views``: offline, on frames of a recorded run where the tracker was on the target the
frame before and the stable size is below ``--max-size-deg``: a stateless pass on a view
centred on the tracker's previous box, the view this many times the target size (4 is
the tracker's), and the share of frames with the target in the view, boxed (IoU >= 0.5),
and the mean IoU.  The onset frames, where the recorded run lost the target, are
reported separately.  Stateless passes see only the frame-0 template, so the numbers
compare view sizes, not the tracker itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from math import radians
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.loss_rate import dualIou, lostFrameMask  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402

BANDS = ((0, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 4), (4, 6), (6, 10), (10, 400))
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "camera.yaml"


def unit(yaw, pitch) -> np.ndarray:
    yaw, pitch = np.radians(np.asarray(yaw, float)), np.radians(np.asarray(pitch, float))
    return np.stack([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)], -1)


def angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0)))


def sequenceData(dataset, root: Path, name: str, results) -> dict | None:
    path = root / "trace" / "ours" / f"{name}.csv"
    if not path.exists() or name not in results:
        return None
    sequence = dataset.sequence(name)
    box, present = sequence.groundTruth("bbox")
    bfov, _ = sequence.groundTruth("bfov")
    width = sequence.frameSize[0]
    sequence.close()
    present = np.asarray(present, bool)
    n = len(present)
    iou = np.zeros(n)
    iou[present] = dualIou(box[present], results[name][present], width)
    size = np.sqrt(np.maximum(bfov[:, 2] * bfov[:, 3], 1e-6))
    stable = np.full(n, size[0])
    for t in range(10, n):
        window = size[t - 10 : t][present[t - 10 : t]]
        stable[t] = np.median(window) if window.size else size[0]
    tracker = np.full((n, 3), np.nan)
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            f = int(row["frame"])
            if f < n:
                tracker[f] = unit(row["yawDeg"], row["pitchDeg"])
    tracker[0] = unit(bfov[0, 0], bfov[0, 1])
    return {
        "present": present,
        "iou": iou,
        "lost": lostFrameMask(iou, present),
        "stable": stable,
        "truth": unit(bfov[:, 0], bfov[:, 1]),
        "tracker": tracker,
    }


def onsets(args: argparse.Namespace) -> None:
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    names = readSequenceFile(args.sequence_file)
    table = {band: [0, 0, 0] for band in BANDS}  # good frames, onsets, escaped
    for root in args.runs:
        results = loadTrackerResults(root / "bbox" / "ours")
        for name in names:
            d = sequenceData(dataset, root, name, results)
            if d is None:
                continue
            good = d["present"] & (d["iou"] >= 0.5)
            for t in range(6, len(good)):
                band = next(b for b in BANDS if b[0] <= d["stable"][t] < b[1])
                if good[t - 1]:
                    table[band][0] += 1
                if d["lost"][t] and not d["lost"][t - 1] and d["iou"][t - 5 : t].mean() >= 0.5:
                    table[band][1] += 1
                    half = 2.0 * d["stable"][t]
                    if angle(d["truth"][t], d["tracker"][t - 1]) > half:
                        table[band][2] += 1
    lines = ["stable size band   good frames  onsets  per 1000  escaped the normal view"]
    for band, (good, count, escaped) in table.items():
        lines.append(
            f"  {band[0]:>4}-{band[1]:<5}deg {good:>10} {count:>7} {1000 * count / max(1, good):>8.2f}"
            f"   {escaped / max(1, count):6.1%}"
        )
    _write(args.out, "\n".join(lines), {str(k): v for k, v in table.items()})


def views(args: argparse.Namespace) -> None:
    from track360.backends import createArtrackSession
    from track360.core.config import loadConfig
    from track360.core.types import FrameIndex, FramePacket, SequenceId
    from track360.geometry import makeSphericalPoint
    from track360.runtime.benchmark import _SharedSession
    from track360.runtime.driver import buildRuntime, closeRuntime

    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    session = createArtrackSession(config)
    factors = [float(x) for x in args.factors.split(",")]
    rows = []
    results = loadTrackerResults(args.run / "bbox" / "ours")
    try:
        for name in readSequenceFile(args.sequence_file):
            d = sequenceData(dataset, args.run, name, results)
            if d is None:
                continue
            good = d["present"] & (d["iou"] >= 0.5)
            frames = [
                t
                for t in range(6, len(good))
                if good[t - 1] and d["present"][t] and d["stable"][t] < args.max_size_deg
            ]
            onsetFrames = {
                t
                for t in frames
                if d["lost"][t] and not d["lost"][t - 1] and d["iou"][t - 5 : t].mean() >= 0.5
            }
            picked = sorted(set(frames[:: args.step]) | onsetFrames)
            if not picked:
                continue
            sequence = dataset.sequence(name)
            box, _ = sequence.groundTruth("bbox")
            width = sequence.frameSize[0]
            runtime = buildRuntime(config, artrackSessionFactory=lambda _: _SharedSession(session))
            try:
                first = FramePacket(SequenceId(name), FrameIndex(0), 0, sequence.readRgb(0))
                plan = runtime.controller.buildInitialization(
                    first, initialBfov=sequence.initialBfov()
                )
                template = runtime.geometry.cropViews(first, [plan.templateView])[0]
                runtime.backend.initialize(template, plan.templateBox)
                planner = runtime.controller._planner
                for t in picked:
                    frame = FramePacket(SequenceId(name), FrameIndex(t), t, sequence.readRgb(t))
                    height = frame.rgb.shape[0]
                    c = d["tracker"][t - 1]
                    centre = makeSphericalPoint(
                        np.arctan2(c[0], c[2]), np.arcsin(np.clip(c[1], -1, 1))
                    )
                    size = radians(d["stable"][t])
                    row = {
                        "sequence": name,
                        "frame": t,
                        "onset": t in onsetFrames,
                        "sizeDeg": d["stable"][t],
                    }
                    for factor in factors:
                        spec = planner.probeView(centre, size, size, factor / 4.0, 0)
                        half = 0.5 * spec.bfov.horizontalFovRad
                        inside = float(angle(d["truth"][t], c)) <= np.degrees(half)
                        view = runtime.geometry.cropViews(frame, [spec])[0]
                        found = runtime.backend.inferDetached((view,))
                        overlap = 0.0
                        if found:
                            try:
                                projected = runtime.geometry.projectLocalBoxBoundary(
                                    found[0].bbox, spec, width, height
                                ).bbox
                                overlap = float(
                                    dualIou(
                                        box[t : t + 1],
                                        np.asarray(
                                            [
                                                [
                                                    projected.xPx,
                                                    projected.yPx,
                                                    projected.widthPx,
                                                    projected.heightPx,
                                                ]
                                            ]
                                        ),
                                        width,
                                    )[0]
                                )
                            except Exception:
                                overlap = 0.0
                        row[f"in{factor:g}"] = inside
                        row[f"iou{factor:g}"] = overlap
                    rows.append(row)
                    release = getattr(runtime.geometry, "releaseFrame", None)
                    if callable(release):
                        release()
            finally:
                closeRuntime(runtime)
                sequence.close()
            print(f"  {name}: {len(picked)} frames", flush=True)
    finally:
        session.close()
    lines = [f"{len(rows)} frames below {args.max_size_deg:g} deg (stateless passes)"]
    for label, part in (
        ("all", rows),
        ("onsets", [r for r in rows if r["onset"]]),
        ("others", [r for r in rows if not r["onset"]]),
    ):
        if not part:
            continue
        cells = []
        for factor in factors:
            ious = np.array([r[f"iou{factor:g}"] for r in part])
            ins = np.mean([r[f"in{factor:g}"] for r in part])
            cells.append(
                f"x{factor:g}: in {ins:.1%} boxed {np.mean(ious >= 0.5):.1%} IoU {ious.mean():.3f}"
            )
        lines.append(f"  {label:<7} n={len(part):>5}  " + " | ".join(cells))
    _write(args.out, "\n".join(lines), rows)


def _write(out: Path, text: str, payload) -> None:
    print(text)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    (out / "data.json").write_text(json.dumps(payload, default=float) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("onsets", "views"))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument(
        "--sequence-file", type=Path, default=Path("configs/splits/360vos_dev_train.txt")
    )
    parser.add_argument("--runs", type=Path, nargs="+")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-size-deg", type=float, default=3.0)
    parser.add_argument("--factors", default="4,5,6,8")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    onsets(args) if args.command == "onsets" else views(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
