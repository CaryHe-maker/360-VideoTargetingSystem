"""A second look during normal tracking: does re-centring on the box make it better? (E060)

    python tools/tracking_refine_study.py --dataset-root <dir> --label-root <dir> \\
        --run outputs/E053d_cam_s1 --step 5 --out outputs/E060

Offline, on trusted frames of a recorded baseline-3 run with the target in the picture
(every ``--step``-th): a normal view centred on the tracker's box of the frame before
(about where the tracker looked), a stateless pass (the first look); then a normal view
centred on the first look's box, another stateless pass (the second look).  Stateless
passes see only the frame-0 template, while the tracker uses its memory, so the
recorded IoU of the run is reported next to them for reference.  Reported by how far the
first box lies from its view's centre: IoU, boxed share and score of both looks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from math import radians, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from size_study import sequenceData  # noqa: E402

from track360.backends import createArtrackSession  # noqa: E402
from track360.core.config import loadConfig  # noqa: E402
from track360.core.types import FrameIndex, FramePacket, SequenceId  # noqa: E402
from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.loss_rate import dualIou  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402
from track360.geometry import makeSphericalPoint  # noqa: E402
from track360.runtime.benchmark import _SharedSession  # noqa: E402
from track360.runtime.driver import buildRuntime, closeRuntime  # noqa: E402

BINS = ((0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 9))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument(
        "--sequence-file", type=Path, default=Path("configs/splits/360vos_dev_train.txt")
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "camera.yaml",
    )
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    session = createArtrackSession(config)
    results = loadTrackerResults(args.run / "bbox" / "ours")
    rows = []
    try:
        for name in readSequenceFile(args.sequence_file):
            d = sequenceData(dataset, args.run, name, results)
            if d is None:
                continue
            with (args.run / "trace" / "ours" / f"{name}.csv").open(encoding="utf-8") as stream:
                trace = {int(r["frame"]): r for r in csv.DictReader(stream)}
            frames = [
                t
                for t in range(2, len(d["present"]), args.step)
                if d["present"][t]
                and t in trace
                and t - 1 in trace
                and trace[t]["modeAfter"] == "TRACKING"
                and trace[t]["action"] != "jump"
            ]
            if not frames:
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
                geometry = runtime.geometry

                def look(frame, centre, size):
                    spec = planner.probeView(centre, size, size, 1.0, 0)
                    view = geometry.cropViews(frame, [spec])[0]
                    found = runtime.backend.inferDetached((view,))
                    if not found:
                        return None
                    local = found[0].bbox
                    try:
                        projected = geometry.projectLocalBoxBoundary(
                            local, spec, width, frame.rgb.shape[0]
                        )
                    except Exception:
                        return None
                    b = projected.bbox
                    iou = float(
                        dualIou(
                            box[t : t + 1],
                            np.asarray([[b.xPx, b.yPx, b.widthPx, b.heightPx]]),
                            width,
                        )[0]
                    )
                    offset = 2.0 * sqrt(
                        ((local.xPx + local.widthPx / 2) / spec.outputWidthPx - 0.5) ** 2
                        + ((local.yPx + local.heightPx / 2) / spec.outputHeightPx - 0.5) ** 2
                    )
                    return (
                        projected.bfov.center,
                        iou,
                        float(np.clip(found[0].fusedScore, 0, 1)),
                        offset,
                    )

                for t in frames:
                    frame = FramePacket(SequenceId(name), FrameIndex(t), t, sequence.readRgb(t))
                    previous = trace[t - 1]
                    size = radians(float(previous["sizeDeg"]))
                    centre = makeSphericalPoint(
                        radians(float(previous["yawDeg"])), radians(float(previous["pitchDeg"]))
                    )
                    one = look(frame, centre, size)
                    if one is None:
                        continue
                    two = look(frame, one[0], size)
                    if two is None:
                        continue
                    rows.append(
                        {
                            "sequence": name,
                            "frame": t,
                            "sizeDeg": float(previous["sizeDeg"]),
                            "recorded": float(d["iou"][t]),
                            "iou1": one[1],
                            "score1": one[2],
                            "offset1": one[3],
                            "iou2": two[1],
                            "score2": two[2],
                            "offset2": two[3],
                        }
                    )
                    release = getattr(geometry, "releaseFrame", None)
                    if callable(release):
                        release()
            finally:
                closeRuntime(runtime)
                sequence.close()
            print(f"  {name}: {len(frames)} frames", flush=True)
    finally:
        session.close()
    lines = [f"{len(rows)} trusted frames (stateless looks; recorded = the run's own box)"]
    lines.append(
        "  first box from centre      n | recorded IoU | first: IoU boxed score | second: IoU boxed score"  # noqa: E501
    )
    for low, high in ((0, 9), *BINS):
        part = [r for r in rows if low <= r["offset1"] < high]
        if not part:
            continue

        def m(key):
            return np.mean([r[key] for r in part])

        def b(key):
            return np.mean([r[key] >= 0.5 for r in part])

        label = "all" if high == 9 and low == 0 else f"{low:.1f}-{high:g}"
        lines.append(
            f"  {label:<12} {len(part):>12} | {m('recorded'):.3f} ({b('recorded'):5.1%}) |"
            f" {m('iou1'):.3f} {b('iou1'):5.1%} {m('score1'):.2f} |"
            f" {m('iou2'):.3f} {b('iou2'):5.1%} {m('score2'):.2f}"
        )
    on = [r for r in rows if r["iou1"] >= 0.5]
    if on:
        gain = np.array([r["iou2"] - r["iou1"] for r in on])
        lines.append(
            f"  first box on target (n={len(on)}): IoU {np.mean([r['iou1'] for r in on]):.3f} ->"
            f" {np.mean([r['iou2'] for r in on]):.3f}; second better in {np.mean(gain > 0.01):.1%},"
            f" worse in {np.mean(gain < -0.01):.1%}, lost the target in {np.mean([r['iou2'] < 0.5 for r in on]):.1%}"  # noqa: E501
        )
    text = "\n".join(lines)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.txt").write_text(text + "\n", encoding="utf-8")
    (args.out / "rows.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
