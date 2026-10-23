"""Replay enlarged searches: zoom, first normal look, second look centred on the box (E055, E056).

    python tools/refine_study.py --dataset-root <dir> --label-root <dir> \\
        --runs outputs/E053d_cam_s1 outputs/E053d_cam_s2 --out outputs/E056/replay

Offline, on the enlarged searches of recorded baseline-3 runs with the target in the
picture: the zoom view the run took (its recorded centre and scale), then the normal
view around what it found (the first look, as the tracker does), then a normal view
centred on the first look's box (the second look).  Every pass is stateless and sees the
frame-0 template.  For each look: the box's IoU with the truth, its score, and how far
the first box lies from its view's centre (share of the view's half width).  Searches of
targets below ``--small-deg`` are also replayed with the other zoom scale (2x <-> 4x).
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

from track360.backends import createArtrackSession  # noqa: E402
from track360.core.config import loadConfig  # noqa: E402
from track360.core.types import FrameIndex, FramePacket, SequenceId  # noqa: E402
from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.loss_rate import dualIou  # noqa: E402
from track360.geometry import makeSphericalPoint  # noqa: E402
from track360.runtime.benchmark import _SharedSession  # noqa: E402
from track360.runtime.driver import buildRuntime, closeRuntime  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "camera.yaml"


def point(yawDeg: float, pitchDeg: float):
    return makeSphericalPoint(radians(yawDeg), radians(pitchDeg))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument(
        "--sequence-file", type=Path, default=Path("configs/splits/360vos_dev_train.txt")
    )
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--small-deg", type=float, default=4.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    session = createArtrackSession(config)
    rows = []
    try:
        for root in args.runs:
            for name in readSequenceFile(args.sequence_file):
                path = root / "trace" / "ours" / f"{name}.csv"
                if not path.exists():
                    continue
                with path.open(encoding="utf-8") as stream:
                    trace = list(csv.DictReader(stream))
                searches = []
                trusted = None
                for row in trace:
                    if (
                        row["scan"] in ("2x", "4x")
                        and row.get("scanYawDeg")
                        and trusted is not None
                    ):
                        searches.append((row, trusted))
                    if row["modeAfter"] == "TRACKING" and row["action"] != "jump":
                        trusted = row
                if not searches:
                    continue
                sequence = dataset.sequence(name)
                box, present = sequence.groundTruth("bbox")
                present = np.asarray(present, bool)
                width = sequence.frameSize[0]
                runtime = buildRuntime(
                    config, artrackSessionFactory=lambda _: _SharedSession(session)
                )
                try:
                    first = FramePacket(SequenceId(name), FrameIndex(0), 0, sequence.readRgb(0))
                    plan = runtime.controller.buildInitialization(
                        first, initialBfov=sequence.initialBfov()
                    )
                    template = runtime.geometry.cropViews(first, [plan.templateView])[0]
                    runtime.backend.initialize(template, plan.templateBox)
                    planner = runtime.controller._planner
                    geometry = runtime.geometry

                    def look(frame, spec):
                        """Stateless pass on a view: (projected bfov, ERP box, score) or None."""
                        view = geometry.cropViews(frame, [spec])[0]
                        found = runtime.backend.inferDetached((view,))
                        if not found:
                            return None
                        try:
                            projected = geometry.projectLocalBoxBoundary(
                                found[0].bbox, spec, width, frame.rgb.shape[0]
                            )
                        except Exception:
                            return None
                        local = found[0].bbox
                        offset = (
                            sqrt(
                                ((local.xPx + local.widthPx / 2) / spec.outputWidthPx - 0.5) ** 2
                                + ((local.yPx + local.heightPx / 2) / spec.outputHeightPx - 0.5)
                                ** 2
                            )
                            * 2.0
                        )
                        return projected, float(np.clip(found[0].fusedScore, 0, 1)), offset

                    def iouOf(projected, t) -> float:
                        b = projected.bbox
                        return float(
                            dualIou(
                                box[t : t + 1],
                                np.asarray([[b.xPx, b.yPx, b.widthPx, b.heightPx]]),
                                width,
                            )[0]
                        )

                    for row, trustedRow in searches:
                        t = int(row["frame"])
                        if t >= len(present) or not present[t]:
                            continue
                        frame = FramePacket(SequenceId(name), FrameIndex(t), t, sequence.readRgb(t))
                        size = radians(float(trustedRow["sizeDeg"]))
                        centre = point(float(row["scanYawDeg"]), float(row["scanPitchDeg"]))
                        recorded = float(row["scan"][:-1])
                        scales = [recorded]
                        if float(trustedRow["sizeDeg"]) < args.small_deg:
                            scales.append(2.0 if recorded == 4.0 else 4.0)
                        out = {
                            "run": root.name,
                            "sequence": name,
                            "frame": t,
                            "trustedSizeDeg": float(trustedRow["sizeDeg"]),
                            "recordedScale": recorded,
                        }
                        for scale in scales:
                            key = f"{scale:g}x"
                            zoom = planner.probeView(centre, size, size, scale, 1)
                            coarse = look(frame, zoom)
                            if coarse is None:
                                out[key] = None
                                continue
                            fine = planner.probeView(coarse[0].bfov.center, size, size, 1.0, 64)
                            one = look(frame, fine)
                            if one is None:
                                out[key] = None
                                continue
                            entry = {"iou1": iouOf(one[0], t), "score1": one[1], "offset1": one[2]}
                            if scale == recorded:
                                c = one[0].bfov.center
                                again = planner.probeView(c, size, size, 1.0, 65)
                                two = look(frame, again)
                                if two is not None:
                                    entry.update(
                                        iou2=iouOf(two[0], t), score2=two[1], offset2=two[2]
                                    )
                            out[key] = entry
                        rows.append(out)
                        release = getattr(geometry, "releaseFrame", None)
                        if callable(release):
                            release()
                finally:
                    closeRuntime(runtime)
                    sequence.close()
                print(f"  {root.name} {name}: {len(searches)} searches", flush=True)
    finally:
        session.close()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rows.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} searches replayed -> {args.out / 'rows.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
