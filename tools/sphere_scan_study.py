"""Would a whole-sphere view find large lost targets better than the zoom? (E053)

    python tools/sphere_scan_study.py --dataset-root <dir> --label-root <dir> \\
        --runs outputs/E043_base2_s1 outputs/E043_base2_s2 --min-size-deg 6 \\
        --out outputs/E053/sphere

Offline, on the recorded enlarged searches of base-2 runs (target in the picture) whose
last trusted target was at least ``--min-size-deg`` large: the search the run made is
compared with a whole-sphere view (``ViewPlanner.sphereView``) followed by the normal
view around what it found, the same two forwards.  Both are stateless passes seeing
only the frame-0 template.  Reported per size band: the share of searches with a box on
the target (centre within half its size, as in the event reports), and with one whose
score reaches the taking threshold 0.70.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from math import radians
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_place_study as sp  # noqa: E402

from track360.backends import createArtrackSession  # noqa: E402
from track360.core.config import loadConfig  # noqa: E402
from track360.core.types import BFoV, FrameIndex, FramePacket, SequenceId  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.geometry import makeSphericalPoint  # noqa: E402
from track360.runtime.benchmark import _SharedSession  # noqa: E402
from track360.runtime.driver import buildRuntime, closeRuntime  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "loss_handling.yaml"
TAKE = 0.70
BANDS = ((6, 10), (10, 15), (15, 25), (25, 400))


def onTarget(bfov: BFoV, truth: np.ndarray, sizeDeg: float) -> bool:
    c = bfov.center
    point = np.array([c.x, c.y, c.z])
    size = max(sizeDeg, float(np.degrees(np.sqrt(bfov.horizontalFovRad * bfov.verticalFovRad))))
    return sp.angle(point, truth) < 0.5 * size


def recordedOutcome(row: dict, truth: np.ndarray, sizeDeg: float) -> tuple[bool, bool]:
    found = json.loads(row["candidates"] or "[]")
    hit = take = False
    for c in found:
        p = sp.unit(float(c["yawDeg"]), float(c["pitchDeg"]))
        if sp.angle(p, truth) < 0.5 * max(sizeDeg, float(c["sizeDeg"])):
            hit = True
            take = take or float(c["score"]) >= TAKE
    return hit, take


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--min-size-deg", type=float, default=6.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.camera = None
    warnings.filterwarnings("ignore")
    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    session = createArtrackSession(config)
    rows = []
    try:
        for root in args.runs:
            items = [
                s
                for s in sp.searches(args, root, dataset)
                if s["trace"].size[s["trusted"]] >= args.min_size_deg
            ]
            bySequence: dict[str, list] = {}
            for s in items:
                bySequence.setdefault(s["sequence"], []).append(s)
            for name, part in bySequence.items():
                sequence = dataset.sequence(name)
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
                    for s in part:
                        index = s["frame"]
                        frame = FramePacket(
                            SequenceId(name), FrameIndex(index), index, sequence.readRgb(index)
                        )
                        height, width = frame.rgb.shape[:2]
                        trace = s["trace"]
                        row = trace.rows[s["trusted"]]
                        sizeRad = radians(float(row["sizeDeg"]))
                        centre = makeSphericalPoint(
                            radians(float(row["yawDeg"])), radians(float(row["pitchDeg"]))
                        )
                        spec = planner.sphereView(centre, sizeRad, sizeRad, 1)
                        view = runtime.geometry.cropViews(frame, [spec])[0]
                        coarse = runtime.backend.inferDetached((view,))
                        hit = take = False
                        if coarse:
                            try:
                                pointed = runtime.geometry.projectLocalBoxBoundary(
                                    coarse[0].bbox, spec, width, height
                                )
                                fine = planner.probeView(
                                    pointed.bfov.center, sizeRad, sizeRad, 1.0, 64
                                )
                                fineView = runtime.geometry.cropViews(frame, [fine])[0]
                                found = runtime.backend.inferDetached((fineView,))
                                if found:
                                    box = runtime.geometry.projectLocalBoxBoundary(
                                        found[0].bbox, fine, width, height
                                    )
                                    hit = onTarget(box.bfov, s["truth"], s["targetSize"])
                                    score = float(np.clip(found[0].fusedScore, 0.0, 1.0))
                                    take = hit and score >= TAKE
                            except Exception:
                                pass
                        recordedHit, recordedTake = recordedOutcome(
                            trace.rows[index], s["truth"], s["targetSize"]
                        )
                        rows.append(
                            {
                                "sequence": name,
                                "frame": index,
                                "trustedSizeDeg": float(row["sizeDeg"]),
                                "targetSizeDeg": s["targetSize"],
                                "sphereHit": hit,
                                "sphereTake": take,
                                "recordedHit": recordedHit,
                                "recordedTake": recordedTake,
                            }
                        )
                        release = getattr(runtime.geometry, "releaseFrame", None)
                        if callable(release):
                            release()
                finally:
                    closeRuntime(runtime)
                    sequence.close()
                print(f"  {root.name} {name}: {len(part)} searches", flush=True)
    finally:
        session.close()
    lines = [f"{len(rows)} enlarged searches whose trusted target was >= {args.min_size_deg:g} deg"]
    lines.append(
        "  band (trusted size)   n    recorded: box on target / taken    sphere: on target / taken"
    )
    for low, high in ((args.min_size_deg, 400.0), *BANDS):
        part = [r for r in rows if low <= r["trustedSizeDeg"] < high]
        if not part:
            continue
        mean = {
            k: np.mean([r[k] for r in part])
            for k in ("recordedHit", "recordedTake", "sphereHit", "sphereTake")
        }
        lines.append(
            f"  {low:>5.0f}-{high:<5.0f}deg {len(part):>6}    {mean['recordedHit']:6.1%}"
            f" / {mean['recordedTake']:6.1%}"
            f"            {mean['sphereHit']:6.1%} / {mean['sphereTake']:6.1%}"
        )
    text = "\n".join(lines)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.txt").write_text(text + "\n", encoding="utf-8")
    (args.out / "rows.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
