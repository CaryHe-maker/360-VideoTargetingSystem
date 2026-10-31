"""Two-frame confirmation of search candidates below the taking score (E058, offline).

    python tools/confirm_study.py next-search --dataset-root <dir> --label-root <dir> \\
        --runs outputs/E053d_cam_s1 ... --camera outputs/E053/camera512 --out outputs/E058/search
    python tools/confirm_study.py next-frame ... --runs outputs/E053d_cam_s1 \
        --out outputs/E058/frame

A candidate whose score is in [low, 0.70) is not taken now but held.  It is taken when
it is seen again where it should be (its place turned with the camera's rotation, within
``--tolerance`` of the larger of the two sizes) with a score of at least ``low``:

    next-search  in the next search of the same loss that returned candidates, within
                 ``--window`` frames (reuses that search: no extra forward);
    next-frame   in a normal view taken on the next frame at the held place (that
                 frame's refinement view moved there: no extra forward in the tracker;
                 offline it is replayed as a stateless pass).

Reported per ``low``: held candidates, confirmed ones, and the share on the target, next
to taking every candidate scoring at least ``low`` at once and to the current rule
(>= 0.70).  "On the target": centre within half the larger size, as the event reports.
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
from size_study import angle, unit  # noqa: E402

from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402

TAKE = 0.70


def held(args, dataset):
    """Every scan row's candidates with what is needed, per run and sequence."""
    for root in args.runs:
        for name in readSequenceFile(args.sequence_file):
            path = root / "trace" / "ours" / f"{name}.csv"
            if not path.exists():
                continue
            sequence = dataset.sequence(name)
            bfov, present = sequence.groundTruth("bfov")
            present = np.asarray(present, bool)
            orientation = np.load(args.camera / f"{name}.npz")["orientation"]
            with path.open(encoding="utf-8") as stream:
                rows = [r for r in csv.DictReader(stream) if r["candidates"] not in ("", "[]")]
            yield root, name, sequence, bfov, present, orientation, rows


def onTarget(c: dict, bfov: np.ndarray, present: np.ndarray, f: int) -> bool:
    if f >= len(present) or not present[f]:
        return False
    size = float(np.sqrt(max(bfov[f, 2] * bfov[f, 3], 1e-6)))
    return float(angle(unit(c["yawDeg"], c["pitchDeg"]), unit(bfov[f, 0], bfov[f, 1]))) < 0.5 * max(
        size, float(c["sizeDeg"])
    )


def nextSearch(args) -> None:
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    lows = [float(x) for x in args.lows.split(",")]
    stats = {
        low: {"held": 0, "confirmed": 0, "confirmedOn": 0, "direct": 0, "directOn": 0}
        for low in lows
    }
    rule = [0, 0]
    for _, name, sequence, bfov, present, orientation, rows in held(args, dataset):
        sequence.close()
        for i, row in enumerate(rows):
            f = int(row["frame"])
            found = json.loads(row["candidates"])
            for c in found:
                score = float(c["score"])
                on = onTarget(c, bfov, present, f)
                if score >= TAKE:
                    rule[0] += 1
                    rule[1] += int(on)
                for low in lows:
                    if score >= low:
                        stats[low]["direct"] += 1
                        stats[low]["directOn"] += int(on)
                    if not low <= score < TAKE or row["action"] == "jump":
                        continue
                    stats[low]["held"] += 1
                    place = unit(c["yawDeg"], c["pitchDeg"])
                    for later in rows[i + 1 :]:
                        g = int(later["frame"])
                        if g - f > args.window:
                            break
                        turned = (orientation[g] @ orientation[f].T) @ place
                        again = [
                            d
                            for d in json.loads(later["candidates"])
                            if float(d["score"]) >= low
                            and float(angle(unit(d["yawDeg"], d["pitchDeg"]), turned))
                            < args.tolerance * max(float(d["sizeDeg"]), float(c["sizeDeg"]))
                        ]
                        if again:
                            stats[low]["confirmed"] += 1
                            stats[low]["confirmedOn"] += int(on)
                        break  # only the next search
    report(args, stats, rule, "next search (no extra forward)")


def nextFrame(args) -> None:
    from track360.backends import createArtrackSession
    from track360.core.config import loadConfig
    from track360.core.types import FrameIndex, FramePacket, SequenceId
    from track360.geometry import makeSphericalPoint
    from track360.runtime.benchmark import _SharedSession
    from track360.runtime.driver import buildRuntime, closeRuntime

    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    session = createArtrackSession(config)
    lows = [float(x) for x in args.lows.split(",")]
    stats = {
        low: {"held": 0, "confirmed": 0, "confirmedOn": 0, "direct": 0, "directOn": 0}
        for low in lows
    }
    rule = [0, 0]
    try:
        for _, name, sequence, bfov, present, orientation, rows in held(args, dataset):
            todo = []
            for row in rows:
                f = int(row["frame"])
                for c in json.loads(row["candidates"]):
                    score = float(c["score"])
                    on = onTarget(c, bfov, present, f)
                    if score >= TAKE:
                        rule[0] += 1
                        rule[1] += int(on)
                    for low in lows:
                        if score >= low:
                            stats[low]["direct"] += 1
                            stats[low]["directOn"] += int(on)
                    if (
                        min(lows) <= score < TAKE
                        and row["action"] != "jump"
                        and f + 1 < len(present)
                    ):
                        todo.append((f, c, score, on))
            if not todo:
                sequence.close()
                continue
            runtime = buildRuntime(config, artrackSessionFactory=lambda _: _SharedSession(session))
            try:
                first = FramePacket(SequenceId(name), FrameIndex(0), 0, sequence.readRgb(0))
                plan = runtime.controller.buildInitialization(
                    first, initialBfov=sequence.initialBfov()
                )
                template = runtime.geometry.cropViews(first, [plan.templateView])[0]
                runtime.backend.initialize(template, plan.templateBox)
                planner = runtime.controller._planner
                for f, c, score, on in todo:
                    g = f + 1
                    turned = (orientation[g] @ orientation[f].T) @ unit(c["yawDeg"], c["pitchDeg"])
                    centre = makeSphericalPoint(
                        np.arctan2(turned[0], turned[2]), np.arcsin(np.clip(turned[1], -1, 1))
                    )
                    size = radians(float(c["sizeDeg"]))
                    spec = planner.probeView(centre, size, size, 1.0, 64)
                    frame = FramePacket(SequenceId(name), FrameIndex(g), g, sequence.readRgb(g))
                    view = runtime.geometry.cropViews(frame, [spec])[0]
                    found = runtime.backend.inferDetached((view,))
                    again = 0.0
                    close = False
                    if found:
                        try:
                            projected = runtime.geometry.projectLocalBoxBoundary(
                                found[0].bbox, spec, frame.rgb.shape[1], frame.rgb.shape[0]
                            ).bfov
                            p = projected.center
                            close = float(
                                angle(np.array([p.x, p.y, p.z]), turned)
                            ) < args.tolerance * float(c["sizeDeg"])
                            again = float(np.clip(found[0].fusedScore, 0, 1))
                        except Exception:
                            pass
                    for low in lows:
                        if low <= score < TAKE:
                            stats[low]["held"] += 1
                            if close and again >= low:
                                stats[low]["confirmed"] += 1
                                stats[low]["confirmedOn"] += int(on)
                    release = getattr(runtime.geometry, "releaseFrame", None)
                    if callable(release):
                        release()
            finally:
                closeRuntime(runtime)
                sequence.close()
            print(f"  {name}: {len(todo)} held", flush=True)
    finally:
        session.close()
    report(args, stats, rule, "next frame at the held place (replayed)")


def report(args, stats, rule, label: str) -> None:
    lines = [
        f"{label}; current rule (>= {TAKE}): {rule[0]} candidates, on target {rule[1] / max(1, rule[0]):.1%}",  # noqa: E501
        "  low   held  confirmed  on target | take all >= low: count  on target",
    ]
    for low, s in stats.items():
        lines.append(
            f"  {low:.2f} {s['held']:>6} {s['confirmed']:>9}  {s['confirmedOn'] / max(1, s['confirmed']):8.1%}"  # noqa: E501
            f" | {s['direct']:>20} {s['directOn'] / max(1, s['direct']):9.1%}"
        )
    text = "\n".join(lines)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.txt").write_text(text + "\n", encoding="utf-8")
    (args.out / "stats.json").write_text(
        json.dumps({"rule": rule, "stats": {str(k): v for k, v in stats.items()}}) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=("next-search", "next-frame"))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument(
        "--sequence-file", type=Path, default=Path("configs/splits/360vos_dev_train.txt")
    )
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--camera", type=Path, default=Path("outputs/E053/camera512"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "camera.yaml",
    )
    parser.add_argument("--lows", default="0.4,0.5,0.6")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    nextSearch(args) if args.mode == "next-search" else nextFrame(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
