"""A frame-0-only look as a drift check during trusted tracking (E067, offline).

    python tools/drift_check_study.py --dataset-root <dir> --label-root <dir> \\
        --runs outputs/E062b_s1 outputs/E062b_s2 --step 3 --out outputs/E067

On frames a recorded run entered trusted (every ``--step``-th, target present or not):
the frame's tracking view is rebuilt from the run's own boxes (centred on the box of the
frame before, the seven boxes before as the trajectory) and one stateless pass is made,
which sees the frame-0 template only, not the tracker's appearance memory.  Its box is
compared with the box the tracker recorded.  The two disagreeing is the drift signal.

Reported per agreement threshold: how often the check fires on frames the tracker has
right (IoU >= 0.5), on frames it has wrong (target absent or IoU < 0.1) and on the first
ten frames of a wrong stretch, the share of firing frames that are wrong, and the same
share for the state machine's relative signal flagging as many frames.  Then a switch
rule (fires, stateless score >= 0.70, its DINOv2 similarity above the tracker's): how
often the stateless box is on the target and how often the tracker was right.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from math import pi, radians
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from size_study import sequenceData  # noqa: E402

from track360.backends import createArtrackSession  # noqa: E402
from track360.core.config import loadConfig  # noqa: E402
from track360.core.types import BFoV, FrameIndex, FramePacket, SequenceId  # noqa: E402
from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.loss_rate import dualIou  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402
from track360.geometry import makeSphericalPoint  # noqa: E402
from track360.runtime.benchmark import _SharedSession  # noqa: E402
from track360.runtime.driver import buildRuntime, closeRuntime  # noqa: E402

TRAJECTORY = 7
AGREE = (0.1, 0.3, 0.5)
TAKE = 0.70


def _bfov(row: np.ndarray) -> BFoV:
    # Recorded sizes can reach 180 deg (a box covering a pole), outside a BFoV's range.
    def side(deg: float) -> float:
        return min(max(radians(deg), 1e-4), pi - 1e-3)

    return BFoV(makeSphericalPoint(radians(row[0]), radians(row[1])), side(row[2]), side(row[3]))


def replay(args) -> list[dict]:
    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    session = createArtrackSession(config)
    rows: list[dict] = []
    try:
        for root in args.runs:
            results = loadTrackerResults(root / "bbox" / "ours")
            for name in readSequenceFile(args.sequence_file):
                d = sequenceData(dataset, root, name, results)
                if d is None:
                    continue
                with (root / "trace" / "ours" / f"{name}.csv").open(encoding="utf-8") as stream:
                    trace = {int(r["frame"]): r for r in csv.DictReader(stream)}
                boxes = np.loadtxt(root / "bfov" / "ours" / f"{name}.txt", delimiter=",", ndmin=2)
                frames = [
                    t
                    for t in range(2, len(d["present"]), args.step)
                    if t in trace
                    and trace[t]["modeBefore"] == "TRACKING"
                    and trace[t]["action"] != "jump"
                    and t < len(boxes)
                ]
                if not frames:
                    continue
                sequence = dataset.sequence(name)
                truth, _ = sequence.groundTruth("bbox")
                width, height = sequence.frameSize
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
                    runtime.verifier.setTemplate(template, plan.templateBox)
                    planner = runtime.controller._planner
                    for t in frames:
                        previous = _bfov(boxes[t - 1])
                        history = tuple(_bfov(boxes[max(0, k)]) for k in range(t - TRAJECTORY, t))
                        spec = planner.searchView(
                            previous.center,
                            previous.horizontalFovRad,
                            previous.verticalFovRad,
                            history,
                        )
                        frame = FramePacket(SequenceId(name), FrameIndex(t), t, sequence.readRgb(t))
                        view = runtime.geometry.cropViews(frame, [spec])[0]
                        found = runtime.backend.inferDetached((view,))
                        row = {
                            "run": root.name,
                            "sequence": name,
                            "frame": t,
                            "present": bool(d["present"][t]),
                            "iouT": float(d["iou"][t]),
                            "scoreT": float(trace[t]["backend"] or 0.0),
                            "simT": float(trace[t]["appearance"] or 0.0),
                            "relative": float(trace[t]["relative"] or 0.0),
                            "modeAfter": trace[t]["modeAfter"],
                            "found": False,
                        }
                        if found:
                            try:
                                projected = runtime.geometry.projectLocalBoxBoundary(
                                    found[0].bbox, spec, width, height
                                )
                            except Exception:
                                projected = None
                            if projected is not None:
                                b = projected.bbox
                                box = np.asarray([[b.xPx, b.yPx, b.widthPx, b.heightPx]])
                                row.update(
                                    found=True,
                                    agree=float(dualIou(results[name][t : t + 1], box, width)[0]),
                                    iouD=float(dualIou(truth[t : t + 1], box, width)[0])
                                    if d["present"][t]
                                    else 0.0,
                                    scoreD=float(np.clip(found[0].fusedScore, 0, 1)),
                                    simD=float(runtime.verifier.similarity(view, found[0].bbox)),
                                )
                        rows.append(row)
                        release = getattr(runtime.geometry, "releaseFrame", None)
                        if callable(release):
                            release()
                finally:
                    closeRuntime(runtime)
                    sequence.close()
                print(f"  {root.name} {name}: {len(frames)} frames", flush=True)
    finally:
        session.close()
    # Onsets: wrong frames with a right frame of the same run and sequence 1-10 frames before.
    right = {(r["run"], r["sequence"], r["frame"]) for r in rows if _right(r)}
    for r in rows:
        r["onset"] = _wrong(r) and any(
            (r["run"], r["sequence"], r["frame"] - k) in right for k in range(1, 11)
        )
    return rows


def _right(r: dict) -> bool:
    return r["present"] and r["iouT"] >= 0.5


def _wrong(r: dict) -> bool:
    return (not r["present"]) or r["iouT"] < 0.1


def report(rows: list[dict]) -> str:
    right = [r for r in rows if _right(r)]
    wrong = [r for r in rows if _wrong(r)]
    onset = [r for r in rows if r["onset"]]
    lines = [
        f"{len(rows)} frames entered trusted: tracker right {len(right)}, wrong {len(wrong)}"
        f" ({len(wrong) / max(1, len(rows)):.1%}; absent {sum(not r['present'] for r in wrong)}),"
        f" onset {len(onset)}",
        "  agree<  | fires: right  wrong  onset | wrong among fired | relative signal, same count",
    ]

    def fires(r: dict, a: float) -> bool:
        return (not r["found"]) or r["agree"] < a

    order = sorted(rows, key=lambda r: r["relative"])
    for a in AGREE:
        fired = [r for r in rows if fires(r, a)]
        n = len(fired)
        lowest = order[:n]
        lines.append(
            f"  {a:.1f}     | {np.mean([fires(r, a) for r in right]):6.1%}"
            f" {np.mean([fires(r, a) for r in wrong]):6.1%}"
            f" {np.mean([fires(r, a) for r in onset]) if onset else float('nan'):6.1%}"
            f" | {np.mean([_wrong(r) for r in fired]) if fired else float('nan'):6.1%} of {n}"
            f" | {np.mean([_wrong(r) for r in lowest]) if lowest else float('nan'):6.1%}"
        )
    lines.append(
        "  switch (fires, stateless score >= 0.70, similarity above the tracker's):"
        " n | stateless on target | tracker was right | tracker wrong and stateless on target"
    )
    for a in AGREE:
        switched = [
            r
            for r in rows
            if r["found"] and r["agree"] < a and r["scoreD"] >= TAKE and r["simD"] > r["simT"]
        ]
        if not switched:
            continue
        lines.append(
            f"  {a:.1f}  n={len(switched):>5} | {np.mean([r['iouD'] >= 0.5 for r in switched]):6.1%}"  # noqa: E501
            f" | {np.mean([_right(r) for r in switched]):6.1%}"  # noqa: E501
            f" | {np.mean([_wrong(r) and r['iouD'] >= 0.5 for r in switched]):6.1%}"  # noqa: E501
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
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "camera.yaml",
    )
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--rows", type=Path, default=None, help="report on saved rows only")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    if args.rows is not None:
        rows = json.loads(args.rows.read_text(encoding="utf-8"))
    else:
        rows = replay(args)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "rows.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    text = report(rows)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
