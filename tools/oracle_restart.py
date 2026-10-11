"""What would a perfect re-detection be worth?  Restart the lost track on the true target.

    python tools/oracle_restart.py run --nth 1 --dataset-root <dir> --label-root <dir> \\
        --sequence-file configs/splits/360vos_dev_train.txt --config <config> \\
        --output-root outputs/E035_n1
    python tools/oracle_restart.py report --dataset-root <dir> --label-root <dir> \\
        --output-root outputs/E035_n1 --baseline outputs/E015_train

``run`` tracks as ``tools/benchmark.py run --method ours`` does, with one change: at
chosen frames, when the tracked box has an IoU below 0.1 with the ground truth, the
track is put on the true box of that frame, exactly as a jump to a search candidate
would do it (the tracker's memory starts again from the frame-0 template, the
trajectory input becomes that box).  The frames are chosen from the ground truth:

    reappear   the target is back in the picture after being absent: the ``--nth``
               frame it is present again
    jump       the target is at least ``--jump`` target sizes away from where its
               motion put it, after ``--window`` frames without such a jolt: the
               ``--nth`` frame counted from the one it arrived at
    drift      any other loss: the ``--nth`` frame in a row with an IoU below 0.1,
               when no reappearance or jump came in the ``--window`` frames before the
               run began and no restart of any kind in the last ``--cooldown`` frames

A restart is made only when the tracker is off the target at that frame; a tracker that
came through the event by itself is left alone.  This uses the ground truth during
tracking: it is an upper bound for any re-detection, not a result.  The config must have
``lossHandling: true`` and ``lossActions: none``.

``report`` gives, for the frames after each restart, how many are lost again, by kind.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track360.controller import track_controller  # noqa: E402
from track360.core.types import BFoV, ProjectedObservation  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.loss_rate import dualIou, lostFrameMask  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402
from track360.geometry.projection_math import makeSphericalPoint  # noqa: E402

HORIZONS = (10, 30, 100)
KINDS = ("reappear", "jump", "drift")


def _directions(rows: np.ndarray) -> np.ndarray:
    longitude, latitude = np.radians(rows[:, 0]), np.radians(rows[:, 1])
    return np.column_stack(
        [
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
            np.cos(latitude) * np.cos(longitude),
        ]
    )


def events(spheres: np.ndarray, present: np.ndarray, jump: float, calm: int) -> dict[int, str]:
    """Frames at which the target came back or arrived after a jump.

    A jump is a frame where the target is at least ``jump`` target sizes away from
    where its motion over the two frames before put it, after ``calm`` frames without
    one: a target that jitters all the time gives one event, not one per frame.
    """
    direction = _directions(spheres)
    size = np.radians(np.sqrt(np.clip(spheres[:, 2] * spheres[:, 3], 1e-6, None)))
    found: dict[int, str] = {}
    lastJolt = -(10**9)
    for index in range(1, len(present)):
        if not present[index]:
            continue
        if not present[index - 1]:
            found[index] = "reappear"
            continue
        expected = direction[index - 1]
        if index >= 2 and present[index - 2]:
            previous, older = direction[index - 1], direction[index - 2]
            expected = 2.0 * float(previous @ older) * previous - older
            expected = expected / np.linalg.norm(expected)
        cosine = float(np.clip(direction[index] @ expected, -1.0, 1.0))
        if math.acos(cosine) >= jump * size[index - 1]:
            if index - lastJolt > calm:
                found[index] = "jump"
            lastJolt = index
    return found


def install(dataset: Vot360Dataset, args: argparse.Namespace, log: list[dict[str, object]]):
    truths: dict[str, tuple] = {}
    state: dict[str, dict[str, int]] = {}
    impl = track_controller.TrackControllerImpl
    original = impl.consume
    nth = args.nth

    def truth(name: str):
        if name not in truths:
            sequence = dataset.sequence(name)
            boxes, present = sequence.groundTruth("bbox")
            spheres, _ = sequence.groundTruth("bfov")
            present = np.asarray(present, dtype=bool)
            truths[name] = (
                boxes,
                spheres,
                present,
                sequence.frameSize[0],
                events(spheres, present, args.jump, args.window),
            )
            sequence.close()
        return truths[name]

    def consume(self, plan, observation, candidates=()):
        name = str(self._sequenceId)
        boxes, spheres, present, width, marked = truth(name)
        index = int(plan.frameIndex)
        memo = state.setdefault(name, {"bad": 0, "last": -(10**9), "lastEvent": -(10**9)})
        if index in marked:
            memo["lastEvent"] = index
        if not (index < len(present) and present[index]):
            return original(self, plan, observation, candidates)
        iou = 0.0
        if observation is not None:
            box = observation.bbox
            iou = float(
                dualIou(
                    boxes[index : index + 1],
                    np.asarray([[box.xPx, box.yPx, box.widthPx, box.heightPx]]),
                    width,
                )[0]
            )
        memo["bad"] = memo["bad"] + 1 if iou < 0.1 else 0
        kind = None
        origin = index - (nth - 1)
        if iou < 0.1 and origin in marked:
            # Present on every frame from the event to this one.
            if bool(present[origin : index + 1].all()):
                kind = marked[origin]
        if kind is None and memo["bad"] == nth:
            runStart = index - nth + 1
            if runStart - memo["lastEvent"] > args.window and index - memo["last"] > args.cooldown:
                kind = "drift"
        if kind is None:
            return original(self, plan, observation, candidates)
        pending = self._pending
        row = spheres[index]
        bfov = BFoV(
            makeSphericalPoint(math.radians(row[0]), math.radians(row[1])),
            min(max(math.radians(row[2]), 1e-4), 2 * math.pi - 1e-3),
            min(max(math.radians(row[3]), 1e-4), math.pi - 1e-3),
        )
        height = pending.frame.rgb.shape[0]
        candidate = ProjectedObservation(
            viewId=0,
            bfov=bfov,
            bbox=self._geometry.bfovToBbox(bfov, pending.frame.rgb.shape[1], height),
            modelScore=1.0,
            appearanceScore=1.0,
            motionScore=1.0,
            scaleScore=1.0,
            fusedScore=1.0,
            singleScore=1.0,
        )
        evaluation = self._evaluator.evaluate(
            mode=pending.mode,
            plan=plan,
            observation=observation,
            prediction=pending.prediction,
            predictedBfov=pending.predictedBfov,
            geometry=self._geometry,
            frameWidthPx=pending.frame.rgb.shape[1],
            frameHeightPx=height,
        )
        trace = self._traceOf(pending, evaluation)
        self._lastFrameSuspect = False
        result = self._reacquire(pending, candidate)
        trace.update(modeAfter=self._mode.name, reason="JUMP", action="oracle")
        self._lastFrameTrace = trace
        memo["bad"] = 0
        memo["last"] = index
        log.append({"sequence": name, "frame": index, "kind": kind})
        return result

    impl.consume = consume


def run(args: argparse.Namespace, rest: list[str]) -> int:
    import benchmark

    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    log: list[dict[str, object]] = []
    install(dataset, args, log)
    args.output_root.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        path = args.output_root / "oracle_restarts.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("sequence", "frame", "kind"))
            writer.writeheader()
            writer.writerows(log)

    atexit.register(save)
    argv = [
        "run",
        "--dataset-root",
        args.dataset_root,
        "--output-root",
        str(args.output_root),
        "--method",
        "ours",
        "--no-resume",
        *rest,
    ]
    if args.label_root:
        argv += ["--label-root", args.label_root]
    return benchmark.main(argv)


def report(args: argparse.Namespace) -> int:
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    results = loadTrackerResults(args.output_root / "bbox" / "ours")
    reference = (
        loadTrackerResults(args.baseline / "bbox" / "ours") if args.baseline is not None else {}
    )
    with (args.output_root / "oracle_restarts.csv").open(encoding="utf-8") as stream:
        restarts = list(csv.DictReader(stream))
    bySequence: dict[str, list[tuple[int, str]]] = {}
    for row in restarts:
        bySequence.setdefault(row["sequence"], []).append((int(row["frame"]), row["kind"]))
    items = []
    totals = {"present": 0, "lost": 0, "lostBefore": 0, "iou": 0.0, "iouBefore": 0.0}
    perSequence = []
    for name in sorted(results):
        sequence = dataset.sequence(name)
        boxes, present = sequence.groundTruth("bbox")
        spheres, _ = sequence.groundTruth("bfov")
        width = sequence.frameSize[0]
        sequence.close()
        present = np.asarray(present, dtype=bool)
        iou = np.zeros(len(boxes))
        iou[present] = dualIou(boxes[present], results[name][present], width)
        lost = lostFrameMask(iou, present)
        totals["present"] += int(present.sum())
        totals["lost"] += int(lost.sum())
        totals["iou"] += float(iou[present].sum())
        if name in reference:
            before = np.zeros(len(boxes))
            before[present] = dualIou(boxes[present], reference[name][present], width)
            totals["lostBefore"] += int(lostFrameMask(before, present).sum())
            totals["iouBefore"] += float(before[present].sum())
            perSequence.append((name, float(before[present].mean()), float(iou[present].mean())))
        frames = [frame for frame, _ in bySequence.get(name, [])]
        for frame, kind in bySequence.get(name, []):
            later = [other for other in frames if other > frame]
            nextRestart = min(later) if later else len(lost)
            item: dict[str, object] = {
                "sequence": name,
                "frame": frame,
                "kind": kind,
                "sizeDeg": float(np.sqrt(max(spheres[frame, 2] * spheres[frame, 3], 1e-6))),
            }
            for horizon in HORIZONS:
                window = slice(frame + 1, frame + 1 + horizon)
                there = present[window]
                if there.sum() < max(3, horizon // 2):
                    continue
                item[f"bad{horizon}"] = float((iou[window][there] < 0.1).mean())
                item[f"iou{horizon}"] = float(iou[window][there].mean())
                item[f"again{horizon}"] = bool(
                    lost[window][there].any() or nextRestart <= frame + horizon
                )
            after = np.flatnonzero(lost[frame + 1 :])
            held = int(after[0]) if after.size else len(lost) - frame - 1
            item["heldFrames"] = min(held, nextRestart - frame - 1)
            item["lostAgain"] = bool(after.size) or nextRestart < len(lost)
            items.append(item)
    print(
        f"{len(results)} sequences, {totals['present']} frames with the target, "
        f"{len(items)} restarts ({sum(len(v) > 0 for v in bySequence.values())} sequences)"
    )
    print(
        f"mean IoU over the frames with the target: {totals['iou'] / totals['present']:.3f}"
        + (
            f" (without restarts {totals['iouBefore'] / totals['present']:.3f})"
            if reference
            else ""
        )
    )
    print(
        f"lost frames: {totals['lost']} ({totals['lost'] / totals['present']:.1%})"
        + (
            f" (without restarts {totals['lostBefore']}, "
            f"{totals['lostBefore'] / totals['present']:.1%})"
            if reference
            else ""
        )
    )
    payload: dict[str, object] = {"totals": totals, "restarts": len(items)}

    def table(title: str, chosen: list[dict[str, object]]) -> None:
        if not chosen:
            return
        print(f"\n{title}: {len(chosen)} restarts")
        print(
            f"  {'next frames':<13}{'n':>5}{'frames IoU<0.1':>16}{'mean IoU':>10}"
            f"{'lost again within':>19}"
        )
        payload[title] = {"restarts": len(chosen)}
        for horizon in HORIZONS:
            part = [e for e in chosen if f"bad{horizon}" in e]
            if not part:
                continue
            bad = float(np.mean([e[f"bad{horizon}"] for e in part]))
            mean = float(np.mean([e[f"iou{horizon}"] for e in part]))
            again = float(np.mean([e[f"again{horizon}"] for e in part]))
            payload[title][str(horizon)] = {
                "n": len(part),
                "bad": bad,
                "iou": mean,
                "lostAgain": again,
            }
            print(f"  {horizon:<13}{len(part):>5}{bad:>16.1%}{mean:>10.3f}{again:>19.1%}")
        held = np.asarray([e["heldFrames"] for e in chosen])
        quartiles = np.percentile(held, [25, 50, 75])
        again = sum(bool(e["lostAgain"]) for e in chosen)
        print(
            f"  frames held: median {quartiles[1]:.0f} ({quartiles[0]:.0f}-{quartiles[2]:.0f}); "
            f"lost or restarted again later: {again} of {len(chosen)}"
        )
        payload[title]["heldQuartiles"] = [float(q) for q in quartiles]
        payload[title]["lostAgainLater"] = again

    table("all restarts", items)
    for kind in KINDS:
        table(f"kind {kind}", [e for e in items if e["kind"] == kind])
    for low, high in ((0, 3), (3, 6), (6, 12), (12, 400)):
        table(
            f"target size {low} to {high} degrees",
            [e for e in items if low <= float(e["sizeDeg"]) < high],
        )
    if perSequence:
        print("\nmean IoU per sequence, without and with restarts (changes above 0.02)")
        for name, before, after in sorted(perSequence, key=lambda row: row[2] - row[1]):
            if abs(after - before) > 0.02:
                count = len(bySequence.get(name, []))
                print(
                    f"  {name}  {before:.3f} -> {after:.3f}  "
                    f"({after - before:+.3f}, {count} restarts)"
                )
        payload["perSequence"] = perSequence
    (args.output_root / "oracle_report.json").write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--nth", type=int, default=1, help="1: the first frame")
    parser.add_argument("--jump", type=float, default=1.0, help="target sizes in one frame")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--cooldown", type=int, default=30)
    args, rest = parser.parse_known_args(argv)
    warnings.filterwarnings("ignore")
    if args.command == "run":
        return run(args, rest)
    return report(args)


if __name__ == "__main__":
    raise SystemExit(main())
