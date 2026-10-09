"""What would a perfect re-detection be worth?  Restart the lost track on the true target.

    python tools/oracle_restart.py run --delay 1 --dataset-root <dir> --label-root <dir> \\
        --sequence-file configs/splits/360vos_dev_train.txt --config <config> \\
        --output-root outputs/E035_k1
    python tools/oracle_restart.py report --dataset-root <dir> --label-root <dir> \\
        --output-root outputs/E035_k1

``run`` tracks as ``tools/benchmark.py run --method ours`` does, with one change: when the
tracked box has had an IoU below 0.1 with the ground truth for ``--delay`` frames in a
row (target present), the track is put on the true box of that frame, exactly as a jump
to a search candidate would do it (the tracker's memory starts again from the frame-0
template, the trajectory input becomes that box).  After a restart none is made for
``--cooldown`` frames, so the frames that follow show what the tracker does on its own.
This uses the ground truth during tracking: it is an upper bound for any re-detection,
not a result.  The config must have ``lossHandling: true`` and ``lossActions: none``.

``report`` reads the restarts back and gives, for the frames after each one, how many
are lost again.
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


def install(dataset: Vot360Dataset, delay: int, cooldown: int, log: list[dict[str, object]]):
    truths: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    state: dict[str, dict[str, int]] = {}
    impl = track_controller.TrackControllerImpl
    original = impl.consume

    def truth(name: str):
        if name not in truths:
            sequence = dataset.sequence(name)
            boxes, present = sequence.groundTruth("bbox")
            spheres, _ = sequence.groundTruth("bfov")
            truths[name] = (boxes, spheres, np.asarray(present, dtype=bool), sequence.frameSize[0])
            sequence.close()
        return truths[name]

    def consume(self, plan, observation, candidates=()):
        boxes, spheres, present, width = truth(str(self._sequenceId))
        index = int(plan.frameIndex)
        memo = state.setdefault(str(self._sequenceId), {"bad": 0, "last": -(10**9)})
        if index < len(present) and present[index]:
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
            if memo["bad"] >= delay and index - memo["last"] > cooldown:
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
                memo.update(bad=0, last=index)
                log.append({"sequence": str(self._sequenceId), "frame": index})
                return result
        return original(self, plan, observation, candidates)

    impl.consume = consume


def run(args: argparse.Namespace, rest: list[str]) -> int:
    import benchmark

    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    log: list[dict[str, object]] = []
    install(dataset, args.delay, args.cooldown, log)
    args.output_root.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        path = args.output_root / "oracle_restarts.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("sequence", "frame"))
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
    with (args.output_root / "oracle_restarts.csv").open(encoding="utf-8") as stream:
        restarts = list(csv.DictReader(stream))
    bySequence: dict[str, list[int]] = {}
    for row in restarts:
        bySequence.setdefault(row["sequence"], []).append(int(row["frame"]))
    events = []
    totals = {"frames": 0, "present": 0, "lost": 0}
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
        totals["frames"] += len(lost)
        totals["present"] += int(present.sum())
        totals["lost"] += int(lost.sum())
        for frame in bySequence.get(name, []):
            item: dict[str, object] = {
                "sequence": name,
                "frame": frame,
                "sizeDeg": float(np.sqrt(max(spheres[frame, 2] * spheres[frame, 3], 1e-6))),
            }
            for horizon in HORIZONS:
                window = slice(frame + 1, frame + 1 + horizon)
                there = present[window]
                if there.sum() < horizon // 2:
                    continue
                item[f"bad{horizon}"] = float((iou[window][there] < 0.1).mean())
                item[f"iou{horizon}"] = float(iou[window][there].mean())
                item[f"lostAt{horizon}"] = bool(lost[window][there][-1])
            after = np.flatnonzero(lost[frame + 1 :])
            item["heldFrames"] = int(after[0]) if after.size else int(len(lost) - frame - 1)
            item["lostAgain"] = bool(after.size)
            events.append(item)
    print(
        f"{len(results)} sequences, {totals['present']} frames with the target; "
        f"{len(events)} restarts; lost frames now {totals['lost']} "
        f"({totals['lost'] / totals['present']:.1%} of the frames with the target)"
    )
    payload: dict[str, object] = {"totals": totals, "restarts": len(events)}

    def table(title: str, chosen: list[dict[str, object]]) -> None:
        print(f"\n{title}: {len(chosen)} restarts")
        print(
            f"  {'next frames':<13}{'n':>5}{'frames IoU<0.1':>16}{'mean IoU':>10}"
            f"{'lost at the end':>17}"
        )
        payload[title] = {"restarts": len(chosen)}
        for horizon in HORIZONS:
            part = [e for e in chosen if f"bad{horizon}" in e]
            if not part:
                continue
            bad = float(np.mean([e[f"bad{horizon}"] for e in part]))
            mean = float(np.mean([e[f"iou{horizon}"] for e in part]))
            end = float(np.mean([e[f"lostAt{horizon}"] for e in part]))
            payload[title][str(horizon)] = {
                "n": len(part),
                "bad": bad,
                "iou": mean,
                "lostAtEnd": end,
            }
            print(f"  {horizon:<13}{len(part):>5}{bad:>16.1%}{mean:>10.3f}{end:>17.1%}")
        held = np.asarray([e["heldFrames"] for e in chosen if e["lostAgain"]])
        if held.size:
            quartiles = np.percentile(held, [25, 50, 75])
            print(
                f"  lost again later: {held.size} of {len(chosen)}; frames held until then: "
                f"median {quartiles[1]:.0f} ({quartiles[0]:.0f}-{quartiles[2]:.0f}); "
                f"within 10 frames: {(held < 10).sum()}, within 30: {(held < 30).sum()}"
            )
            payload[title]["lostAgain"] = int(held.size)
            payload[title]["heldQuartiles"] = [float(q) for q in quartiles]
            payload[title]["within10"] = int((held < 10).sum())
            payload[title]["within30"] = int((held < 30).sum())

    table("all restarts", events)
    for low, high in ((0, 3), (3, 6), (6, 12), (12, 400)):
        table(
            f"target size {low} to {high} degrees",
            [e for e in events if low <= float(e["sizeDeg"]) < high],
        )
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
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--cooldown", type=int, default=100)
    args, rest = parser.parse_known_args(argv)
    warnings.filterwarnings("ignore")
    if args.command == "run":
        return run(args, rest)
    return report(args)


if __name__ == "__main__":
    raise SystemExit(main())
