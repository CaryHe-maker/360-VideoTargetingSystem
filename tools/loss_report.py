"""One report of how a run with loss handling noticed, searched for and took back the target.

    python tools/loss_report.py --dataset-root <dir> --label-root <dir> \\
        --output-root outputs/E030_a_train --tier light --baseline-p50-ms 25 \\
        --json outputs/V21/base2_report.json

Reads the result files and the per-frame state trace of a run and gives, against the
ground truth, the event-level numbers the V2.1 plan is steered by:

- noticing: the share of lost frames the state machine did not trust (detection rate),
  the share of untrusted frames that were lost (precision), and for every stretch of
  lost frames how many frames passed before the first untrusted one;
- place: when a stretch was noticed, how far the place the search starts from (the
  last trusted position) was from the target, in target sizes;
- searching: of the enlarged-view searches made while the target was in the picture,
  how many had the target in the view, returned a box on it, and ended in a jump that
  held;
- jumps: every jump sorted by the IoU before and after it;
- cost: forwards per frame over the run and per sequence, the median latency, and
  whether they are inside the limits of a cost tier.

A lost frame is one of at least five consecutive frames with IoU below 0.1 while the
target is in the picture, as in the loss rate.  The other tools this one builds on
(``state_trace.py``, ``search_outcomes.py``) remain the place for the detailed tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import search_outcomes  # noqa: E402
import state_trace  # noqa: E402

from track360.core.errors import Track360Error  # noqa: E402
from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.loss_rate import dualIou, lostFrameMask  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402

GOOD_IOU, LOST_IOU = 0.5, 0.1
REACH_SIZES = (1.0, 2.0, 4.0, 8.0)
# Cost tiers of the V2.1 plan: mean forwards per frame, the same for the worst
# sequence, and the median latency as a multiple of baseline 1.
TIERS = {"light": (1.3, 1.6, 1.35), "heavy": (2.0, 2.5, 2.1)}
JUMP_CLASSES = ("recovered", "partly", "not recovered", "needless", "harmful", "absent")


def _direction(yawDeg: float, pitchDeg: float) -> np.ndarray:
    yaw, pitch = np.radians(float(yawDeg)), np.radians(float(pitchDeg))
    return np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(first @ second), -1.0, 1.0))))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open index ranges of the consecutive true entries."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2], strict=True)]


def noticing(args: argparse.Namespace) -> dict[str, object]:
    """Detection rate, precision, and delay and place for every stretch of lost frames."""
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    results = loadTrackerResults(args.output_root / "bbox" / args.method)
    lostFrames = lostUntrusted = untrusted = untrustedLost = goodFrames = goodUntrusted = 0
    stretches: list[dict[str, object]] = []
    for name in sorted(results):
        if args.only is not None and name not in args.only:
            continue
        path = args.output_root / "trace" / args.method / f"{name}.csv"
        if not path.exists():
            continue
        sequence = dataset.sequence(name)
        truthBox, present = sequence.groundTruth("bbox")
        truthBfov, _ = sequence.groundTruth("bfov")
        width = sequence.frameSize[0]
        sequence.close()
        count = len(results[name])
        present = np.asarray(present, dtype=bool)[:count]
        truthBox, truthBfov = truthBox[:count], truthBfov[:count]
        iou = np.zeros(count)
        iou[present] = dualIou(truthBox[present], results[name][present], width)
        lost = lostFrameMask(iou, present)
        with path.open(encoding="utf-8") as stream:
            rows = {int(row["frame"]): row for row in csv.DictReader(stream)}
        # Frame 0 is the initialisation: it has no trace row and counts as trusted.
        doubted = np.zeros(count, dtype=bool)
        for index, row in rows.items():
            if index < count:
                doubted[index] = row["modeAfter"] != "TRACKING"
        lostFrames += int(lost.sum())
        lostUntrusted += int((lost & doubted).sum())
        untrusted += int((doubted & present).sum())
        untrustedLost += int((doubted & lost).sum())
        good = present & (iou >= GOOD_IOU)
        goodFrames += int(good.sum())
        goodUntrusted += int((good & doubted).sum())
        for start, end in _runs(lost):
            hits = np.flatnonzero(doubted[start:end])
            item: dict[str, object] = {
                "sequence": name,
                "start": start,
                "length": end - start,
                "noticedAfter": int(hits[0]) if hits.size else None,
            }
            if hits.size:
                noticed = start + int(hits[0])
                trusted = next(
                    (
                        rows[index]
                        for index in range(noticed - 1, 0, -1)
                        if index in rows
                        and rows[index]["modeAfter"] == "TRACKING"
                        and rows[index]["action"] != "jump"
                    ),
                    None,
                )
                if trusted is not None:
                    centre = _direction(trusted["yawDeg"], trusted["pitchDeg"])
                    for label, index in (("AtStart", start), ("WhenNoticed", noticed)):
                        size = float(np.sqrt(max(truthBfov[index, 2] * truthBfov[index, 3], 1e-6)))
                        truth = _direction(truthBfov[index, 0], truthBfov[index, 1])
                        item[f"reach{label}"] = _angle(centre, truth) / size
            stretches.append(item)
    noticed = [item for item in stretches if item["noticedAfter"] is not None]
    delays = np.asarray([item["noticedAfter"] for item in noticed], dtype=float)

    def reach(key: str) -> dict[str, object]:
        values = np.asarray([item[key] for item in noticed if key in item], dtype=float)
        if not values.size:
            return {"count": 0}
        return {
            "count": int(values.size),
            "median": float(np.median(values)),
            **{f"within{size:g}": float((values <= size).mean()) for size in REACH_SIZES},
        }

    return {
        "lostFrames": lostFrames,
        "detectionRate": lostUntrusted / lostFrames if lostFrames else None,
        "untrustedFrames": untrusted,
        "precision": untrustedLost / untrusted if untrusted else None,
        "goodFramesDoubted": goodUntrusted / goodFrames if goodFrames else None,
        "stretches": len(stretches),
        "stretchesNoticed": len(noticed),
        "delayMedian": float(np.median(delays)) if delays.size else None,
        "noticedWithin10": float((delays <= 10).sum() / len(stretches)) if stretches else None,
        "reachAtStart": reach("reachAtStart"),
        "reachWhenNoticed": reach("reachWhenNoticed"),
        "events": stretches,
    }


def gates(attempts: list[dict[str, object]]) -> dict[str, object]:
    """The three gates of an enlarged-view search made while the target was in the picture."""
    enlarged = [a for a in attempts if a["scan"] != "1x" and a["present"]]
    inView = [a for a in enlarged if float(a.get("reach", 9.0)) <= 1.0]
    onTarget = [a for a in inView if a.get("bestOnTarget")]
    held = [
        a
        for a in onTarget
        if a["jumped"] and a.get("iouAfter") is not None and float(a["iouAfter"]) >= LOST_IOU
    ]

    def share(part: list, whole: list) -> float | None:
        return len(part) / len(whole) if whole else None

    return {
        "searches": len(enlarged),
        "inView": len(inView),
        "onTarget": len(onTarget),
        "heldJump": len(held),
        "gate1": share(inView, enlarged),
        "gate2": share(onTarget, inView),
        "gate3": share(held, onTarget),
        "all": share(held, enlarged),
    }


def jumpClasses(jumps: list[dict[str, object]]) -> dict[str, int]:
    """Jumps by the mean IoU of the 5 frames before and the 10 frames after."""
    counts = dict.fromkeys(JUMP_CLASSES, 0)
    for jump in jumps:
        before, after = jump["iouBefore"], jump["iouAfter"]
        if not jump["present"] or after is None:
            counts["absent"] += 1
        elif before is not None and before >= GOOD_IOU:
            counts["needless" if after >= GOOD_IOU else "harmful"] += 1
        elif after >= GOOD_IOU:
            counts["recovered"] += 1
        elif after >= LOST_IOU:
            counts["partly"] += 1
        else:
            counts["not recovered"] += 1
    return counts


def cost(args: argparse.Namespace) -> dict[str, object]:
    """Forwards per frame and latency from the run reports, against a cost tier."""
    perSequence: dict[str, float] = {}
    frames = forwards = 0
    latencies: list[float] = []
    for path in sorted((args.output_root / "reports" / args.method).glob("*.json")):
        if path.stem == "run" or (args.only is not None and path.stem not in args.only):
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("frames") or report.get("forwards") is None:
            continue
        # The first frame initialises the tracker and has no forward of its own.
        steps = max(1, int(report["frames"]) - 1)
        perSequence[path.stem] = int(report["forwards"]) / steps
        frames += steps
        forwards += int(report["forwards"])
        latencies.append(float(report["p50LatencyMs"]))
    result: dict[str, object] = {
        "forwardsPerFrame": forwards / frames if frames else None,
        "worstSequence": max(perSequence, key=perSequence.get) if perSequence else None,
        "worstForwardsPerFrame": max(perSequence.values()) if perSequence else None,
        "p50LatencyMs": float(np.median(latencies)) if latencies else None,
    }
    if args.tier is not None and frames:
        mean, worst, ratio = TIERS[args.tier]
        checks = {
            "forwardsPerFrame": result["forwardsPerFrame"] <= mean,
            "worstForwardsPerFrame": result["worstForwardsPerFrame"] <= worst,
        }
        if args.baseline_p50_ms is not None:
            checks["p50LatencyMs"] = result["p50LatencyMs"] <= ratio * args.baseline_p50_ms
        result["tier"] = args.tier
        result["limits"] = {
            "forwardsPerFrame": mean,
            "worstForwardsPerFrame": worst,
            "p50LatencyMs": None if args.baseline_p50_ms is None else ratio * args.baseline_p50_ms,
        }
        result["checks"] = checks
        result["withinTier"] = all(checks.values())
    return result


def _percent(value: object) -> str:
    return "   n/a" if value is None else f"{100.0 * float(value):5.1f}%"


def render(payload: dict[str, object]) -> str:
    notice, gate, jump, costs = (
        payload["noticing"],
        payload["gates"],
        payload["jumps"],
        payload["cost"],
    )
    lines = [
        "noticing",
        f"  lost frames {notice['lostFrames']}, detection rate {_percent(notice['detectionRate'])}"
        f", precision {_percent(notice['precision'])}"
        f", good frames doubted {_percent(notice['goodFramesDoubted'])}",
        f"  stretches of lost frames {notice['stretches']}, noticed {notice['stretchesNoticed']}"
        f", median delay {notice['delayMedian']} frames"
        f", noticed within 10 frames {_percent(notice['noticedWithin10'])}",
        "place the search starts from (last trusted position), distance to the target in"
        " target sizes",
    ]
    for key, label in (
        ("reachAtStart", "to where the target was when the stretch began"),
        ("reachWhenNoticed", "to where the target was when it was noticed"),
    ):
        item = notice[key]
        if not item["count"]:
            continue
        within = "  ".join(
            f"<={size:g}: {_percent(item[f'within{size:g}'])}" for size in REACH_SIZES
        )
        lines.append(f"  {label}: n={item['count']} median {item['median']:.2f}  {within}")
    lines += [
        "enlarged-view searches with the target in the picture",
        f"  searches {gate['searches']}; in view {gate['inView']} ({_percent(gate['gate1'])})"
        f"; box on target {gate['onTarget']} ({_percent(gate['gate2'])})"
        f"; jump that held {gate['heldJump']} ({_percent(gate['gate3'])})"
        f"; all three {_percent(gate['all'])}",
        f"jumps {sum(jump.values())}: "
        + ", ".join(f"{name} {jump[name]}" for name in JUMP_CLASSES),
        "cost",
        f"  forwards per frame {costs['forwardsPerFrame']:.3f}"
        f", worst sequence {costs['worstSequence']} {costs['worstForwardsPerFrame']:.3f}"
        f", median P50 latency {costs['p50LatencyMs']:.1f} ms",
    ]
    if "checks" in costs:
        marks = ", ".join(
            f"{name} {'ok' if passed else 'OVER'} (limit {costs['limits'][name]:.3g})"
            for name, passed in costs["checks"].items()
        )
        verdict = "inside" if costs["withinTier"] else "OUTSIDE"
        lines.append(f"  {costs['tier']} tier: {marks} -> {verdict}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", default="ours")
    parser.add_argument("--sequence-file", type=Path, default=None)
    parser.add_argument("--tier", choices=sorted(TIERS), default=None)
    parser.add_argument(
        "--baseline-p50-ms",
        type=float,
        default=None,
        help="median latency of baseline 1 measured in the same session",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    args.only = None if args.sequence_file is None else set(readSequenceFile(args.sequence_file))
    warnings.filterwarnings("ignore")
    try:
        _, summary = state_trace.analyse(args)
        attempts = search_outcomes.analyse(args)
        payload = {
            "outputRoot": str(args.output_root),
            "noticing": noticing(args),
            "gates": gates(attempts),
            "jumps": jumpClasses(summary["jumps"]["events"]),
            "cost": cost(args),
        }
    except Track360Error as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(render(payload))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
