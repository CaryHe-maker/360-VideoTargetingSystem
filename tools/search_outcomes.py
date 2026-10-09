"""Why did a search attempt of a lost track succeed or fail?

    python tools/search_outcomes.py --dataset-root <dir> --label-root <dir> \\
        --output-root outputs/E030_a_train --json outputs/E030/a_search_outcomes.json

Reads the state trace of a run with ``scanMode: zoom`` and, for every frame on which a
search was made (``1x``: the look in place; ``2x`` / ``4x``: an enlarged view followed
by a view of the normal size), works out from the ground truth:

- whether the target was in the picture, and how far it was from the middle of the
  searched view, in half-widths of that view (at most 1: inside it);
- what came back: nothing above the acceptance score, the box the tracker already
  had, a box on the target, or a box somewhere else;
- for the attempts that ended in a jump, whether the next frames were right;
- how other acceptance rules would have chosen among the same boxes: a fixed score,
  or the found box against the tracked box of the same frame (its backend score,
  its similarity to the template, or both).

The middle of an enlarged view is the place the track was last trusted; the look in
place uses the tracker's own view.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from track360.datasets.vot360 import Vot360Dataset
from track360.evaluation.loss_rate import dualIou
from track360.evaluation.vot360_metrics import loadTrackerResults

VIEW_FACTOR, MIN_VIEW_DEG = 4.0, 2.0
AFTER_FRAMES = 10
KINDS = ("1x", "2x", "3x", "4x")


def _direction(yawDeg: float, pitchDeg: float) -> np.ndarray:
    yaw, pitch = np.radians(float(yawDeg)), np.radians(float(pitchDeg))
    return np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(first @ second), -1.0, 1.0))))


def analyse(args: argparse.Namespace) -> list[dict[str, object]]:
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    results = loadTrackerResults(args.output_root / "bbox" / args.method)
    attempts = []
    for path in sorted((args.output_root / "trace" / args.method).glob("*.csv")):
        name = path.stem
        sequence = dataset.sequence(name)
        truthBox, present = sequence.groundTruth("bbox")
        truthBfov, _ = sequence.groundTruth("bfov")
        width = sequence.frameSize[0]
        sequence.close()
        present = np.asarray(present, dtype=bool)
        iou = np.zeros(len(truthBox))
        iou[present] = dualIou(truthBox[present], results[name][present], width)
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        trusted = previous = None
        lostSince = None
        for row in rows:
            index = int(row["frame"])
            if row["modeBefore"] == "LOST" and lostSince is None:
                lostSince = index
            if row["modeBefore"] != "LOST":
                lostSince = None
            if row["scan"] in KINDS and previous is not None:
                scale = float(row["scan"][:-1])
                inPlace = row["scan"] == "1x"
                anchor = previous if inPlace or trusted is None else trusted
                size = max(float((trusted or previous)["sizeDeg"]), 1e-3)
                half = 0.5 * scale * max(MIN_VIEW_DEG, VIEW_FACTOR * size)
                item: dict[str, object] = {
                    "sequence": name,
                    "frame": index,
                    "scan": row["scan"],
                    "lostFrames": 0 if lostSince is None else index - lostSince,
                    "present": bool(present[index]),
                    "targetSizeDeg": float(
                        np.sqrt(max(truthBfov[index, 2] * truthBfov[index, 3], 1e-6))
                    ),
                    "jumped": row["action"] == "jump",
                }
                candidates = json.loads(row["candidates"] or "[]")
                accepted = [c for c in candidates if c["verdict"] in ("accepted", "same_place")]
                best = max(candidates, key=lambda c: c["score"]) if candidates else None
                if present[index]:
                    truth = _direction(truthBfov[index, 0], truthBfov[index, 1])
                    centre = _direction(anchor["yawDeg"], anchor["pitchDeg"])
                    item["reach"] = _angle(truth, centre) / half
                    if best is not None:
                        place = _direction(best["yawDeg"], best["pitchDeg"])
                        item["bestOnTarget"] = (
                            _angle(truth, place)
                            < 0.5 * max(item["targetSizeDeg"], float(best["sizeDeg"]))
                        )
                tracked = _direction(previous["yawDeg"], previous["pitchDeg"])
                if best is not None:
                    place = _direction(best["yawDeg"], best["pitchDeg"])
                    trackedSize = max(float(previous["sizeDeg"]), 1e-3)
                    item["bestOnTrackedBox"] = bool(
                        _angle(tracked, place) < 0.5 * max(trackedSize, float(best["sizeDeg"]))
                        and max(trackedSize, float(best["sizeDeg"]))
                        < 1.5 * min(trackedSize, float(best["sizeDeg"]))
                    )
                    item["bestScore"] = float(best["score"])
                    item["bestSimilarity"] = float(best["similarity"] or 0.0)
                item["trackedScore"] = float(row["backend"] or 0.0) if row["hasBox"] == "1" else 0.0
                item["trackedSimilarity"] = float(row["appearance"] or 0.0)
                item["returned"] = bool(candidates)
                item["accepted"] = bool(accepted)
                if item["jumped"]:
                    after = iou[index : index + AFTER_FRAMES][present[index : index + AFTER_FRAMES]]
                    before = iou[max(0, index - 5) : index][present[max(0, index - 5) : index]]
                    item["iouAfter"] = float(after.mean()) if after.size else None
                    item["iouBefore"] = float(before.mean()) if before.size else None
                attempts.append(item)
            if row["modeAfter"] == "TRACKING" and row["action"] != "jump":
                trusted = row
            previous = row
    return attempts


def report(attempts: list[dict[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    print(
        f"  {'search':<8}{'attempts':>9}{'absent':>8}{'in view':>9}{'near':>7}{'far':>7}"
        f"{'jumps':>7}{'right':>7}"
    )
    for kind in KINDS:
        part = [a for a in attempts if a["scan"] == kind]
        if not part:
            continue
        here = [a for a in part if a["present"]]
        inView = [a for a in here if a["reach"] <= 1.0]
        near = [a for a in here if 1.0 < a["reach"] <= 2.0]
        far = [a for a in here if a["reach"] > 2.0]
        jumps = [a for a in part if a["jumped"]]
        right = [a for a in jumps if (a.get("iouAfter") or 0.0) >= 0.1]
        payload[kind] = {
            "attempts": len(part),
            "absent": len(part) - len(here),
            "inView": len(inView),
            "near": len(near),
            "far": len(far),
            "jumps": len(jumps),
            "right": len(right),
        }
        print(
            f"  {kind:<8}{len(part):>9}{len(part) - len(here):>8}{len(inView):>9}{len(near):>7}"
            f"{len(far):>7}{len(jumps):>7}{len(right):>7}"
        )
    print("\nwhat came back, by where the target was (share of the attempts of that kind)")
    print(
        f"  {'search':<8}{'target':<12}{'n':>6}{'nothing':>9}{'low score':>11}"
        f"{'tracked box':>13}{'on target':>11}{'elsewhere':>11}{'jump right':>12}"
    )
    payload["byPlace"] = {}
    for kind in KINDS:
        for label, test in (
            ("in view", lambda a: a["present"] and a["reach"] <= 1.0),
            ("outside", lambda a: a["present"] and a["reach"] > 1.0),
            ("absent", lambda a: not a["present"]),
        ):
            part = [a for a in attempts if a["scan"] == kind and test(a)]
            if not part:
                continue
            counts: Counter[str] = Counter()
            for a in part:
                if not a["returned"]:
                    counts["nothing"] += 1
                elif not a["accepted"]:
                    counts["low score"] += 1
                elif a.get("bestOnTrackedBox"):
                    counts["tracked box"] += 1
                elif a.get("bestOnTarget"):
                    counts["on target"] += 1
                else:
                    counts["elsewhere"] += 1
            right = sum(1 for a in part if a["jumped"] and (a.get("iouAfter") or 0.0) >= 0.1)
            payload["byPlace"][f"{kind} / {label}"] = {
                "n": len(part), **counts, "jumpRight": right
            }
            cells = "".join(
                f"{counts[key] / len(part):>{width}.1%}"
                for key, width in (
                    ("nothing", 9), ("low score", 11), ("tracked box", 13),
                    ("on target", 11), ("elsewhere", 11),
                )
            )
            print(f"  {kind:<8}{label:<12}{len(part):>6}{cells}{right:>12}")
    episodes: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for a in attempts:
        episodes[(str(a["sequence"]), int(a["frame"]) - int(a["lostFrames"]))].append(a)
    print("\nthe jumps that did not land on the target (IoU after < 0.1), by why")
    reasons: Counter[str] = Counter()
    for a in attempts:
        if not a["jumped"] or (a.get("iouAfter") or 0.0) >= 0.1:
            continue
        if not a["present"]:
            reasons["target not in the picture"] += 1
        elif a.get("bestOnTrackedBox"):
            reasons["came back to the box the tracker had"] += 1
        elif a["reach"] > 1.0:
            reasons["target outside the searched view"] += 1
        elif a.get("bestOnTarget"):
            reasons["landed on the target but lost it within ten frames"] += 1
        else:
            reasons["target in the view, another object taken"] += 1
    for reason, count in reasons.most_common():
        print(f"  {reason:<52}{count:>4}")
    payload["failedJumps"] = dict(reasons)
    payload["rules"] = rules(attempts)
    return payload


RULES = {
    "score >= 0.70 (in use)": lambda a: a["bestScore"] >= 0.70,
    "score >= 0.50": lambda a: a["bestScore"] >= 0.50,
    "score > tracked score": lambda a: a["bestScore"] > a["trackedScore"],
    "similarity > tracked similarity": lambda a: a["bestSimilarity"] > a["trackedSimilarity"],
    "both higher": lambda a: a["bestScore"] > a["trackedScore"]
    and a["bestSimilarity"] > a["trackedSimilarity"],
    "mean of the two higher": lambda a: a["bestScore"] + a["bestSimilarity"]
    > a["trackedScore"] + a["trackedSimilarity"],
    "score higher by 0.1": lambda a: a["bestScore"] > a["trackedScore"] + 0.1,
    "mean of the two higher by 0.1": lambda a: a["bestScore"] + a["bestSimilarity"]
    > a["trackedScore"] + a["trackedSimilarity"] + 0.2,
    "score > tracked and >= 0.50": lambda a: a["bestScore"] > max(a["trackedScore"], 0.50),
}


def rules(attempts: list[dict[str, object]]) -> dict[str, object]:
    """What each acceptance rule would take among the boxes the searches returned.

    Boxes that are the tracked box found again are left out.  The run itself used
    one rule, so the boxes of later frames depend on it: this is a comparison on
    the same recorded boxes, not what a run with another rule would give.
    """
    payload: dict[str, object] = {}
    groups = (("enlarged views (2x, 3x, 4x)", KINDS[1:]), ("look in place (1x)", KINDS[:1]))
    for label, kinds in groups:
        part = [
            a
            for a in attempts
            if a["scan"] in kinds and a["returned"] and not a.get("bestOnTrackedBox")
        ]
        onTarget = [a for a in part if a["present"] and a.get("bestOnTarget")]
        print(
            f"\n{label}: {len(part)} boxes returned that are not the tracked box, "
            f"{len(onTarget)} of them on the target"
        )
        print(f"  {'rule':<34}{'accepted':>9}{'on target':>11}{'precision':>11}{'recall':>9}")
        payload[label] = {}
        for name, rule in RULES.items():
            taken = [a for a in part if rule(a)]
            right = [a for a in taken if a["present"] and a.get("bestOnTarget")]
            precision = len(right) / len(taken) if taken else 0.0
            recall = len(right) / len(onTarget) if onTarget else 0.0
            payload[label][name] = {
                "accepted": len(taken), "onTarget": len(right),
                "precision": precision, "recall": recall,
            }
            print(
                f"  {name:<34}{len(taken):>9}{len(right):>11}{precision:>11.1%}{recall:>9.1%}"
            )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", default="ours")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    attempts = analyse(args)
    print(f"{len(attempts)} search attempts")
    payload = report(attempts)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"summary": payload, "attempts": attempts}, indent=1) + "\n"
        args.json.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
