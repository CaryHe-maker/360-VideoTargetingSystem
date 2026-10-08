"""The three confidence scores on the lost frames of each kind of loss.

    python tools/loss_cause_scores.py --events outputs/E019/loss_events.csv \\
        --tables outputs/E018/tune_freerun.csv outputs/E018/holdout_freerun.csv

Joins the loss events of ``tools/loss_causes.py`` with the per-frame tables of
``tools/fusion_freerun.py`` (both must come from runs with the same results) and
reports, for the lost frames that follow each kind of event, the backend score, the
appearance score, the motion score and the fused state score.  The good frames of the
same runs are the reference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CAUSES = ("disappeared", "grew", "shrank", "jumped", "other")
OFFSET_SCALE = 0.5
SIZE_SCALE = 0.1
WEIGHTS = (0.40, 0.55, 0.05)
UNCERTAIN_SCORE = 0.60
EARLY_FRAMES = 5


def loadFrames(paths: list[Path]) -> dict[tuple[str, int], dict[str, float]]:
    frames = {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                template = max(0.0, min(1.0, float(row["simTemplate"])))
                recent = max(0.0, min(1.0, float(row["simRecent"])))
                backend = max(0.0, min(1.0, float(row["score"])))
                motion = float(
                    np.exp(-0.5 * (float(row["offset"]) / OFFSET_SCALE) ** 2)
                    * np.exp(-0.5 * (float(row["logScale"]) / SIZE_SCALE) ** 2)
                )
                appearance = max(template, recent)
                frames[(row["sequence"], int(row["frame"]))] = {
                    "present": float(row["present"]),
                    "iou": float(row["iou"]),
                    "lost": float(row["lostRun"]),
                    "backend": backend,
                    "template": template,
                    "appearance": appearance,
                    "motion": motion,
                    "offset": float(row["offset"]),
                    "state": WEIGHTS[0] * backend + WEIGHTS[1] * appearance + WEIGHTS[2] * motion,
                }
    return frames


def summarize(rows: list[dict[str, float]]) -> dict[str, object]:
    if not rows:
        return {"frames": 0}
    result: dict[str, object] = {"frames": len(rows)}
    for key in ("backend", "template", "appearance", "motion", "state", "offset"):
        values = np.asarray([row[key] for row in rows])
        result[key] = [float(v) for v in np.quantile(values, [0.25, 0.5, 0.75])]
    state = np.asarray([row["state"] for row in rows])
    backend = np.asarray([row["backend"] for row in rows])
    result["flagged"] = float((state < UNCERTAIN_SCORE).mean())
    result["backendBelowHalf"] = float((backend < 0.5).mean())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--tables", type=Path, nargs="+", required=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    frames = loadFrames(args.tables)
    groups: dict[str, dict[str, list]] = {
        cause: {"all": [], "early": [], "late": []} for cause in CAUSES
    }
    with args.events.open(encoding="utf-8") as stream:
        for event in csv.DictReader(stream):
            sequence, frame = event["sequence"], int(event["frame"])
            remaining, position = int(event["lostFrames"]), 0
            while remaining > 0 and (sequence, frame) in frames:
                row = frames[(sequence, frame)]
                if row["present"] and row["lost"]:
                    part = "early" if position < EARLY_FRAMES else "late"
                    groups[event["cause"]]["all"].append(row)
                    groups[event["cause"]][part].append(row)
                    remaining -= 1
                    position += 1
                elif row["present"]:
                    break
                frame += 1
    good = [row for row in frames.values() if row["present"] and row["iou"] >= 0.5]
    payload = {"good": summarize(good)}
    for cause in CAUSES:
        payload[cause] = {part: summarize(rows) for part, rows in groups[cause].items()}

    def line(label: str, item: dict[str, object]) -> str:
        if not item.get("frames"):
            return f"  {label:<26}{0:>7}"

        def cell(key: str) -> str:
            low, mid, high = item[key]
            return f"{mid:>7.2f} ({low:.2f}-{high:.2f})"

        return (
            f"  {label:<26}{item['frames']:>7}{cell('backend')}{cell('template')}"
            f"{cell('appearance')}{cell('motion')}{cell('state')}{item['flagged']:>9.1%}"
        )

    print(
        f"median (25%-75%) of each score; weights {WEIGHTS}, flagged = state < {UNCERTAIN_SCORE}"
    )
    print(
        f"  {'frames':<26}{'n':>7}{'backend':>19}{'to template':>19}{'appearance':>19}"
        f"{'motion':>19}{'state':>19}{'flagged':>9}"
    )
    print(line("good frames (IoU >= 0.5)", payload["good"]))
    for cause in CAUSES:
        print(line(f"lost after: {cause}", payload[cause]["all"]))
        print(line(f"   first {EARLY_FRAMES} lost frames", payload[cause]["early"]))
        print(line("   later lost frames", payload[cause]["late"]))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
