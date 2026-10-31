"""Replay the state machine on recorded scores with other parameters (E082, offline).

    python tools/state_replay_study.py --dataset-root <dir> --label-root <dir> \\
        --runs outputs/E080a_s1 ... --out outputs/E082/replay

The relative rule (``TrackStateMachine``) is run again over each recorded frame's
backend score and appearance similarity, with one parameter changed at a time; a
recorded jump resets it as it did in the run.  The tracker's boxes stay the recorded
ones, so this counts how the judgement would change, not what the changed judgement
would do to the track (frozen memory, searches and jumps follow the judgement online).
Reported per setting, per run: frames distrusted while the tracker is wrong (absent or
IoU < 0.1) and while it is right (IoU >= 0.5), lost frames (where a search runs),
releases and their share on the target, and frames from the onset of a wrong stretch to
the first distrusted frame; and how often the replay with the current parameters agrees
with the recorded judgement.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from size_study import sequenceData  # noqa: E402

from track360.controller.state_machine import TrackStateMachine  # noqa: E402
from track360.controller.state_model import TrackMode  # noqa: E402
from track360.core.config import loadConfig  # noqa: E402
from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402

SETTINGS = {
    "current": {},
    "enter -0.45": {"relativeEnterDeviation": -0.45},
    "enter -0.60": {"relativeEnterDeviation": -0.60},
    "enter -0.68": {"relativeEnterDeviation": -0.68},
    "enter -0.75": {"relativeEnterDeviation": -0.75},
    "release 2": {"releaseFrames": 2},
    "release 5": {"releaseFrames": 5},
    "release 8": {"releaseFrames": 8},
    "lost 2": {"lostAfterFrames": 2},
    "lost 8": {"lostAfterFrames": 8},
    "share 0.25": {"share": 0.25},
    "share 0.75": {"share": 0.75},
    "gate 0.00": {"relativeGate": 0.0},
    "gate 0.15": {"relativeGate": 0.15},
}


def replay(rows, tuning, share):
    machine = TrackStateMachine(tuning)
    machine.initialize()
    if share is not None:
        machine.useReleaseShare(share)
    mode = TrackMode.TRACKING
    out = []
    for r in rows:
        if r["action"] == "jump":
            machine.reset()
            mode = TrackMode.TRACKING
            out.append((TrackMode.TRACKING, ""))
            continue
        has = r["hasBox"] == "1"
        decision = machine.transition(
            mode,
            min(1.0, max(0.0, float(r["stateScore"] or 0.0))),
            measurementAccepted=True,
            backendScore=float(r["backend"]) if has and r["backend"] else None,
            appearanceScore=float(r["appearance"]) if r["appearance"] else None,
        )
        out.append((decision.nextMode, decision.reason.name))
        mode = decision.nextMode
    return out


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
        default=Path(__file__).resolve().parents[1] / "configs" / "lookalike.yaml",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    base = loadConfig(args.config).backendTuning
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    totals = {name: np.zeros(8) for name in SETTINGS}
    agree = [0, 0]
    for root in args.runs:
        results = loadTrackerResults(root / "bbox" / "ours")
        for name in readSequenceFile(args.sequence_file):
            d = sequenceData(dataset, root, name, results)
            if d is None:
                continue
            with (root / "trace" / "ours" / f"{name}.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            present, iou = d["present"], d["iou"]
            frames = [int(r["frame"]) for r in rows]
            right = [f < len(present) and present[f] and iou[f] >= 0.5 for f in frames]
            wrong = [f < len(present) and ((not present[f]) or iou[f] < 0.1) for f in frames]
            for label, change in SETTINGS.items():
                share = change.get("share")
                tuning = replace(base, **{k: v for k, v in change.items() if k != "share"})
                decided = replay(rows, tuning, share)
                untrusted = [m is not TrackMode.TRACKING for m, _ in decided]
                lost = [m is TrackMode.LOST for m, _ in decided]
                releases = [i for i, (_, reason) in enumerate(decided) if reason == "RELEASED"]
                delays = []
                for i in range(1, len(frames)):
                    if wrong[i] and right[i - 1]:
                        k = next((j for j in range(i, len(frames)) if untrusted[j]), None)
                        if k is not None and all(not right[j] for j in range(i, k + 1)):
                            delays.append(k - i)
                totals[label] += [
                    sum(u and w for u, w in zip(untrusted, wrong, strict=True)),
                    sum(u and r for u, r in zip(untrusted, right, strict=True)),
                    sum(lost),
                    len(releases),
                    sum(right[i] for i in releases),
                    sum(delays),
                    len(delays),
                    1,
                ]
                if label == "current":
                    for (m, _), r in zip(decided, rows, strict=True):
                        agree[0] += (m is TrackMode.TRACKING) == (r["modeAfter"] == "TRACKING")
                        agree[1] += 1
    runs = len(args.runs)
    lines = [
        f"replay with the current parameters agrees with the recorded trust on"
        f" {agree[0] / max(1, agree[1]):.1%} of frames (the release hold of baseline 5 is"
        " not replayed)",
        "per run: distrusted wrong / right | lost frames | releases (on target) |"
        " mean delay to the first distrust after a wrong onset",
    ]
    for label, t in totals.items():
        t = t / runs
        lines.append(
            f"  {label:<12} {t[0]:7.0f} / {t[1]:5.0f} | {t[2]:6.0f} |"
            f" {t[3]:5.1f} ({t[4] / max(t[3], 1e-9):.1%}) | {t[5] / max(t[6], 1e-9):5.1f}"
            f" (n={t[6]:.0f})"
        )
    text = "\n".join(lines)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.txt").write_text(text + "\n", encoding="utf-8")
    (args.out / "totals.json").write_text(
        json.dumps({k: (v / runs).tolist() for k, v in totals.items()}) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
