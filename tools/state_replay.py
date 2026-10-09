"""Replay the state machine over recorded per-frame scores, without the tracker.

    python tools/state_replay.py --table outputs/E018/tune_freerun.csv \\
        --events outputs/E019/loss_events.csv --rule relative

Feeds the scores of a free-running table (``tools/fusion_freerun.py``) to the real
``TrackStateMachine`` frame by frame and reports in which state the good frames
(IoU >= 0.5) and the lost frames ended up.  Nothing acts on the states, so this is what
a run with ``lossActions: none`` gives.  The thresholds are those of ``--config``.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from track360.controller.state_evaluator import fuseStateScore
from track360.controller.state_machine import TrackStateMachine
from track360.controller.state_model import TrackMode
from track360.core.config import BackendTuningConfig, loadConfig

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
EARLY_FRAMES = 5
STATES = ("TRACKING", "UNCERTAIN", "LOST")


def loadTable(path: Path, tuning: BackendTuningConfig) -> dict[str, dict[str, np.ndarray]]:
    rows: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.setdefault(row["sequence"], []).append(row)
    table = {}
    for name, items in rows.items():
        items.sort(key=lambda row: int(row["frame"]))

        def column(key: str, items: list[dict[str, str]] = items) -> np.ndarray:
            return np.asarray([float(row[key]) for row in items])

        present, iou = column("present") > 0, column("iou")
        offset, scale = column("offset"), column("logScale")
        table[name] = {
            "frame": column("frame").astype(int),
            "good": present & (iou >= 0.5),
            "lost": present & (column("lostRun") > 0),
            "backend": np.clip(column("score"), 0.0, 1.0),
            "appearance": np.clip(column("simTemplate"), 0.0, 1.0),
            "motion": np.exp(-0.5 * (offset / tuning.motionOffsetScale) ** 2)
            * np.exp(-0.5 * (scale / tuning.motionSizeScale) ** 2),
        }
    return table


def lossParts(table, events: Path | None) -> dict[str, dict[str, np.ndarray]]:
    parts: dict[str, dict[str, np.ndarray]] = {name: {} for name in table}
    if events is None:
        return parts
    position = {
        name: {int(frame): index for index, frame in enumerate(sequence["frame"])}
        for name, sequence in table.items()
    }
    with events.open(encoding="utf-8") as stream:
        for event in csv.DictReader(stream):
            name = event["sequence"]
            if name not in table:
                continue
            frame, remaining, count = int(event["frame"]), int(event["lostFrames"]), 0
            while remaining > 0 and frame in position[name]:
                index = position[name][frame]
                if table[name]["lost"][index]:
                    for key in (event["cause"], "early" if count < EARLY_FRAMES else "late"):
                        parts[name].setdefault(key, np.zeros(len(table[name]["frame"]), bool))[
                            index
                        ] = True
                    remaining -= 1
                    count += 1
                frame += 1
    return parts


def replay(sequence: dict[str, np.ndarray], tuning: BackendTuningConfig) -> np.ndarray:
    """The state after each frame, as an index into ``STATES``."""
    machine = TrackStateMachine(tuning)
    machine.initialize()
    mode = TrackMode.TRACKING
    states = np.zeros(len(sequence["frame"]), dtype=int)
    for index in range(len(states)):
        backend = float(sequence["backend"][index])
        appearance = float(sequence["appearance"][index])
        motion = float(sequence["motion"][index])
        mode = machine.transition(
            mode,
            fuseStateScore(backend, appearance, motion, tuning),
            measurementAccepted=True,
            backendScore=backend,
            appearanceScore=appearance,
        ).nextMode
        states[index] = STATES.index(mode.name)
    return states


def measure(table, parts, tuning: BackendTuningConfig) -> dict[str, object]:
    keys = ("good", "lost", "early", "late", "disappeared", "grew", "shrank", "jumped", "other")
    counts = {key: np.zeros(len(STATES)) for key in keys}
    episodes = onGood = 0
    for name, sequence in table.items():
        states = replay(sequence, tuning)
        masks = {"good": sequence["good"], "lost": sequence["lost"], **parts[name]}
        for key, mask in masks.items():
            counts[key] += np.bincount(states[mask], minlength=len(STATES))
        # A LOST episode starts where the state becomes LOST.
        starts = np.flatnonzero((states == 2) & (np.r_[0, states[:-1]] != 2))
        episodes += len(starts)
        onGood += int(sequence["good"][starts].sum())
    result: dict[str, object] = {"lostEpisodes": episodes, "lostEpisodesOnGoodFrames": onGood}
    for key in keys:
        total = counts[key].sum()
        result[key] = {
            "frames": int(total),
            **{
                state: float(counts[key][index] / total) if total else float("nan")
                for index, state in enumerate(STATES)
            },
        }
    return result


def show(label: str, result: dict[str, object]) -> None:
    print(
        f"{label}: {result['lostEpisodes']} LOST episodes, "
        f"{result['lostEpisodesOnGoodFrames']} of them starting on a good frame"
    )
    print(f"  {'frames':<14}{'n':>7}{'TRACKING':>10}{'UNCERTAIN':>11}{'LOST':>8}")
    for key, row in result.items():
        if isinstance(row, dict) and row.get("frames"):
            print(
                f"  {key:<14}{row['frames']:>7}{row['TRACKING']:>10.1%}"
                f"{row['UNCERTAIN']:>11.1%}{row['LOST']:>8.1%}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rule", choices=("fused", "relative"), default="relative")
    parser.add_argument("--latch", action="store_true", help="fused rule with the latch")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    tuning = replace(
        loadConfig(args.config).backendTuning,
        lossHandling=True,
        stateRule=args.rule,
        stateLatch=args.latch,
    )
    table = loadTable(args.table, tuning)
    parts = lossParts(table, args.events)
    payload: dict[str, object] = {}
    payload["result"] = measure(table, parts, tuning)
    label = f"{args.rule}{' + latch' if args.latch else ''}, thresholds of the config"
    show(label, payload["result"])
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
