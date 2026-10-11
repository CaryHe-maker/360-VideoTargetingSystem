"""Loss detection against each sequence's own baseline, compared with fixed thresholds.

    python tools/baseline_analysis.py --tune outputs/E018/tune_freerun.csv \\
        --holdout outputs/E018/holdout_freerun.csv --events outputs/E019/loss_events.csv \\
        --json outputs/E021/baseline.json

Works on the per-frame tables of ``tools/fusion_freerun.py``; the tracker is not run.

A fixed threshold compares a score with one number for every sequence.  The baseline
rule compares it with what the same sequence showed so far: a running median and
spread (median absolute deviation) of the frames trusted up to now, using only earlier
frames.  The deviation from that baseline is thresholded instead.

    mode     z: (value - median) / spread; diff: value - median; ratio: value / median - 1
    window   how many trusted frames the baseline remembers (0: all of them;
             -1: only the warm-up frames, the baseline never moves afterwards)
    warm     frames at the start that are trusted without a check
    gate     a frame enters the baseline only when its deviation is above -gate
             (0: always)

The ``oracle`` rows subtract the median of the sequence's good frames, which uses the
ground truth and the whole sequence: they bound what any per-sequence level can do.

Every detector is compared at the same false-alarm rate (flagged share of the frames
with IoU >= 0.5).  The settings are chosen on the tune table and then applied to the
hold-out table twice: with the threshold re-matched there, and with the tune threshold
carried over unchanged, which is what deployment does.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import deque
from pathlib import Path

import numpy as np

RATES = (0.008, 0.02, 0.05)
SELECT_RATE = 0.02
EARLY_FRAMES = 5
OFFSET_SCALE, SIZE_SCALE = 0.5, 0.1
WEIGHTS = (0.40, 0.55, 0.05)
# Smallest spread a baseline may report, per channel, so a very steady start does not
# turn every later wobble into a large deviation.
FLOORS = {"template": 0.05, "backend": 0.03, "offset": 0.10, "scale": 0.02}
WINDOWS = (-1, 30, 100, 300, 0)
WARMS = (10, 30)
GATES = {"z": (0.0, 2.0, 3.0), "diff": (0.0, 0.1, 0.2), "ratio": (0.0, 0.15, 0.3)}
COMBINATIONS = {
    "template": ("template",),
    "backend": ("backend",),
    "template+backend": ("template", "backend"),
}
LATCH_RELEASE, LATCH_HOLD = 0.5, 3
# A latched fixed threshold releases this far above the threshold; chosen on tune.
FIXED_RELEASES = (0.05, 0.10, 0.20)

Sequence = dict[str, np.ndarray]


def loadTable(path: Path) -> dict[str, Sequence]:
    rows: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.setdefault(row["sequence"], []).append(row)
    table = {}
    for name, items in rows.items():
        items.sort(key=lambda row: int(row["frame"]))

        def column(key: str, items: list[dict[str, str]] = items) -> np.ndarray:
            return np.asarray([float(row[key]) for row in items])

        present = column("present") > 0
        iou = column("iou")
        template = np.clip(column("simTemplate"), 0.0, 1.0)
        backend = np.clip(column("score"), 0.0, 1.0)
        offset, scale = column("offset"), np.abs(column("logScale"))
        motion = np.exp(-0.5 * (offset / OFFSET_SCALE) ** 2) * np.exp(
            -0.5 * (scale / SIZE_SCALE) ** 2
        )
        table[name] = {
            "frame": column("frame").astype(int),
            "good": present & (iou >= 0.5),
            "lost": present & (column("lostRun") > 0),
            # Channels: larger is better for all of them.
            "template": template,
            "backend": backend,
            "offset": -offset,
            "scale": -scale,
            "fused": WEIGHTS[0] * backend + WEIGHTS[1] * template + WEIGHTS[2] * motion,
        }
    return table


def lossParts(table: dict[str, Sequence], events: Path) -> dict[str, dict[str, np.ndarray]]:
    """Masks of the lost frames of each cause, and of their first frames."""
    position = {
        name: {int(frame): index for index, frame in enumerate(sequence["frame"])}
        for name, sequence in table.items()
    }
    parts = {name: {} for name in table}
    with events.open(encoding="utf-8") as stream:
        for event in csv.DictReader(stream):
            name = event["sequence"]
            if name not in table:
                continue
            sequence = table[name]
            frame, remaining, count = int(event["frame"]), int(event["lostFrames"]), 0
            while remaining > 0 and frame in position[name]:
                index = position[name][frame]
                if sequence["lost"][index]:
                    for key in (event["cause"], "early" if count < EARLY_FRAMES else "late"):
                        parts[name].setdefault(key, np.zeros(len(sequence["frame"]), bool))[
                            index
                        ] = True
                    remaining -= 1
                    count += 1
                frame += 1
    return parts


def causalDeviation(
    values: np.ndarray, mode: str, window: int, warm: int, gate: float, floor: float
):
    """Deviation of every frame from the baseline of the trusted frames before it."""
    history: deque[float] = deque(maxlen=window if window > 0 else None)
    deviation = np.zeros(len(values))
    for index, value in enumerate(values):
        if len(history) < warm:
            history.append(float(value))
            continue
        past = np.fromiter(history, dtype=float)
        median = float(np.median(past))
        if mode == "z":
            spread = max(1.4826 * float(np.median(np.abs(past - median))), floor)
            deviation[index] = (value - median) / spread
        elif mode == "diff":
            deviation[index] = value - median
        else:
            deviation[index] = value / max(median, 1e-6) - 1.0
        if window >= 0 and (gate <= 0.0 or deviation[index] > -gate):
            history.append(float(value))
    return deviation


def latch(signal: np.ndarray, threshold: float, release: float) -> np.ndarray:
    """Flag from the first frame below the threshold until the signal is back for a while."""
    flagged = np.zeros(len(signal), bool)
    active, calm = False, 0
    for index, value in enumerate(signal):
        if not active:
            active = value < threshold
            calm = 0
        else:
            calm = calm + 1 if value > release else 0
            if calm >= LATCH_HOLD:
                active = False
        flagged[index] = active
    return flagged


class Detector:
    """One signal per frame for every sequence; smaller means more likely lost."""

    def __init__(
        self,
        signals: dict[str, np.ndarray],
        latched: bool = False,
        releaseMargin: float | None = None,
    ) -> None:
        self.signals = signals
        self.latched = latched
        # Deviations release at a share of the threshold, raw scores a margin above it.
        self.releaseMargin = releaseMargin

    def flags(self, threshold: float) -> dict[str, np.ndarray]:
        if self.latched:
            release = (
                threshold * LATCH_RELEASE
                if self.releaseMargin is None
                else threshold + self.releaseMargin
            )
            return {
                name: latch(signal, threshold, release)
                for name, signal in self.signals.items()
            }
        return {name: signal < threshold for name, signal in self.signals.items()}

    def falseAlarm(self, table: dict[str, Sequence], threshold: float) -> float:
        flags = self.flags(threshold)
        hit = sum(int(flags[name][seq["good"]].sum()) for name, seq in table.items())
        return hit / max(1, sum(int(seq["good"].sum()) for seq in table.values()))

    def threshold(self, table: dict[str, Sequence], rate: float) -> float:
        good = np.concatenate([self.signals[name][seq["good"]] for name, seq in table.items()])
        if not self.latched:
            return float(np.quantile(good, rate))
        low, high = float(good.min()) - 1.0, float(np.quantile(good, 0.25))
        for _ in range(18):
            middle = 0.5 * (low + high)
            if self.falseAlarm(table, middle) > rate:
                high = middle
            else:
                low = middle
        return low

    def measure(self, table, parts, threshold: float) -> dict[str, float]:
        flags = self.flags(threshold)
        result = {"threshold": threshold, "falseAlarm": self.falseAlarm(table, threshold)}
        keys = ("all", "early", "late", "disappeared", "grew", "shrank", "jumped", "other")
        hit, total = dict.fromkeys(keys, 0), dict.fromkeys(keys, 0)
        noisy = 0
        for name, sequence in table.items():
            good = sequence["good"]
            if good.sum() >= 20 and flags[name][good].mean() > 0.10:
                noisy += 1
            masks = {"all": sequence["lost"], **parts[name]}
            for key, mask in masks.items():
                hit[key] += int(flags[name][mask].sum())
                total[key] += int(mask.sum())
        for key in keys:
            result[key] = hit[key] / total[key] if total[key] else float("nan")
        # Sequences whose own false-alarm rate is above 10%: where the rule does not fit.
        result["noisySequences"] = noisy
        return result


def fixedDetectors(table: dict[str, Sequence]) -> dict[str, Detector]:
    items = {
        f"fixed: {key}": Detector({name: seq[key] for name, seq in table.items()})
        for key in ("template", "backend", "fused")
    }

    def level(sequence: Sequence, key: str) -> np.ndarray:
        good = sequence[key][sequence["good"]]
        return sequence[key] - (float(np.median(good)) if len(good) else 0.0)

    for label, channels in COMBINATIONS.items():
        items[f"oracle: {label}"] = Detector(
            {
                name: np.min([level(seq, channel) for channel in channels], axis=0)
                for name, seq in table.items()
            }
        )
    return items


def baselineDetector(
    table, channels, mode, window, warm, gate, cache, latched=False
) -> Detector:
    signals = {}
    for name, sequence in table.items():
        parts = []
        for channel in channels:
            key = (id(table), name, channel, mode, window, warm, gate)
            if key not in cache:
                cache[key] = causalDeviation(
                    sequence[channel], mode, window, warm, gate, FLOORS[channel]
                )
            parts.append(cache[key])
        signals[name] = np.min(parts, axis=0)
    return Detector(signals, latched)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tune", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    tune, holdout = loadTable(args.tune), loadTable(args.holdout)
    tuneParts, holdoutParts = lossParts(tune, args.events), lossParts(holdout, args.events)
    cache: dict[tuple, np.ndarray] = {}
    payload: dict[str, object] = {"search": [], "tune": {}, "holdout": {}, "transfer": {}}

    # 1. Search the baseline settings on tune, per channel combination.
    chosen: dict[str, tuple[str, int, int, float]] = {}
    for label, channels in COMBINATIONS.items():
        best = None
        settings = [
            (mode, window, warm, gate)
            for mode in GATES
            for window, warm, gate in itertools.product(WINDOWS, WARMS, GATES[mode])
            # A baseline that never moves has no gate.
            if window >= 0 or gate == 0.0
        ]
        for mode, window, warm, gate in settings:
            detector = baselineDetector(tune, channels, mode, window, warm, gate, cache)
            result = detector.measure(tune, tuneParts, detector.threshold(tune, SELECT_RATE))
            payload["search"].append(
                {
                    "channels": label,
                    "mode": mode,
                    "window": window,
                    "warm": warm,
                    "gate": gate,
                    **result,
                }
            )
            if best is None or result["all"] > best[0]:
                best = (result["all"], (mode, window, warm, gate))
        chosen[label] = best[1]
    payload["chosen"] = {label: list(value) for label, value in chosen.items()}

    # The same latch on the fixed thresholds, to separate its effect from the baseline's.
    margins = {}
    for key in ("template", "fused"):
        scores = {}
        for margin in FIXED_RELEASES:
            detector = Detector({name: seq[key] for name, seq in tune.items()}, True, margin)
            scores[margin] = detector.measure(
                tune, tuneParts, detector.threshold(tune, SELECT_RATE)
            )["all"]
        margins[key] = max(scores, key=scores.get)
    payload["fixedReleaseMargins"] = margins

    def detectors(table) -> dict[str, Detector]:
        items = fixedDetectors(table)
        for key, margin in margins.items():
            items[f"fixed: {key}, latched (release +{margin:g})"] = Detector(
                {name: seq[key] for name, seq in table.items()}, True, margin
            )
        for label, channels in COMBINATIONS.items():
            mode, window, warm, gate = chosen[label]
            span = {-1: "frozen", 0: "all"}.get(window, window)
            name = f"baseline: {label} ({mode}, window {span}, warm {warm}, gate {gate:g})"
            items[name] = baselineDetector(table, channels, mode, window, warm, gate, cache)
            items[name + ", latched"] = baselineDetector(
                table, channels, mode, window, warm, gate, cache, latched=True
            )
        return items

    tuneDetectors, holdoutDetectors = detectors(tune), detectors(holdout)

    def show(title: str, rows: dict[str, dict[str, float]]) -> None:
        print(f"\n{title}")
        print(
            f"  {'detector':<72}{'thr':>8}{'FA':>7}{'all':>7}{'early':>7}{'late':>7}"
            f"{'disap':>7}{'grew':>7}{'shrank':>7}{'jumped':>7}{'other':>7}{'noisy':>6}"
        )
        for name, row in rows.items():
            columns = ("all", "early", "late", "disappeared", "grew", "shrank", "jumped")
            cells = "".join(f"{row[key]:>7.1%}" for key in (*columns, "other"))
            print(
                f"  {name:<72}{row['threshold']:>8.3f}{row['falseAlarm']:>7.1%}{cells}"
                f"{row['noisySequences']:>6}"
            )

    for rate in RATES:
        key = f"{rate:.1%}"
        thresholds = {name: d.threshold(tune, rate) for name, d in tuneDetectors.items()}
        payload["tune"][key] = {
            name: d.measure(tune, tuneParts, thresholds[name]) for name, d in tuneDetectors.items()
        }
        payload["holdout"][key] = {
            name: d.measure(holdout, holdoutParts, d.threshold(holdout, rate))
            for name, d in holdoutDetectors.items()
        }
        payload["transfer"][key] = {
            name: d.measure(holdout, holdoutParts, thresholds[name])
            for name, d in holdoutDetectors.items()
        }
        show(f"tune, false alarms matched to {key}", payload["tune"][key])
        show(f"hold-out, false alarms matched to {key} on hold-out", payload["holdout"][key])
        show(f"hold-out with the tune threshold for {key}", payload["transfer"][key])
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
