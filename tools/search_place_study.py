"""Where the enlarged searches should have been centred (evaluation log E053).

    python tools/search_place_study.py --dataset-root <dir> --label-root <dir> \\
        --runs outputs/E043_base2_s1 ... --camera outputs/E053/camera --out outputs/E053/place

Offline, on the recorded runs of base 2: every enlarged search made with the target in
the picture, with the view the run used (centred on the last trusted position, half
width as in ``loss_report``).  Two parts:

diagnosis   each search whose view missed the target is put in one class, in order:
            ``drift`` the view would have held it centred on where the target really
            was when the tracker last had it (IoU >= 0.5); ``rotation`` it would have,
            had that place also been turned by the camera's rotation since then;
            ``far`` neither.
rules       centres a system can compute, from the state trace alone (no ground
            truth), and the share of searches whose view would have held the target.
            Each rule also with the camera rotation since the chosen frame applied.

Searches are those of the recorded runs: with another centre the run itself would
have gone differently.  Rules are compared on the same searches.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_outcomes  # noqa: E402

from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.loss_rate import dualIou  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402


def unit(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    yaw, pitch = np.radians(yaw), np.radians(pitch)
    return np.stack(
        [np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)], axis=-1
    )


def angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


class Trace:
    """One sequence's state trace as arrays indexed by frame."""

    def __init__(self, path: Path, count: int) -> None:
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.rows = {int(r["frame"]): r for r in rows}
        self.count = count
        n = count
        self.pos = np.full((n, 3), np.nan)
        self.size = np.full(n, np.nan)
        self.backend = np.full(n, np.nan)
        self.appearance = np.full(n, np.nan)
        self.trusted = np.zeros(n, dtype=bool)
        self.jump = np.zeros(n, dtype=bool)
        for f, r in self.rows.items():
            if f >= n:
                continue
            self.pos[f] = unit(float(r["yawDeg"]), float(r["pitchDeg"]))
            self.size[f] = float(r["sizeDeg"])
            self.backend[f] = float(r["backend"] or 0.0) if r["hasBox"] == "1" else 0.0
            self.appearance[f] = float(r["appearance"] or 0.0)
            self.trusted[f] = r["modeAfter"] == "TRACKING" and r["action"] != "jump"
            self.jump[f] = r["action"] == "jump"


# ---------------------------------------------------------------- centre rules
# Each takes the trace, the index of the search frame, and the index of the last
# trusted frame before it, and returns the frame whose tracker position to use.

Rule = Callable[[Trace, int, int], int]


def ruleTrusted(trace: Trace, index: int, trusted: int) -> int:
    return trusted


def ruleBeforeJump(sizes: float, window: int) -> Rule:
    """The last position before the latest sudden move of the tracker's box."""

    def rule(trace: Trace, index: int, trusted: int) -> int:
        start = max(1, trusted - window)
        for f in range(trusted, start, -1):
            if trace.jump[f] or np.isnan(trace.pos[f]).any() or np.isnan(trace.pos[f - 1]).any():
                continue
            step = angle(trace.pos[f], trace.pos[f - 1])
            if step > sizes * max(trace.size[f - 1], 0.5):
                return f - 1
        return trusted

    return rule


def _reference(trace: Trace, upto: int, values: np.ndarray) -> float:
    past = values[1:upto][trace.trusted[1:upto]]
    past = past[np.isfinite(past)]
    return float(np.median(past)) if past.size >= 5 else float("nan")


def ruleAppearance(share: float, window: int, run: int = 3) -> Rule:
    """The last frame before ``run`` frames in a row fell below ``share`` of the median
    appearance of the earlier trusted frames."""

    def rule(trace: Trace, index: int, trusted: int) -> int:
        start = max(1, trusted - window)
        level = _reference(trace, start, trace.appearance)
        if not np.isfinite(level):
            return trusted
        good = trace.appearance >= share * level
        # Walk back from the trusted frame: the latest frame that was good and was
        # followed by a run of bad frames.
        for f in range(trusted, start, -1):
            if good[f] and not good[f + 1 : f + 1 + run].any() and f + run <= trusted:
                return f
        return trusted

    return rule


def ruleChangePoint(window: int, margin: float) -> Rule:
    """The split of the recent appearance that best separates high before from low after."""

    def rule(trace: Trace, index: int, trusted: int) -> int:
        start = max(1, trusted - window)
        values = trace.appearance[start : trusted + 1]
        if values.size < 10 or not np.isfinite(values).all():
            return trusted
        csum = np.cumsum(values)
        n = values.size
        best, split = margin, None
        for k in range(3, n - 3):
            before = csum[k - 1] / k
            after = (csum[-1] - csum[k - 1]) / (n - k)
            if before - after > best:
                best, split = before - after, k
        return trusted if split is None else start + split - 1

    return rule


def ruleEarliest(*rules: Rule) -> Rule:
    def rule(trace: Trace, index: int, trusted: int) -> int:
        return min(r(trace, index, trusted) for r in rules)

    return rule


# ---------------------------------------------------------------- study


def searches(args: argparse.Namespace, root: Path, dataset: Vot360Dataset):
    """Every enlarged search with the target in the picture, with what the rules need."""
    results = loadTrackerResults(root / "bbox" / "ours")
    for path in sorted((root / "trace" / "ours").glob("*.csv")):
        name = path.stem
        sequence = dataset.sequence(name)
        truthBox, present = sequence.groundTruth("bbox")
        truthBfov, _ = sequence.groundTruth("bfov")
        width = sequence.frameSize[0]
        sequence.close()
        count = len(results[name])
        present = np.asarray(present, dtype=bool)[:count]
        iou = np.zeros(count)
        iou[present] = dualIou(truthBox[:count][present], results[name][present], width)
        trace = Trace(path, count)
        camera = None
        if args.camera is not None and (args.camera / f"{name}.npz").exists():
            camera = np.load(args.camera / f"{name}.npz")
        trusted = None
        good = None
        for index in sorted(trace.rows):
            row = trace.rows[index]
            if (
                row["scan"] in ("2x", "3x", "4x")
                and trusted is not None
                and index < count
                and present[index]
            ):
                scale = float(row["scan"][:-1])
                size = max(trace.size[trusted], 1e-3)
                half = (
                    0.5
                    * scale
                    * max(search_outcomes.MIN_VIEW_DEG, search_outcomes.VIEW_FACTOR * size)
                )
                yield {
                    "sequence": name,
                    "frame": index,
                    "scan": row["scan"],
                    "trace": trace,
                    "trusted": trusted,
                    "good": good,
                    "half": half,
                    "truth": unit(truthBfov[index, 0], truthBfov[index, 1]),
                    "truthAt": lambda f, t=truthBfov: unit(t[f, 0], t[f, 1]),
                    "targetSize": float(
                        np.sqrt(max(truthBfov[index, 2] * truthBfov[index, 3], 1e-6))
                    ),
                    "camera": camera,
                }
            if trace.trusted[index] if index < count else False:
                trusted = index
            if index < count and present[index] and iou[index] >= 0.5:
                good = index


def turned(camera, start: int, end: int, point: np.ndarray) -> np.ndarray:
    if camera is None:
        return point
    o = camera["orientation"]
    return (o[end] @ o[start].T) @ point


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--camera", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    rules: dict[str, Rule] = {"trusted (now)": ruleTrusted}
    for sizes in (1.5, 3.0):
        for window in (60, 200):
            rules[f"before jump {sizes:g}x, {window}"] = ruleBeforeJump(sizes, window)
    for share in (0.7, 0.8, 0.9):
        for window in (60, 200):
            rules[f"appearance {share:g}, {window}"] = ruleAppearance(share, window)
    for window in (60, 200):
        for margin in (0.05, 0.10):
            rules[f"change point {window}, {margin:g}"] = ruleChangePoint(window, margin)
    rules["earliest(jump 1.5x 200, appearance 0.8 200)"] = ruleEarliest(
        rules["before jump 1.5x, 200"], rules["appearance 0.8, 200"]
    )
    classes = {"in view": 0, "drift": 0, "rotation": 0, "far": 0, "no good frame": 0}
    farDistance: list[float] = []
    hits = {name: [0, 0] for name in rules}  # plain, with camera rotation
    total = 0
    perRun = []
    for root in args.runs:
        runTotal, runHits = 0, {name: 0 for name in rules}
        for s in searches(args, root, dataset):
            total += 1
            runTotal += 1
            trace, truth, half = s["trace"], s["truth"], s["half"]
            now = trace.pos[s["trusted"]]
            if angle(now, truth) <= half:
                classes["in view"] += 1
            elif s["good"] is None:
                classes["no good frame"] += 1
            else:
                last = s["truthAt"](s["good"])
                if angle(last, truth) <= half:
                    classes["drift"] += 1
                elif angle(turned(s["camera"], s["good"], s["frame"], last), truth) <= half:
                    classes["rotation"] += 1
                else:
                    classes["far"] += 1
                    farDistance.append(angle(last, truth) / max(s["targetSize"], 1e-3))
            for name, rule in rules.items():
                frame = rule(trace, s["frame"], s["trusted"])
                centre = trace.pos[frame]
                if np.isnan(centre).any():
                    centre = now
                inside = angle(centre, truth) <= half
                hits[name][0] += int(inside)
                rotated = turned(s["camera"], frame, s["frame"], centre)
                hits[name][1] += int(angle(rotated, truth) <= half)
                runHits[name] += int(inside)
        perRun.append({"run": str(root), "searches": runTotal, "hits": runHits})
    lines = [f"{total} enlarged searches with the target in the picture ({len(args.runs)} runs)"]
    lines.append("diagnosis of the searches:")
    for name, value in classes.items():
        lines.append(f"  {name:<16}{value:>7}  {value / total:6.1%}")
    if farDistance:
        q = np.percentile(farDistance, [25, 50, 75, 90])
        lines.append(
            "  far: distance from the true last place, target sizes, quartiles and 90th: "
            + ", ".join(f"{v:.1f}" for v in q)
        )
    lines.append("centre rules: share of searches with the target in view (plain / camera-turned)")
    for name, (plain, rotated) in hits.items():
        lines.append(f"  {name:<46}{plain / total:7.1%}{rotated / total:8.1%}")
    text = "\n".join(lines)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.txt").write_text(text + "\n", encoding="utf-8")
    (args.out / "report.json").write_text(
        json.dumps(
            {
                "total": total,
                "classes": classes,
                "farDistance": farDistance,
                "rules": {k: {"plain": v[0], "turned": v[1]} for k, v in hits.items()},
                "perRun": perRun,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
