"""Look-alikes found by DINOv2 as the gallery's negatives (E078, offline, dev train).

    python tools/hard_negative_study.py --dataset-root <dir> --label-root <dir> \\
        --runs outputs/E066_s1 outputs/E066_s2 --out outputs/E078

As ``negative_gallery_study.py`` (E076a), with a second source of negatives: on
confident frames (trusted, relative signal >= ``--confident``, margin >= 0 in the
doubt-only gallery),
every ``--harvest``-th frame, the DINOv2 patch similarity to the frame-0 target is
computed over a view ``--scale`` times the normal one around the tracker's box; its
best local maxima farther than ``--apart`` tracker sizes from the box are cut out at the
tracker's size and kept as negatives (at most ``--per-harvest`` each time).

Four galleries are scored side by side on the same frames, all with the same
positives (frame-0 template and confident trusted frames):

- doubt: negatives are the boxes of doubted frames (E076b online);
- doubt+rejected: plus the scan candidates not taken (E076a offline);
- lookalike: only the harvested look-alikes;
- doubt+lookalike: both.

Reported per gallery: AUROC for wrong against right on trusted frames and on releases,
for elsewhere against on the target on taken jumps and on all candidates scoring at
least 0.70, and how often the best of several such candidates is on the target.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import replace
from math import sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from negative_gallery_study import _bfov, auroc  # noqa: E402
from proposal_study import DenseDino  # noqa: E402
from size_study import angle, sequenceData, unit  # noqa: E402

from track360.backends.appearance import AppearanceModels, targetCrop  # noqa: E402
from track360.controller import TrackControllerImpl  # noqa: E402
from track360.controller.view_planner import localBoxOfBfov  # noqa: E402
from track360.core.config import loadConfig  # noqa: E402
from track360.core.types import BBoxXYWH, FrameIndex, FramePacket, SequenceId  # noqa: E402
from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402
from track360.geometry import SphericalGeometryImpl  # noqa: E402

GALLERIES = ("doubt", "doubt+rejected", "lookalike", "doubt+lookalike")


class Galleries:
    """Shared positives; negatives by source."""

    def __init__(self, template: np.ndarray, keep: int) -> None:
        self.template = template
        self.keep = keep
        self.positives: list[np.ndarray] = []
        self.sources: dict[str, list[np.ndarray]] = {"doubt": [], "rejected": [], "lookalike": []}

    def _push(self, store: list, feature: np.ndarray) -> None:
        store.append(feature)
        del store[: max(0, len(store) - self.keep)]

    def addPositive(self, feature: np.ndarray) -> None:
        self._push(self.positives, feature)

    def addNegative(self, source: str, feature: np.ndarray) -> None:
        self._push(self.sources[source], feature)

    def margins(self, feature: np.ndarray) -> dict[str, float]:
        pos = max([float(feature @ self.template)] + [float(feature @ p) for p in self.positives])

        def best(names: tuple[str, ...]) -> float:
            values = [float(feature @ n) for name in names for n in self.sources[name]]
            return max(values) if values else 0.0

        return {
            "doubt": pos - best(("doubt",)),
            "doubt+rejected": pos - best(("doubt", "rejected")),
            "lookalike": pos - best(("lookalike",)),
            "doubt+lookalike": pos - best(("doubt", "lookalike")),
        }


def replay(args) -> list[dict]:
    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    hub = Path(config.model.weights).parent / "hub"
    models = AppearanceModels(hub, ("dinov2",))
    dense = DenseDino(hub, args.dino_side)
    out: list[dict] = []
    for root in args.runs:
        results = loadTrackerResults(root / "bbox" / "ours")
        for name in readSequenceFile(args.sequence_file):
            d = sequenceData(dataset, root, name, results)
            if d is None:
                continue
            with (root / "trace" / "ours" / f"{name}.csv").open(encoding="utf-8") as stream:
                trace = {int(r["frame"]): r for r in csv.DictReader(stream)}
            boxes = np.loadtxt(root / "bfov" / "ours" / f"{name}.txt", delimiter=",", ndmin=2)
            sequence = dataset.sequence(name)
            truth, _ = sequence.groundTruth("bfov")
            width, height = sequence.frameSize
            geometry = SphericalGeometryImpl(
                boundarySamplesPerEdge=config.geometry.boundarySamplesPerEdge
            )
            controller = TrackControllerImpl(geometry, config)
            planner = controller._planner

            def embed(frame, target) -> np.ndarray:
                spec = planner.probeView(
                    target.center, target.horizontalFovRad, target.verticalFovRad, 1.0, 0
                )
                view = geometry.cropViews(frame, [spec])[0]
                feature = models.embed(targetCrop(view.rgb, localBoxOfBfov(spec, target)))
                return feature["dinov2"].cpu().numpy().reshape(-1)

            def wide(frame, target):
                spec = planner.probeView(
                    target.center,
                    target.horizontalFovRad,
                    target.verticalFovRad,
                    args.scale,
                    0,
                )
                spec = replace(spec, outputWidthPx=args.width, outputHeightPx=args.width)
                return spec, geometry.cropViews(frame, [spec])[0].rgb

            try:
                first = FramePacket(SequenceId(name), FrameIndex(0), 0, sequence.readRgb(0))
                initial = sequence.initialBfov()
                plan = controller.buildInitialization(first, initialBfov=initial)
                template = geometry.cropViews(first, [plan.templateView])[0]
                ref = models.embed(targetCrop(template.rgb, plan.templateBox))
                gallery = Galleries(ref["dinov2"].cpu().numpy().reshape(-1), args.keep)
                spec0, rgb0 = wide(first, initial)
                dense.setTarget(rgb0, localBoxOfBfov(spec0, initial))
                harvested = 0
                for t in range(1, min(len(boxes), len(d["present"]))):
                    r = trace.get(t)
                    if r is None:
                        continue
                    frame = FramePacket(SequenceId(name), FrameIndex(t), t, sequence.readRgb(t))
                    tracked = _bfov(*boxes[t][:4])
                    mine = embed(frame, tracked)
                    present = bool(d["present"][t])
                    margins = gallery.margins(mine)
                    row = {
                        "run": root.name,
                        "sequence": name,
                        "frame": t,
                        "right": bool(present and d["iou"][t] >= 0.5),
                        "wrong": bool((not present) or d["iou"][t] < 0.1),
                        "modeAfter": r["modeAfter"],
                        "reason": r["reason"],
                        "template": float(mine @ gallery.template),
                        **{f"m:{k}": v for k, v in margins.items()},
                        "candidates": [],
                    }
                    rejected = []
                    if r["candidates"] not in ("", "[]"):
                        for c in json.loads(r["candidates"]):
                            place = _bfov(c["yawDeg"], c["pitchDeg"], c["sizeDeg"], c["sizeDeg"])
                            feature = embed(frame, place)
                            size = float(np.sqrt(max(truth[t, 2] * truth[t, 3], 1e-6)))
                            on = present and float(
                                angle(
                                    unit(c["yawDeg"], c["pitchDeg"]),
                                    unit(truth[t, 0], truth[t, 1]),
                                )
                            ) < 0.5 * max(float(c["sizeDeg"]), size)
                            cm = gallery.margins(feature)
                            row["candidates"].append(
                                {
                                    "score": float(c["score"]),
                                    "verdict": c["verdict"],
                                    "onTarget": bool(on),
                                    "template": float(feature @ gallery.template),
                                    **{f"m:{k}": v for k, v in cm.items()},
                                }
                            )
                            if c["verdict"] != "accepted":
                                rejected.append(feature)
                    out.append(row)
                    # The galleries learn from this frame only after it was scored.
                    for feature in rejected:
                        gallery.addNegative("rejected", feature)
                    relative = r["relative"]
                    confident = (
                        r["modeAfter"] == "TRACKING"
                        and r["action"] != "jump"
                        and relative not in ("", "None")
                        and float(relative) >= args.confident
                    )
                    if t % args.every == 0 and r["action"] != "jump":
                        if confident:
                            gallery.addPositive(mine)
                        elif r["modeAfter"] in ("UNCERTAIN", "LOST"):
                            gallery.addNegative("doubt", mine)
                    if confident and margins["doubt"] >= 0.0 and t % args.harvest == 0:
                        spec, rgb = wide(frame, tracked)
                        box = localBoxOfBfov(spec, tracked)
                        cx, cy = box.xPx + 0.5 * box.widthPx, box.yPx + 0.5 * box.heightPx
                        side = max(sqrt(box.widthPx * box.heightPx), 4.0)
                        kept = 0
                        for x, y, _ in dense.candidates(rgb, side, count=10):
                            if sqrt((x - cx) ** 2 + (y - cy) ** 2) < args.apart * side:
                                continue
                            local = BBoxXYWH(
                                xPx=x - 0.5 * box.widthPx,
                                yPx=y - 0.5 * box.heightPx,
                                widthPx=box.widthPx,
                                heightPx=box.heightPx,
                            )
                            try:
                                place = geometry.projectLocalBoxBoundary(
                                    local, spec, width, height
                                ).bfov
                            except Exception:
                                continue
                            gallery.addNegative("lookalike", embed(frame, place))
                            kept += 1
                            harvested += 1
                            if kept >= args.per_harvest:
                                break
                    release = getattr(geometry, "releaseFrame", None)
                    if callable(release):
                        release()
            finally:
                sequence.close()
            print(f"  {root.name} {name}: {harvested} look-alikes", flush=True)
    return out


def report(rows: list[dict]) -> str:
    lines = []
    keys = ("template", *(f"m:{g}" for g in GALLERIES))

    def table(label: str, bad: list[dict], good: list[dict]) -> None:
        cells = "  ".join(
            f"{k.removeprefix('m:')} {auroc([-x[k] for x in bad], [-x[k] for x in good]):.3f}"
            for k in keys
        )
        lines.append(f"  {label} ({len(good)} / {len(bad)}): {cells}")

    lines.append("AUROC, low value = wrong / elsewhere (right-or-on / wrong-or-elsewhere):")
    trusted = [r for r in rows if r["modeAfter"] == "TRACKING"]
    table("trusted frames", [r for r in trusted if r["wrong"]], [r for r in trusted if r["right"]])
    released = [r for r in rows if r["reason"] == "RELEASED"]
    table("releases", [r for r in released if r["wrong"]], [r for r in released if r["right"]])
    cands = [c for r in rows for c in r["candidates"]]
    jumps = [c for c in cands if c["verdict"] == "accepted"]
    table(
        "taken jumps",
        [c for c in jumps if not c["onTarget"]],
        [c for c in jumps if c["onTarget"]],
    )
    strong = [c for c in cands if c["score"] >= 0.70]
    table(
        "candidates >= 0.70",
        [c for c in strong if not c["onTarget"]],
        [c for c in strong if c["onTarget"]],
    )
    groups = []
    for r in rows:
        s = [c for c in r["candidates"] if c["score"] >= 0.70]
        if len(s) >= 2 and any(c["onTarget"] for c in s):
            groups.append(s)
    if groups:
        lines.append(
            f"  best of several candidates >= 0.70 on the target ({len(groups)} searches): "
            + "  ".join(
                f"{k.removeprefix('m:')} {np.mean([max(s, key=lambda c: c[k])['onTarget'] for s in groups]):.1%}"  # noqa: E501
                for k in keys
            )
        )
    for g in GALLERIES:
        key = f"m:{g}"
        good = [r for r in trusted if r["right"]]
        bad = [r for r in trusted if r["wrong"]]
        lines.append(
            f"  {g}: trusted frames below -0.15 / -0.25: right"
            f" {np.mean([r[key] < -0.15 for r in good]):.1%} / {np.mean([r[key] < -0.25 for r in good]):.1%},"  # noqa: E501
            f" wrong {np.mean([r[key] < -0.15 for r in bad]):.1%} / {np.mean([r[key] < -0.25 for r in bad]):.1%}"  # noqa: E501
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
        default=Path(__file__).resolve().parents[1] / "configs" / "second_look.yaml",
    )
    parser.add_argument(
        "--confident",
        type=float,
        default=-0.2,
        help="relative signal from which a trusted frame is confident (median about -0.2)",
    )
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--keep", type=int, default=50)
    parser.add_argument("--harvest", type=int, default=10)
    parser.add_argument("--per-harvest", type=int, default=3)
    parser.add_argument("--apart", type=float, default=1.5)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--dino-side", type=int, default=448)
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
