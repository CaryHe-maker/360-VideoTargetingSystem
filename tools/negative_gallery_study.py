"""Does a per-sequence gallery of what the target is not tell right from wrong? (E076a)

    python tools/negative_gallery_study.py --dataset-root <dir> --label-root <dir> \\
        --runs outputs/E066_s1 outputs/E066_s2 --out outputs/E076a

Offline, along recorded runs.  Every frame's tracker box and every scan candidate is
cut out (a normal view around it) and embedded with DINOv2 ViT-S/14, as the appearance
verifier does.  A gallery is built causally, from what the system knew at each frame:

- positives: the frame-0 template, and the tracker's box on trusted frames whose
  relative signal is at least 0 (every ``--every``-th, the last ``--keep``);
- negatives: the tracker's box on doubted frames (every ``--every``-th, the last
  ``--keep``), and the scan candidates that were not taken.

A box's margin is its best cosine similarity to a positive minus its best similarity
to a negative (0 while there are no negatives).  Reported, against the similarity to
the frame-0 template the system uses now: the AUROC for wrong against right on trusted
frames, on releases, and on taken jumps, and how often the best candidate of a search
by margin is on the target compared with the best by template similarity.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from math import radians
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from size_study import angle, sequenceData, unit  # noqa: E402

from track360.backends.appearance import AppearanceModels, targetCrop  # noqa: E402
from track360.controller import TrackControllerImpl  # noqa: E402
from track360.controller.view_planner import localBoxOfBfov  # noqa: E402
from track360.core.config import loadConfig  # noqa: E402
from track360.core.types import BFoV, FrameIndex, FramePacket, SequenceId  # noqa: E402
from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402
from track360.geometry import SphericalGeometryImpl, makeSphericalPoint  # noqa: E402


def _bfov(yaw: float, pitch: float, width: float, height: float) -> BFoV:
    def side(deg: float) -> float:
        return min(max(radians(deg), 1e-4), np.pi - 1e-3)

    return BFoV(makeSphericalPoint(radians(yaw), radians(pitch)), side(width), side(height))


def auroc(pos, neg) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    values = np.concatenate([pos, neg])
    ranks = values.argsort().argsort() + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Gallery:
    def __init__(self, template: np.ndarray, keep: int) -> None:
        self.template = template
        self.positives: list[np.ndarray] = []
        self.negatives: list[np.ndarray] = []
        self.keep = keep

    def add(self, feature: np.ndarray, positive: bool) -> None:
        store = self.positives if positive else self.negatives
        store.append(feature)
        del store[: max(0, len(store) - self.keep)]

    def margin(self, feature: np.ndarray) -> float:
        pos = max([float(feature @ self.template)] + [float(feature @ p) for p in self.positives])
        if not self.negatives:
            return pos - 0.0
        return pos - max(float(feature @ n) for n in self.negatives)


def replay(args) -> list[dict]:
    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    models = AppearanceModels(Path(config.model.weights).parent / "hub", ("dinov2",))
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
            geometry = SphericalGeometryImpl(
                boundarySamplesPerEdge=config.geometry.boundarySamplesPerEdge
            )
            controller = TrackControllerImpl(geometry, config)
            planner = controller._planner

            def embed(frame, target: BFoV) -> np.ndarray:
                spec = planner.probeView(
                    target.center, target.horizontalFovRad, target.verticalFovRad, 1.0, 0
                )
                view = geometry.cropViews(frame, [spec])[0]
                feature = models.embed(targetCrop(view.rgb, localBoxOfBfov(spec, target)))
                return feature["dinov2"].cpu().numpy().reshape(-1)

            try:
                first = FramePacket(SequenceId(name), FrameIndex(0), 0, sequence.readRgb(0))
                plan = controller.buildInitialization(first, initialBfov=sequence.initialBfov())
                template = geometry.cropViews(first, [plan.templateView])[0]
                ref = models.embed(targetCrop(template.rgb, plan.templateBox))
                gallery = Gallery(ref["dinov2"].cpu().numpy().reshape(-1), args.keep)
                count = 0
                for t in range(1, min(len(boxes), len(d["present"]))):
                    r = trace.get(t)
                    if r is None:
                        continue
                    frame = FramePacket(SequenceId(name), FrameIndex(t), t, sequence.readRgb(t))
                    mine = embed(frame, _bfov(*boxes[t][:4]))
                    present = bool(d["present"][t])
                    right = present and d["iou"][t] >= 0.5
                    wrong = (not present) or d["iou"][t] < 0.1
                    row = {
                        "run": root.name,
                        "sequence": name,
                        "frame": t,
                        "right": bool(right),
                        "wrong": bool(wrong),
                        "modeBefore": r["modeBefore"],
                        "modeAfter": r["modeAfter"],
                        "reason": r["reason"],
                        "action": r["action"],
                        "template": float(mine @ gallery.template),
                        "margin": gallery.margin(mine),
                        "candidates": [],
                    }
                    if r["candidates"] not in ("", "[]"):
                        for c in json.loads(r["candidates"]):
                            place = _bfov(c["yawDeg"], c["pitchDeg"], c["sizeDeg"], c["sizeDeg"])
                            feature = embed(frame, place)
                            on = present and float(
                                angle(
                                    unit(c["yawDeg"], c["pitchDeg"]),
                                    unit(truth[t, 0], truth[t, 1]),
                                )
                            ) < 0.5 * max(
                                float(c["sizeDeg"]),
                                float(np.sqrt(max(truth[t, 2] * truth[t, 3], 1e-6))),
                            )
                            row["candidates"].append(
                                {
                                    "score": float(c["score"]),
                                    "verdict": c["verdict"],
                                    "onTarget": bool(on),
                                    "template": float(feature @ gallery.template),
                                    "margin": gallery.margin(feature),
                                }
                            )
                            if c["verdict"] != "accepted":
                                gallery.add(feature, positive=False)
                    out.append(row)
                    # The gallery learns from this frame only after it was scored.
                    if t % args.every == 0 and r["action"] != "jump":
                        relative = r["relative"]
                        if (
                            r["modeAfter"] == "TRACKING"
                            and relative not in ("", "None")
                            and float(relative) >= 0.0
                        ):
                            gallery.add(mine, positive=True)
                        elif r["modeAfter"] in ("UNCERTAIN", "LOST"):
                            gallery.add(mine, positive=False)
                    count += 1
                    release = getattr(geometry, "releaseFrame", None)
                    if callable(release):
                        release()
            finally:
                sequence.close()
            print(f"  {root.name} {name}: {count} frames", flush=True)
    return out


def report(rows: list[dict]) -> str:
    lines = []

    def compare(label: str, part: list[dict]) -> None:
        bad = [r for r in part if r["wrong"]]
        good = [r for r in part if r["right"]]
        lines.append(
            f"  {label}: right {len(good)}, wrong {len(bad)} | AUROC (low = wrong):"
            f" template {auroc([-r['template'] for r in bad], [-r['template'] for r in good]):.3f}"
            f"  margin {auroc([-r['margin'] for r in bad], [-r['margin'] for r in good]):.3f}"
        )

    compare("trusted frames", [r for r in rows if r["modeAfter"] == "TRACKING"])
    compare("releases", [r for r in rows if r["reason"] == "RELEASED"])
    jumps = []
    searches = []
    for r in rows:
        cands = r["candidates"]
        for c in cands:
            if c["verdict"] == "accepted":
                jumps.append(c)
        strong = [c for c in cands if c["score"] >= 0.70]
        if len(strong) >= 2 and any(c["onTarget"] for c in strong):
            searches.append(strong)
    good = [c for c in jumps if c["onTarget"]]
    bad = [c for c in jumps if not c["onTarget"]]
    lines.append(
        f"  taken jumps: on target {len(good)}, elsewhere {len(bad)} | AUROC (low = elsewhere):"
        f" template {auroc([-c['template'] for c in bad], [-c['template'] for c in good]):.3f}"
        f"  margin {auroc([-c['margin'] for c in bad], [-c['margin'] for c in good]):.3f}"
    )
    for floor in (0.0, -0.05, -0.1):
        kept = [c for c in jumps if c["margin"] >= floor]
        lines.append(
            f"    taking only margin >= {floor:+.2f}: kept {len(kept)} of {len(jumps)},"
            f" on target {sum(c['onTarget'] for c in kept)} of {len(good)}"
        )
    if searches:
        byTemplate = np.mean([max(s, key=lambda c: c["template"])["onTarget"] for s in searches])
        byMargin = np.mean([max(s, key=lambda c: c["margin"])["onTarget"] for s in searches])
        lines.append(
            f"  searches with two or more candidates >= 0.70, one on the target ({len(searches)}):"
            f" best by template on target {byTemplate:.1%}, best by margin {byMargin:.1%}"
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
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--keep", type=int, default=50)
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
