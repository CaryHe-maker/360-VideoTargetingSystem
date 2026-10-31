"""Can a second model list the target and its look-alikes around the track? (E075, offline)

    python tools/proposal_study.py --dataset-root <dir> --label-root <dir> \\
        --runs outputs/E066_s1 outputs/E066_s2 --step 6 --out outputs/E075

On every ``--step``-th frame of a recorded run (target in the picture or not) a view
``--scale`` times the normal search view is taken around the tracker's box of the
frame before, at ``--width`` pixels, and two models list candidates in it:

- dinov2: the patch features of DINOv2 ViT-S/14 compared with the mean patch feature
  inside the frame-0 target box; the local maxima of the similarity map, kept apart by
  the target's size, are the candidates (a place, no box);
- yoloe: YOLOE-11S with the frame-0 target box as its visual prompt; its boxes are the
  candidates.

A candidate is on the target when its centre lies within half the target's size of
the true centre.  Reported per model: how often the target is in the view, the share
of those frames with the target among the first 1 / 3 / 5 / 10 candidates, the number
of candidates above the model's threshold, on frames the tracker has wrong (absent or
IoU < 0.1) the share with the target first and the share with a candidate on the
tracker's box, and the time per call.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import replace
from math import radians, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from size_study import sequenceData  # noqa: E402

from track360.controller import TrackControllerImpl  # noqa: E402
from track360.controller.view_planner import localBoxOfBfov  # noqa: E402
from track360.core.config import loadConfig  # noqa: E402
from track360.core.types import BFoV, FrameIndex, FramePacket, SequenceId  # noqa: E402
from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402
from track360.geometry import (
    SphericalGeometryImpl,  # noqa: E402
    makeSphericalPoint,  # noqa: E402
)

KS = (1, 3, 5, 10)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DenseDino:
    """Local maxima of the DINOv2 patch similarity to the frame-0 target."""

    def __init__(self, hub: Path, side: int) -> None:
        import torch

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.device_count() > 0 else "cpu")
        self.model = (
            torch.hub.load(
                str(hub / "facebookresearch_dinov2_main"),
                "dinov2_vits14",
                source="local",
                verbose=False,
            )
            .to(self.device)
            .eval()
        )
        self.side = side - side % 14
        self.grid = self.side // 14
        self.prototype = None

    def _patches(self, rgb: np.ndarray):
        import cv2

        torch = self.torch
        image = cv2.resize(rgb, (self.side, self.side), interpolation=cv2.INTER_AREA)
        x = (image.astype(np.float32) / 255.0 - _MEAN) / _STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            tokens = self.model.forward_features(x)["x_norm_patchtokens"][0]
        return torch.nn.functional.normalize(tokens, dim=1)  # (grid*grid, 384)

    def setTarget(self, rgb: np.ndarray, box) -> None:
        tokens = self._patches(rgb)
        scale = self.grid / rgb.shape[1]
        cx = (np.arange(self.grid) + 0.5)[None, :].repeat(self.grid, 0)
        cy = (np.arange(self.grid) + 0.5)[:, None].repeat(self.grid, 1)
        inside = (
            (cx >= box.xPx * scale)
            & (cx <= (box.xPx + box.widthPx) * scale)
            & (cy >= box.yPx * scale)
            & (cy <= (box.yPx + box.heightPx) * scale)
        ).reshape(-1)
        if not inside.any():
            inside[np.argmin((cx - self.grid / 2) ** 2 + (cy - self.grid / 2) ** 2)] = True
        mask = self.torch.from_numpy(inside).to(self.device)
        self.prototype = self.torch.nn.functional.normalize(tokens[mask].mean(0), dim=0)

    def candidates(self, rgb: np.ndarray, sizePx: float, count: int = 10):
        """(x, y, score) in view pixels, best first."""
        torch = self.torch
        tokens = self._patches(rgb)
        sim = (tokens @ self.prototype).reshape(1, 1, self.grid, self.grid)
        radius = max(1, int(round(0.5 * sizePx * self.grid / rgb.shape[1])))
        peaks = sim == torch.nn.functional.max_pool2d(sim, 2 * radius + 1, 1, radius)
        values, order = torch.sort(sim[peaks], descending=True)
        index = torch.nonzero(peaks.reshape(-1)).reshape(-1)[order]
        out = []
        cell = rgb.shape[1] / self.grid
        for value, i in zip(values[:count].tolist(), index[:count].tolist(), strict=True):
            out.append(((i % self.grid + 0.5) * cell, (i // self.grid + 0.5) * cell, value))
        return out


class VisualPromptYoloe:
    """YOLOE with the frame-0 target as its visual prompt."""

    def __init__(self, weights: Path) -> None:
        import torch
        from ultralytics import YOLOE

        self.model = YOLOE(str(weights))
        self.device = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
        self.ready = False

    def setTarget(self, rgb: np.ndarray, box) -> None:
        prompts = {
            "bboxes": np.array(
                [[box.xPx, box.yPx, box.xPx + box.widthPx, box.yPx + box.heightPx]],
                dtype=np.float32,
            ),
            "cls": np.array([0]),
        }
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        # With a reference image the prompt becomes the model's only class; later
        # calls detect that class without the prompt.
        self.model.predict(
            bgr, visual_prompts=prompts, refer_image=bgr, verbose=False, device=self.device
        )
        self.ready = True

    def boxes(self, rgb: np.ndarray, count: int = 10):
        """(x0, y0, x1, y1, confidence) in view pixels, best first."""
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        result = self.model.predict(
            bgr, conf=0.05, max_det=count, verbose=False, imgsz=rgb.shape[1], device=self.device
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        return [(*map(float, xyxy[i]), float(conf[i])) for i in np.argsort(-conf)[:count]]

    def candidates(self, rgb: np.ndarray, sizePx: float, count: int = 10):
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        result = self.model.predict(
            bgr, conf=0.01, max_det=count, verbose=False, imgsz=rgb.shape[1], device=self.device
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        order = np.argsort(-conf)[:count]
        return [
            (
                float(0.5 * (xyxy[i, 0] + xyxy[i, 2])),
                float(0.5 * (xyxy[i, 1] + xyxy[i, 3])),
                float(conf[i]),
            )
            for i in order
        ]


def _synchronize() -> None:
    import torch

    if torch.cuda.device_count() > 0:
        torch.cuda.synchronize()


def _bfov(row) -> BFoV:
    def side(deg: float) -> float:
        return min(max(radians(deg), 1e-4), np.pi - 1e-3)

    return BFoV(makeSphericalPoint(radians(row[0]), radians(row[1])), side(row[2]), side(row[3]))


def replay(args) -> list[dict]:
    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    hub = Path(config.model.weights).parent / "hub"
    models = {}
    if "dinov2" in args.models:
        models["dinov2"] = DenseDino(hub, args.dino_side)
    if "yoloe" in args.models:
        models["yoloe"] = VisualPromptYoloe(args.yoloe)
    rows: list[dict] = []
    for root in args.runs:
        results = loadTrackerResults(root / "bbox" / "ours")
        names = readSequenceFile(args.sequence_file)
        if args.sequences:
            names = [n for n in names if n in args.sequences.split(",")]
        for name in names:
            d = sequenceData(dataset, root, name, results)
            if d is None:
                continue
            boxes = np.loadtxt(root / "bfov" / "ours" / f"{name}.txt", delimiter=",", ndmin=2)
            sequence = dataset.sequence(name)
            truth, _ = sequence.groundTruth("bfov")
            geometry = SphericalGeometryImpl(
                boundarySamplesPerEdge=config.geometry.boundarySamplesPerEdge
            )
            planner = TrackControllerImpl(geometry, config)._planner
            try:

                def view(frame, around: BFoV):
                    spec = planner.probeView(
                        around.center,
                        around.horizontalFovRad,
                        around.verticalFovRad,
                        args.scale,
                        0,
                    )
                    spec = replace(
                        spec, outputWidthPx=args.width, outputHeightPx=args.width, priorBox=None
                    )
                    return spec, geometry.cropViews(frame, [spec])[0].rgb

                first = FramePacket(SequenceId(name), FrameIndex(0), 0, sequence.readRgb(0))
                target = sequence.initialBfov()
                spec0, rgb0 = view(first, target)
                box0 = localBoxOfBfov(spec0, target)
                for model in models.values():
                    model.setTarget(rgb0, box0)
                count = 0
                for t in range(args.step, min(len(boxes), len(d["present"])), args.step):
                    frame = FramePacket(SequenceId(name), FrameIndex(t), t, sequence.readRgb(t))
                    spec, rgb = view(frame, _bfov(boxes[t - 1]))
                    present = bool(d["present"][t])
                    row = {
                        "run": root.name,
                        "sequence": name,
                        "frame": t,
                        "present": present,
                        "iouT": float(d["iou"][t]),
                    }
                    here = None
                    if present:
                        gt = localBoxOfBfov(spec, _bfov(truth[t]))
                        here = (
                            gt.xPx + 0.5 * gt.widthPx,
                            gt.yPx + 0.5 * gt.heightPx,
                            max(sqrt(gt.widthPx * gt.heightPx), 4.0),
                        )
                        row["inView"] = bool(
                            0 <= here[0] < args.width and 0 <= here[1] < args.width
                        )
                    tracker = localBoxOfBfov(spec, _bfov(boxes[t]))
                    mine = (
                        tracker.xPx + 0.5 * tracker.widthPx,
                        tracker.yPx + 0.5 * tracker.heightPx,
                        max(sqrt(tracker.widthPx * tracker.heightPx), 4.0),
                    )
                    for key, model in models.items():
                        started = time.perf_counter()
                        found = model.candidates(rgb, mine[2])
                        _synchronize()
                        row[f"{key}Ms"] = 1000.0 * (time.perf_counter() - started)
                        row[f"{key}Scores"] = [round(c[2], 4) for c in found]
                        row[f"{key}OnTarget"] = (
                            [
                                sqrt((c[0] - here[0]) ** 2 + (c[1] - here[1]) ** 2) < 0.5 * here[2]
                                for c in found
                            ]
                            if here is not None
                            else [False] * len(found)
                        )
                        row[f"{key}OnTracker"] = [
                            sqrt((c[0] - mine[0]) ** 2 + (c[1] - mine[1]) ** 2) < 0.5 * mine[2]
                            for c in found
                        ]
                    rows.append(row)
                    count += 1
                    release = getattr(geometry, "releaseFrame", None)
                    if callable(release):
                        release()
            finally:
                sequence.close()
            print(f"  {root.name} {name}: {count} frames", flush=True)
    return rows


def report(rows: list[dict], keys: list[str]) -> str:
    seen = [r for r in rows if r.get("inView")]
    wrong = [r for r in rows if (not r["present"]) or r["iouT"] < 0.1]
    wrongSeen = [r for r in wrong if r.get("inView")]
    lines = [
        f"{len(rows)} frames; target in the view {len(seen)}; tracker wrong {len(wrong)}"
        f" (target in the view on {len(wrongSeen)})"
    ]
    for key in keys:
        if not rows or f"{key}Scores" not in rows[0]:
            continue
        recall = [
            np.mean([any(r[f"{key}OnTarget"][:k]) for r in seen]) if seen else float("nan")
            for k in KS
        ]
        ms = np.array([r[f"{key}Ms"] for r in rows])
        lines.append(
            f"  {key}: target among the first "
            + " / ".join(f"{k}: {v:.1%}" for k, v in zip(KS, recall, strict=True))
            + f" | mean candidates {np.mean([len(r[f'{key}Scores']) for r in rows]):.1f}"
            + f" | ms median {np.median(ms):.1f} p95 {np.percentile(ms, 95):.1f}"
        )
        if wrongSeen:
            first = np.mean(
                [bool(r[f"{key}OnTarget"][:1] and r[f"{key}OnTarget"][0]) for r in wrongSeen]
            )
            anyTop5 = np.mean([any(r[f"{key}OnTarget"][:5]) for r in wrongSeen])
            onTracker = np.mean([any(r[f"{key}OnTracker"][:5]) for r in wrongSeen])
            lines.append(
                f"    tracker wrong, target in the view: target first {first:.1%},"
                f" among the first 5 {anyTop5:.1%};"
                f" a candidate on the tracker box {onTracker:.1%}"
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
    parser.add_argument("--sequences", default=None, help="comma-separated subset")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "second_look.yaml",
    )
    parser.add_argument("--models", default="dinov2,yoloe")
    parser.add_argument(
        "--yoloe",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "yoloe" / "yoloe-11s-seg.pt",
    )
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--dino-side", type=int, default=644)
    parser.add_argument("--step", type=int, default=6)
    parser.add_argument("--rows", type=Path, default=None, help="report on saved rows only")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.models = args.models.split(",")
    warnings.filterwarnings("ignore")
    if args.rows is not None:
        rows = json.loads(args.rows.read_text(encoding="utf-8"))
    else:
        rows = replay(args)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "rows.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    text = report(rows, args.models)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
