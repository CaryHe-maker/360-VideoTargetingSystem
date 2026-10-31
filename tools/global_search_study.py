"""Can a one-shot detector find a lost target anywhere on the sphere? (evaluation log E053)

    python tools/global_search_study.py --dataset-root <dir> --label-root <dir> \\
        --run outputs/E043_base2_s1 --camera outputs/E053/camera --every 2 \\
        --out outputs/E053/global

Offline, on the enlarged searches of a recorded base-2 run (target in the picture):
the whole frame is cut into six 90-degree perspective faces (a cube map, 960 pixels,
about the resolution of the frame), OWLv2 (google/owlv2-base-patch16-ensemble,
Apache-2.0, local weights) looks in every face for the frame-0 target given as an
image, and the detections are ranked by score over the sphere.  A detection hits when
its centre is within one target size of the target.  Reported: the share of searches
with a hit among the first 1, 3, 5, 7 detections, by the class of the search
(``search_place_study``), and the time of one global search.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_place_study as sp  # noqa: E402

from track360.datasets.vot360 import Vot360Dataset  # noqa: E402

FACE = 960
TOPS = (1, 3, 5, 7)
# Yaw and pitch of the six cube faces, degrees.
FACES = ((0, 0), (90, 0), (180, 0), (-90, 0), (0, 90), (0, -90))


def basis(yawDeg: float, pitchDeg: float) -> np.ndarray:
    """Rows: right, up, forward of a view looking at (yaw, pitch)."""
    yaw, pitch = np.radians(yawDeg), np.radians(pitchDeg)
    forward = np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
    worldUp = np.array([0.0, 1.0, 0.0]) if abs(pitchDeg) < 89 else np.array([0.0, 0.0, -1.0])
    right = np.cross(worldUp, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.stack([right, up, forward])


_MAPS: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def perspective(rgb: np.ndarray, axes: np.ndarray, fovDeg: float, size: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    key = (width, height, fovDeg, size, axes.tobytes())
    if key not in _MAPS:
        half = np.tan(np.radians(fovDeg) / 2.0)
        s = (np.arange(size) + 0.5) / size * 2.0 - 1.0
        u, v = np.meshgrid(s * half, -s * half)
        d = u[..., None] * axes[0] + v[..., None] * axes[1] + axes[2]
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        yaw = np.arctan2(d[..., 0], d[..., 2])
        pitch = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
        mapX = ((yaw + np.pi) / (2.0 * np.pi) * width - 0.5).astype(np.float32)
        mapY = ((np.pi / 2.0 - pitch) / np.pi * height - 0.5).astype(np.float32)
        _MAPS[key] = cv2.convertMaps(mapX, mapY, cv2.CV_16SC2)
    first, second = _MAPS[key]
    return cv2.remap(rgb, first, second, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def faceDirection(axes: np.ndarray, fovDeg: float, x: float, y: float, size: int) -> np.ndarray:
    half = np.tan(np.radians(fovDeg) / 2.0)
    u = ((x / size) * 2.0 - 1.0) * half
    v = -((y / size) * 2.0 - 1.0) * half
    d = u * axes[0] + v * axes[1] + axes[2]
    return d / np.linalg.norm(d)


class Detector:
    """OWLv2 image-guided detection with the query embedded once and faces batched."""

    def __init__(self, batch: int = 3) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        name = "google/owlv2-base-patch16-ensemble"
        processor = Owlv2Processor.from_pretrained(name)
        self.model = Owlv2ForObjectDetection.from_pretrained(name).cuda().eval().half()
        self.mean = torch.tensor(processor.image_processor.image_mean, device="cuda").view(
            1, 3, 1, 1
        )
        self.std = torch.tensor(processor.image_processor.image_std, device="cuda").view(1, 3, 1, 1)
        self.batch = batch
        self.query: torch.Tensor | None = None

    def _pixels(self, images: list[np.ndarray]) -> torch.Tensor:
        x = torch.from_numpy(np.stack(images)).cuda().permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != FACE:
            x = torch.nn.functional.interpolate(x, size=(FACE, FACE), mode="bilinear")
        return ((x - self.mean) / self.std).half()

    @torch.inference_mode()
    def setQuery(self, query: np.ndarray) -> None:
        feature_map = self.model.image_embedder(pixel_values=self._pixels([query]))[0]
        b, h, w, d = feature_map.shape
        feats = feature_map.reshape(b, h * w, d)
        self.query, _, _ = self.model.embed_image_query(feats, feature_map)

    @torch.inference_mode()
    def detect(
        self, faces: list[np.ndarray], perFace: int = 30
    ) -> list[tuple[int, float, float, float]]:
        """(face, x, y, score) of the best detections of every face, in face pixels."""
        assert self.query is not None
        out = []
        for start in range(0, len(faces), self.batch):
            part = faces[start : start + self.batch]
            feature_map = self.model.image_embedder(pixel_values=self._pixels(part))[0]
            b, h, w, d = feature_map.shape
            feats = feature_map.reshape(b, h * w, d)
            logits, _ = self.model.class_predictor(feats, self.query.expand(b, -1, -1), None)
            boxes = self.model.box_predictor(feats, feature_map)
            scores = torch.sigmoid(logits[..., 0].float())
            top = torch.topk(scores, perFace, dim=1)
            for k in range(b):
                for score, index in zip(
                    top.values[k].tolist(), top.indices[k].tolist(), strict=True
                ):
                    cx, cy = boxes[k, index, 0].item(), boxes[k, index, 1].item()
                    out.append((start + k, cx * FACE, cy * FACE, float(score)))
        return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--camera", type=Path, default=None)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    detector = Detector()
    axesOf = [basis(y, p) for y, p in FACES]
    queries: dict[str, np.ndarray] = {}
    queryOf = None
    rows = []
    timings = []
    items = list(sp.searches(args, args.run, dataset))[:: args.every]
    print(f"{len(items)} searches", flush=True)
    for number, item in enumerate(items, 1):
        name, frame = item["sequence"], item["frame"]
        sequence = dataset.sequence(name)
        if name not in queries:
            bfov, _ = sequence.groundTruth("bfov")
            first = sequence.readRgb(0)
            size = max(float(np.sqrt(bfov[0, 2] * bfov[0, 3])), 1.0)
            axes = basis(bfov[0, 0], bfov[0, 1])
            queries[name] = perspective(first, axes, min(1.3 * size, 150.0), 224)
        rgb = sequence.readRgb(frame)
        sequence.close()
        started = time.perf_counter()
        faces = [perspective(rgb, axes, 90.0, FACE) for axes in axesOf]
        if detector.query is None or queryOf != name:
            detector.setQuery(queries[name])
            queryOf = name
        found = detector.detect(faces)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        found.sort(key=lambda d: -d[3])
        truth = item["truth"]
        size = item["targetSize"]
        # Keep detections at least one trusted target size apart (system information).
        apart = max(float(item["trace"].size[item["trusted"]]), 1.0)
        kept: list[np.ndarray] = []
        for face, x, y, score in found:
            d = faceDirection(axesOf[face], 90.0, x, y, FACE)
            if all(sp.angle(d, other) >= apart for other in kept):
                kept.append(d)
            if len(kept) >= max(TOPS):
                break
        hitsAt = [sp.angle(d, truth) < size for d in kept]
        trace = item["trace"]
        centre = trace.pos[item["trusted"]]
        if sp.angle(centre, truth) <= item["half"]:
            kind = "in view"
        elif item["good"] is None:
            kind = "no good frame"
        elif sp.angle(item["truthAt"](item["good"]), truth) <= item["half"]:
            kind = "drift"
        else:
            kind = "far"
        rows.append(
            {
                "sequence": name,
                "frame": frame,
                "class": kind,
                "targetSizeDeg": size,
                "hits": hitsAt,
                "topScore": found[0][3] if found else None,
            }
        )
        if number % 50 == 0:
            print(
                f"  {number}/{len(items)}  {np.mean(timings) * 1000:.0f} ms per global search",
                flush=True,
            )
    lines = [f"{len(rows)} searches; one global search {np.median(timings) * 1000:.0f} ms (median)"]

    def table(label: str, part: list[dict]) -> None:
        if not part:
            return
        shares = [np.mean([any(r["hits"][:k]) for r in part]) for k in TOPS]
        lines.append(
            f"  {label:<26} n={len(part):>4}  top 1/3/5/7: "
            + " / ".join(f"{s:.1%}" for s in shares)
        )

    table("all", rows)
    for kind in ("in view", "drift", "far", "no good frame"):
        table(kind, [r for r in rows if r["class"] == kind])
    for low, high in ((0, 3), (3, 6), (6, 12), (12, 400)):
        table(f"target {low}-{high} deg", [r for r in rows if low <= r["targetSizeDeg"] < high])
    text = "\n".join(lines)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.txt").write_text(text + "\n", encoding="utf-8")
    (args.out / "rows.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "models" / "hf"))
    raise SystemExit(main())
