"""Look for a lost target on the whole sphere with a one-shot detector (experimental, E054).

The frame is cut into six 90-degree perspective faces (a cube map at about the frame's
own resolution) and OWLv2 (google/owlv2-base-patch16-ensemble, Apache-2.0, weights
kept locally under ``models/hf``) looks in every face for the frame-0 target, given as
an image.  The best detections over the sphere, kept apart by at least one target
size, are the places where the tracker then takes a normal view (E053: the target was
among the first three for 43% of the searches, 48% of those the zoom missed).

Needs the optional ``transformers`` package.
"""

from __future__ import annotations

import os
from math import asin, atan2, degrees, radians, sqrt, tan
from pathlib import Path

import cv2
import numpy as np

from track360.core.types import BFoV, SphericalPoint
from track360.geometry import makeSphericalPoint

FACE = 960
MODEL = "google/owlv2-base-patch16-ensemble"
# Yaw and pitch of the six cube faces, degrees.
FACES = ((0, 0), (90, 0), (180, 0), (-90, 0), (0, 90), (0, -90))


def _basis(yawDeg: float, pitchDeg: float) -> np.ndarray:
    """Rows: right, up, forward of a view looking at (yaw, pitch)."""
    yaw, pitch = radians(yawDeg), radians(pitchDeg)
    forward = np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
    worldUp = np.array([0.0, 1.0, 0.0]) if abs(pitchDeg) < 89 else np.array([0.0, 0.0, -1.0])
    right = np.cross(worldUp, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.stack([right, up, forward])


class GlobalSearcher:
    def __init__(self, modelRoot: str | Path, batch: int = 3) -> None:
        os.environ.setdefault("HF_HOME", str(Path(modelRoot)))
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self._torch = torch
        processor = Owlv2Processor.from_pretrained(MODEL)
        self._model = Owlv2ForObjectDetection.from_pretrained(MODEL).cuda().eval().half()
        self._mean = torch.tensor(processor.image_processor.image_mean, device="cuda").view(
            1, 3, 1, 1
        )
        self._std = torch.tensor(processor.image_processor.image_std, device="cuda").view(
            1, 3, 1, 1
        )
        self._batch = batch
        self._axes = [_basis(y, p) for y, p in FACES]
        self._maps: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self._query = None

    # ---------------------------------------------------------------- images
    def _perspective(self, rgb: np.ndarray, axes: np.ndarray, fovDeg: float, size: int):
        height, width = rgb.shape[:2]
        key = (width, height, round(fovDeg, 4), size, axes.tobytes())
        if key not in self._maps:
            half = tan(radians(fovDeg) / 2.0)
            s = (np.arange(size) + 0.5) / size * 2.0 - 1.0
            u, v = np.meshgrid(s * half, -s * half)
            d = u[..., None] * axes[0] + v[..., None] * axes[1] + axes[2]
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            yaw = np.arctan2(d[..., 0], d[..., 2])
            pitch = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
            mapX = ((yaw + np.pi) / (2.0 * np.pi) * width - 0.5).astype(np.float32)
            mapY = ((np.pi / 2.0 - pitch) / np.pi * height - 0.5).astype(np.float32)
            maps = cv2.convertMaps(mapX, mapY, cv2.CV_16SC2)
            if len(self._maps) > 32:
                self._maps.clear()
            self._maps[key] = maps
        first, second = self._maps[key]
        return cv2.remap(rgb, first, second, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def _pixels(self, images: list[np.ndarray]):
        torch = self._torch
        x = torch.from_numpy(np.stack(images)).cuda().permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != FACE:
            x = torch.nn.functional.interpolate(x, size=(FACE, FACE), mode="bilinear")
        return ((x - self._mean) / self._std).half()

    # ---------------------------------------------------------------- use
    def begin(self, rgb: np.ndarray, target: BFoV) -> None:
        """Take the frame-0 target as the query image."""
        torch = self._torch
        c = target.center
        axes = _basis(degrees(atan2(c.x, c.z)), degrees(asin(max(-1.0, min(1.0, c.y)))))
        size = max(degrees(sqrt(target.horizontalFovRad * target.verticalFovRad)), 1.0)
        query = self._perspective(rgb, axes, min(1.3 * size, 150.0), 224)
        with torch.inference_mode():
            featureMap = self._model.image_embedder(pixel_values=self._pixels([query]))[0]
            b, h, w, d = featureMap.shape
            self._query, _, _ = self._model.embed_image_query(
                featureMap.reshape(b, h * w, d), featureMap
            )

    def search(self, rgb: np.ndarray, count: int, apartRad: float) -> list[SphericalPoint]:
        """The ``count`` best places on the sphere, at least ``apartRad`` apart."""
        torch = self._torch
        assert self._query is not None, "begin() first"
        faces = [self._perspective(rgb, axes, 90.0, FACE) for axes in self._axes]
        found: list[tuple[float, np.ndarray]] = []
        with torch.inference_mode():
            for start in range(0, len(faces), self._batch):
                part = faces[start : start + self._batch]
                featureMap = self._model.image_embedder(pixel_values=self._pixels(part))[0]
                b, h, w, d = featureMap.shape
                feats = featureMap.reshape(b, h * w, d)
                logits, _ = self._model.class_predictor(feats, self._query.expand(b, -1, -1), None)
                boxes = self._model.box_predictor(feats, featureMap)
                scores = torch.sigmoid(logits[..., 0].float())
                top = torch.topk(scores, 30, dim=1)
                centres = (
                    torch.gather(
                        boxes[..., :2].float(), 1, top.indices[..., None].expand(-1, -1, 2)
                    )
                    .cpu()
                    .numpy()
                )
                values = top.values.cpu().numpy()
                for k in range(b):
                    axes = self._axes[start + k]
                    for (cx, cy), score in zip(centres[k], values[k], strict=True):
                        u, v = (2.0 * cx - 1.0), -(2.0 * cy - 1.0)  # tan(45 deg) = 1
                        direction = u * axes[0] + v * axes[1] + axes[2]
                        found.append((float(score), direction / np.linalg.norm(direction)))
        found.sort(key=lambda item: -item[0])
        kept: list[np.ndarray] = []
        limit = np.cos(apartRad)
        for _, direction in found:
            if all(float(direction @ other) < limit for other in kept):
                kept.append(direction)
            if len(kept) >= count:
                break
        return [
            makeSphericalPoint(atan2(d[0], d[2]), asin(max(-1.0, min(1.0, float(d[1])))))
            for d in kept
        ]
