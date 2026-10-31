"""Frozen image models that compare a tracked box with the frame-0 template.

The tracker's own score says how well a box is placed, and it stays high after the
tracker moved onto another object.  A feature similarity between the current box and
the template asks the other question: is this still the same thing?
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from track360.core.types import BBoxXYWH, LocalView

MODEL_NAMES = ("dinov2", "dino", "resnet18")
CROP_SIZE = 224
# The crop is a square this many times the longer side of the box.
CROP_MARGIN = 1.1
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def targetCrop(
    rgb: NDArray[np.uint8], box: BBoxXYWH, size: int = CROP_SIZE, margin: float = CROP_MARGIN
) -> NDArray[np.uint8]:
    """A square crop around ``box`` that keeps its aspect; outside the image is black."""
    import cv2

    side = max(2, int(round(max(box.widthPx, box.heightPx) * margin)))
    x0 = int(round(box.xPx + 0.5 * box.widthPx - 0.5 * side))
    y0 = int(round(box.yPx + 0.5 * box.heightPx - 0.5 * side))
    x1, y1 = x0 + side, y0 + side
    height, width = rgb.shape[:2]
    left, top = max(0, -x0), max(0, -y0)
    right, bottom = max(0, x1 - width), max(0, y1 - height)
    crop = rgb[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    if left or top or right or bottom:
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT)
    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interpolation)


def histogramSimilarity(first: NDArray[np.uint8], second: NDArray[np.uint8]) -> float:
    """1 minus the Bhattacharyya distance of the hue-saturation histograms."""
    import cv2

    def histogram(image: NDArray[np.uint8]) -> NDArray[np.float32]:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        values = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        return cv2.normalize(values, values, alpha=1.0, norm_type=cv2.NORM_L1)

    distance = cv2.compareHist(histogram(first), histogram(second), cv2.HISTCMP_BHATTACHARYYA)
    return float(1.0 - distance)


class AppearanceModels:
    """The frozen image models, loaded once from ``hubDirectory``."""

    def __init__(self, hubDirectory: str | Path, names: Sequence[str] = MODEL_NAMES) -> None:
        import torch
        import torchvision

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.hub.set_dir(str(hubDirectory))
        localDinov2 = Path(hubDirectory) / "facebookresearch_dinov2_main"
        loaders = {
            "dinov2": lambda: (
                # The copy already in the hub directory needs no network.
                torch.hub.load(str(localDinov2), "dinov2_vits14", source="local", verbose=False)
                if localDinov2.is_dir()
                else torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True, verbose=False
                )
            ),
            "dino": lambda: torch.hub.load(
                "facebookresearch/dino:main", "dino_vits16", trust_repo=True, verbose=False
            ),
            "resnet18": lambda: _headless(
                torchvision.models.resnet18(
                    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                ),
                torch,
            ),
        }
        self._models = {name: loaders[name]().to(self._device).eval() for name in names}
        self._mean = torch.tensor(_MEAN, device=self._device).view(1, 3, 1, 1)
        self._std = torch.tensor(_STD, device=self._device).view(1, 3, 1, 1)

    @property
    def torch(self) -> Any:
        return self._torch

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._models)

    def embed(self, crop: NDArray[np.uint8]) -> dict[str, Any]:
        """Unit-length feature of one RGB crop from every model."""
        torch = self._torch
        image = torch.from_numpy(np.ascontiguousarray(crop)).to(self._device)
        image = image.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image = (image - self._mean) / self._std
        features = {}
        with torch.inference_mode():
            for name, model in self._models.items():
                feature = model(image).reshape(1, -1)
                features[name] = torch.nn.functional.normalize(feature, dim=1)
        return features

    def similarity(self, first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
        return {name: float((first[name] * second[name]).sum().item()) for name in first}


def _headless(model: Any, torch: Any) -> Any:
    model.fc = torch.nn.Identity()
    return model


_SHARED: dict[tuple[str, str], AppearanceModels] = {}


class AppearanceVerifier:
    """Similarity of a box in a local view to the frame-0 template of one sequence.

    The template is the only reference and it never changes: a reference that follows
    the tracked boxes also follows the tracker onto a wrong object (evaluation log E019).
    """

    def __init__(self, model: str, hubDirectory: str | Path) -> None:
        if model not in MODEL_NAMES:
            raise ValueError(f"unknown appearance model '{model}'")
        key = (model, str(hubDirectory))
        # Loading a model takes seconds; one instance serves every sequence of a run.
        if key not in _SHARED:
            _SHARED[key] = AppearanceModels(hubDirectory, (model,))
        self._models = _SHARED[key]
        self._model = model
        self._template: Any | None = None
        self._lastFeature: Any | None = None
        # Experimental (E076): what this sequence's target looks like and what it does not.
        self._galleryKeep = 0
        self._gallery: AppearanceGallery | None = None

    def useGallery(self, keep: int) -> None:
        """Experimental (E076): keep a gallery of ``keep`` positives and negatives."""
        self._galleryKeep = int(keep)

    def setTemplate(self, template: LocalView, templateBox: BBoxXYWH) -> None:
        self._template = self._models.embed(targetCrop(template.rgb, templateBox))[self._model]
        if self._galleryKeep > 0:
            self._gallery = AppearanceGallery(self._template, self._galleryKeep)
            self._prototype = self._patchPrototype(template.rgb, templateBox)

    def embedBox(self, view: LocalView, box: BBoxXYWH) -> Any:
        """The unit feature of a box (E080); does not change ``lastFeature``."""
        return self._models.embed(targetCrop(view.rgb, box))[self._model]

    def _patches(self, rgb: Any, side: int) -> Any:
        import cv2

        torch = self._models.torch
        model = self._models._models[self._model]
        side -= side % 14
        image = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(image)).to(self._models._device)
        x = x.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = (x - self._models._mean) / self._models._std
        with torch.inference_mode():
            tokens = model.forward_features(x)["x_norm_patchtokens"][0]
        return torch.nn.functional.normalize(tokens, dim=1), side // 14

    def _patchPrototype(self, rgb: Any, box: BBoxXYWH, side: int = 448) -> Any:
        tokens, grid = self._patches(rgb, side)
        scale = grid / rgb.shape[1]
        centres = (np.arange(grid) + 0.5) / scale
        cx, cy = np.meshgrid(centres, centres)
        inside = (
            (cx >= box.xPx)
            & (cx <= box.xPx + box.widthPx)
            & (cy >= box.yPx)
            & (cy <= box.yPx + box.heightPx)
        ).reshape(-1)
        if not inside.any():
            inside[
                np.argmin(
                    (cx - box.xPx - 0.5 * box.widthPx) ** 2
                    + (cy - box.yPx - 0.5 * box.heightPx) ** 2
                )
            ] = True  # noqa: E501
        mask = self._models.torch.from_numpy(inside).to(tokens.device)
        return self._models.torch.nn.functional.normalize(tokens[mask].mean(0), dim=0)

    def lookalikes(
        self, rgb: Any, box: BBoxXYWH, count: int, apart: float, side: int = 448
    ) -> list[tuple[float, float]]:
        """E080: centres (view pixels) of the places in ``rgb`` most like the frame-0
        target by DINOv2 patch features, at least ``apart`` box sizes from ``box``."""
        torch = self._models.torch
        tokens, grid = self._patches(rgb, side)
        sim = (tokens @ self._prototype).reshape(1, 1, grid, grid)
        size = max(sqrt(box.widthPx * box.heightPx), 4.0)
        radius = max(1, int(round(0.5 * size * grid / rgb.shape[1])))
        peaks = sim == torch.nn.functional.max_pool2d(sim, 2 * radius + 1, 1, radius)
        values, order = torch.sort(sim[peaks], descending=True)
        index = torch.nonzero(peaks.reshape(-1)).reshape(-1)[order]
        cell = rgb.shape[1] / grid
        cx, cy = box.xPx + 0.5 * box.widthPx, box.yPx + 0.5 * box.heightPx
        out = []
        for i in index[:20].tolist():
            x, y = (i % grid + 0.5) * cell, (i // grid + 0.5) * cell
            if sqrt((x - cx) ** 2 + (y - cy) ** 2) < apart * size:
                continue
            out.append((x, y))
            if len(out) >= count:
                break
        return out

    @property
    def lastFeature(self) -> Any | None:
        """The feature of the box of the last ``similarity`` call."""
        return self._lastFeature

    @property
    def gallery(self) -> AppearanceGallery | None:
        return self._gallery

    def similarity(self, view: LocalView, box: BBoxXYWH) -> float | None:
        """How much the box looks like the target of frame 0."""
        if self._template is None:
            return None
        feature = self._models.embed(targetCrop(view.rgb, box))[self._model]
        self._lastFeature = feature
        return float((feature * self._template).sum().item())


class AppearanceGallery:
    """Experimental (E076): the frame-0 template and the latest boxes of this sequence
    known to be the target (positives) or not (negatives).

    A box's margin is its best similarity to a positive minus its best similarity to a
    negative; without negatives it is the best similarity to a positive.
    """

    def __init__(self, template: Any, keep: int) -> None:
        self._template = template
        self._keep = keep
        self._positives: list[Any] = []
        # Negatives by source (E080): "doubt" are boxes of doubted frames, "lookalike"
        # look-alikes found around a confidently tracked target.
        self._negatives: dict[str, list[Any]] = {"doubt": [], "lookalike": []}
        # Which negatives a scan candidate is compared with.
        self.candidateSources: tuple[str, ...] = ("doubt",)

    def add(self, feature: Any, positive: bool, source: str = "doubt") -> None:
        store = self._positives if positive else self._negatives[source]
        store.append(feature)
        del store[: max(0, len(store) - self._keep)]

    def margin(self, feature: Any, sources: tuple[str, ...] = ("doubt",)) -> float:
        import torch

        positives = torch.cat([self._template, *self._positives], dim=0)
        best = float((positives @ feature.reshape(-1)).max().item())
        negatives = [n for source in sources for n in self._negatives[source]]
        if not negatives:
            return best
        stacked = torch.cat(negatives, dim=0)
        return best - float((stacked @ feature.reshape(-1)).max().item())


__all__ = [
    "MODEL_NAMES",
    "AppearanceGallery",
    "AppearanceModels",
    "AppearanceVerifier",
    "histogramSimilarity",
    "targetCrop",
]
