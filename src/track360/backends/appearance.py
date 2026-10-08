"""Frozen image models that compare a tracked box with the frame-0 template.

The tracker's own score says how well a box is placed, and it stays high after the
tracker moved onto another object.  A feature similarity between the current box and
the template asks the other question: is this still the same thing?
"""

from __future__ import annotations

from collections.abc import Sequence
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
        loaders = {
            "dinov2": lambda: torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True, verbose=False
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
    """Similarity of a box in a local view to the template of one sequence."""

    def __init__(self, model: str, hubDirectory: str | Path) -> None:
        if model not in MODEL_NAMES:
            raise ValueError(f"unknown appearance model '{model}'")
        key = (model, str(hubDirectory))
        # Loading a model takes seconds; one instance serves every sequence of a run.
        if key not in _SHARED:
            _SHARED[key] = AppearanceModels(hubDirectory, (model,))
        self._models = _SHARED[key]
        self._model = model
        self._template: dict[str, Any] | None = None

    def setTemplate(self, template: LocalView, templateBox: BBoxXYWH) -> None:
        self._template = self._models.embed(targetCrop(template.rgb, templateBox))

    def similarity(self, view: LocalView, box: BBoxXYWH) -> float | None:
        if self._template is None:
            return None
        features = self._models.embed(targetCrop(view.rgb, box))
        return self._models.similarity(self._template, features)[self._model]


__all__ = [
    "MODEL_NAMES",
    "AppearanceModels",
    "AppearanceVerifier",
    "histogramSimilarity",
    "targetCrop",
]
