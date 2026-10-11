"""Record candidate "is this still the target?" signals next to a tracking run.

The probe only observes: for every tracked frame it scores the box the tracker
returned against the frame-0 template in several ways and writes one value per frame
and signal.  Tracking itself is untouched, so the result files of a probed run equal
those of a plain run.

Signals
    ``dinov2`` / ``dino`` / ``resnet18``
        cosine similarity between features of the current box crop and of the template
        crop, from a small frozen image model.
    ``hist``
        similarity of the HSV colour histograms of the two crops.
    ``clean``
        the tracker's own score when it is run again on the same search image with its
        accumulated state removed: appearance feature reset to the template, trajectory
        neutral.
    ``cleanIou``
        overlap of that second box with the box of the normal pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from track360.backends.appearance import (
    MODEL_NAMES as MODEL_SIGNALS,
)
from track360.backends.appearance import (
    AppearanceModels,
    histogramSimilarity,
    targetCrop,
)
from track360.core.types import BBoxXYWH, LocalObservation, LocalView

SIGNALS = (*MODEL_SIGNALS, "hist", "clean", "cleanIou")
PROBE_DIRECTORY = "probe"


def boxIou(first: BBoxXYWH, second: BBoxXYWH) -> float:
    width = min(first.xPx + first.widthPx, second.xPx + second.widthPx) - max(
        first.xPx, second.xPx
    )
    height = min(first.yPx + first.heightPx, second.yPx + second.heightPx) - max(
        first.yPx, second.yPx
    )
    if width <= 0.0 or height <= 0.0:
        return 0.0
    intersection = width * height
    union = first.widthPx * first.heightPx + second.widthPx * second.heightPx - intersection
    return float(intersection / union) if union > 0.0 else 0.0


class AppearanceProbe:
    """Collect the signals of one sequence."""

    def __init__(self, models: AppearanceModels | None, session: Any | None = None) -> None:
        self._models = models
        self._session = session
        self._templateCrop: NDArray[np.uint8] | None = None
        self._templateFeatures: dict[str, Any] | None = None
        self._anchor: Any | None = None
        self._rows: dict[int, dict[str, float]] = {}
        # The feature of every frame's box from the first model, for offline analysis.
        self._features: dict[int, NDArray[np.float32]] = {}
        self._templateFeature: NDArray[np.float32] | None = None

    def begin(self, template: LocalView, templateBox: BBoxXYWH) -> None:
        self._templateCrop = targetCrop(template.rgb, templateBox)
        if self._models is not None:
            self._templateFeatures = self._models.embed(self._templateCrop)
            first = self._models.names[0]
            self._templateFeature = self._templateFeatures[first][0].float().cpu().numpy()
        if self._session is not None:
            self._anchor = self._session.encodeTemplate(template.rgb, templateBox)

    def observe(self, frameIndex: int, view: LocalView, observation: LocalObservation) -> None:
        if self._templateCrop is None:
            return
        crop = targetCrop(view.rgb, observation.bbox)
        row: dict[str, float] = {"hist": histogramSimilarity(self._templateCrop, crop)}
        if self._models is not None and self._templateFeatures is not None:
            features = self._models.embed(crop)
            row.update(self._models.similarity(self._templateFeatures, features))
            self._features[frameIndex] = (
                features[self._models.names[0]][0].float().cpu().numpy()
            )
        if self._anchor is not None and view.spec.priorBox is not None:
            # A second pass with nothing carried over from earlier frames: the appearance
            # feature starts from the template again and no trajectory is given.
            self._anchor.memory.clear()
            prediction = self._session.inferBatch(
                (view.rgb,), (self._anchor,), priorBoxes=(view.spec.priorBox,)
            )[0]
            self._anchor.memory.clear()
            row["clean"] = float(prediction.modelScore)
            row["cleanIou"] = boxIou(prediction.bbox, observation.bbox)
        self._rows[frameIndex] = row

    def write(self, root: str | Path, method: str, sequence: str, frameCount: int) -> None:
        """One file per signal, one value per frame; frames without a value hold NaN."""
        for signal in SIGNALS:
            if not any(signal in row for row in self._rows.values()):
                continue
            values = np.full(frameCount, np.nan, dtype=np.float64)
            for frameIndex, row in self._rows.items():
                if signal in row and 0 <= frameIndex < frameCount:
                    values[frameIndex] = row[signal]
            path = probePath(root, method, signal, sequence)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(path, values, fmt="%.6f")
        if self._templateFeature is not None:
            # Row 0 is the template; a frame without a box keeps a zero row.
            table = np.zeros((frameCount, len(self._templateFeature)), dtype=np.float16)
            table[0] = self._templateFeature
            for frameIndex, feature in self._features.items():
                if 0 < frameIndex < frameCount:
                    table[frameIndex] = feature
            path = Path(root) / PROBE_DIRECTORY / method / "features" / f"{sequence}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, table)


def probePath(root: str | Path, method: str, signal: str, sequence: str) -> Path:
    return Path(root) / PROBE_DIRECTORY / method / signal / f"{sequence}.txt"


__all__ = [
    "MODEL_SIGNALS",
    "PROBE_DIRECTORY",
    "SIGNALS",
    "AppearanceModels",
    "AppearanceProbe",
    "boxIou",
    "histogramSimilarity",
    "probePath",
    "targetCrop",
]
