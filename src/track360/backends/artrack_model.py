"""ARTrackV2-B-256 runtime adapter.

The vendored implementation is intentionally kept behind this small adapter.  The
rest of the system only sees normalized RGB boxes and confidence values,
so replacing the model does not change the spherical controller contract.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from track360.core.errors import ModelError, ProtocolError
from track360.core.types import BBoxXYWH

_VARIANT = "artrackv2_b_256"
_TEMPLATE_SIZE = 128
_SEARCH_SIZE = 256
_TEMPLATE_FACTOR = 2.0
_SEARCH_FACTOR = 4.0
_BINS = 400


@dataclass(frozen=True, slots=True)
class ARTrackPrediction:
    bbox: BBoxXYWH
    modelScore: float
    appearanceScore: float
    predictedIoU: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("modelScore", self.modelScore),
            ("appearanceScore", self.appearanceScore),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ModelError(f"{name} must be finite and in [0, 1], actual={value}")
        if self.predictedIoU is not None and (
            not np.isfinite(self.predictedIoU) or not 0.0 <= self.predictedIoU <= 1.0
        ):
            raise ModelError(
                f"predictedIoU must be finite and in [0, 1], actual={self.predictedIoU}"
            )


@dataclass(frozen=True, slots=True)
class ARTrackTemplate:
    tensor: Any
    bbox: BBoxXYWH
    # Source view FOV lets the search crop preserve angular scale when the
    # controller changes perspective FOV between frames.
    sourceHorizontalFovRad: float | None = None
    sourceVerticalFovRad: float | None = None
    # Per-target state a session keeps between frames; copies of a template share it.
    memory: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


@runtime_checkable
class ARTrackSession(Protocol):
    def encodeTemplate(self, rgb: NDArray[np.uint8], bbox: BBoxXYWH) -> ARTrackTemplate: ...

    def infer(
        self, rgb: NDArray[np.uint8], templateFeatures: Sequence[object]
    ) -> ARTrackPrediction: ...

    def close(self) -> None: ...


class ARTrackBackend:
    """Validated facade around an ARTrackV2 session."""

    def __init__(self, session: ARTrackSession) -> None:
        if not isinstance(session, ARTrackSession):
            raise ProtocolError("session must implement the ARTrackSession protocol")
        self._session = session
        self._closed = False

    @property
    def trajectoryLength(self) -> int:
        """Previous boxes the session reads with every image."""
        return int(getattr(self._session, "trajectoryLength", 0))

    @property
    def lastProfile(self) -> dict[str, int | float | bool | str]:
        return dict(getattr(self._session, "lastProfile", {}))

    def encodeTemplate(self, rgb: NDArray[np.uint8], bbox: BBoxXYWH) -> ARTrackTemplate:
        self._requireOpen()
        _requireRgb(rgb)
        try:
            return self._session.encodeTemplate(rgb, bbox)
        except (ModelError, ProtocolError):
            raise
        except Exception as error:
            raise ModelError(f"ARTrackV2 template encoding failed: {error}") from error

    def encodeTemplateView(self, view: Any, bbox: BBoxXYWH) -> ARTrackTemplate:
        return replace(
            self.encodeTemplate(view.rgb, bbox),
            sourceHorizontalFovRad=view.spec.bfov.horizontalFovRad,
            sourceVerticalFovRad=view.spec.bfov.verticalFovRad,
        )

    def infer(
        self, rgb: NDArray[np.uint8], templateFeatures: Sequence[object]
    ) -> ARTrackPrediction:
        return self.inferBatch((rgb,), templateFeatures)[0]

    def inferBatch(
        self,
        rgbs: Sequence[NDArray[np.uint8]],
        templateFeatures: Sequence[object],
        imageFovs: Sequence[tuple[float, float]] | None = None,
        priorBoxes: Sequence[BBoxXYWH] | None = None,
        trajectories: Sequence[Sequence[BBoxXYWH]] | None = None,
    ) -> tuple[ARTrackPrediction, ...]:
        self._requireOpen()
        images = tuple(rgbs)
        if priorBoxes is not None and len(priorBoxes) != len(images):
            raise ProtocolError("ARTrackV2 prior boxes must match the image batch")
        if trajectories is not None and len(trajectories) != len(images):
            raise ProtocolError("ARTrackV2 trajectories must match the image batch")
        if imageFovs is not None and len(imageFovs) != len(images):
            raise ProtocolError("ARTrackV2 image FOV metadata must match the image batch")
        for rgb in images:
            _requireRgb(rgb)
        if not images:
            return ()
        if not templateFeatures:
            raise ProtocolError("ARTrackV2 inference requires at least one template feature")
        try:
            if priorBoxes is not None:
                # Only sequence-level sessions take trajectories.
                extra = {} if trajectories is None else {"trajectories": trajectories}
                predictions = tuple(
                    self._session.inferBatch(
                        images,
                        templateFeatures,
                        imageFovs=imageFovs,
                        priorBoxes=priorBoxes,
                        **extra,
                    )
                )
            elif callable(getattr(self._session, "inferBatch", None)):
                predictions = tuple(self._session.inferBatch(images, templateFeatures))
            else:
                predictions = tuple(self._session.infer(rgb, templateFeatures) for rgb in images)
        except (ModelError, ProtocolError):
            raise
        except Exception as error:
            raise ModelError(f"ARTrackV2 batch inference failed: {error}") from error
        if len(predictions) != len(images) or any(
            not isinstance(prediction, ARTrackPrediction) for prediction in predictions
        ):
            raise ModelError("ARTrackV2 session returned an invalid prediction batch")
        return predictions

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._session.close()
        except Exception as error:
            raise ModelError(f"ARTrackV2 session close failed: {error}") from error
        finally:
            self._closed = True

    def _requireOpen(self) -> None:
        if self._closed:
            raise ProtocolError("ARTrackV2 backend is closed")


def _sampleTarget(
    rgb: NDArray[np.uint8], bbox: BBoxXYWH, factor: float, outputSize: int
) -> tuple[NDArray[np.uint8], float, NDArray[np.bool_]]:
    import cv2

    cropSize = max(1, math.ceil(math.sqrt(bbox.widthPx * bbox.heightPx) * factor))
    x1 = round(bbox.xPx + 0.5 * bbox.widthPx - 0.5 * cropSize)
    y1 = round(bbox.yPx + 0.5 * bbox.heightPx - 0.5 * cropSize)
    x2, y2 = x1 + cropSize, y1 + cropSize
    left, right = max(0, -x1), max(x2 - rgb.shape[1] + 1, 0)
    top, bottom = max(0, -y1), max(y2 - rgb.shape[0] + 1, 0)
    crop = rgb[y1 + top : y2 - bottom, x1 + left : x2 - right]
    padded = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT)
    mask = np.ones(padded.shape[:2], dtype=np.bool_)
    if top or bottom or left or right:
        mask[
            top : padded.shape[0] - bottom if bottom else None,
            left : padded.shape[1] - right if right else None,
        ] = False
    return (
        np.ascontiguousarray(cv2.resize(padded, (outputSize, outputSize))),
        outputSize / cropSize,
        mask,
    )


def _centeredPrior(templateBox: BBoxXYWH, width: int, height: int) -> BBoxXYWH:
    return BBoxXYWH(
        xPx=max(0.0, width * 0.5 - templateBox.widthPx * 0.5),
        yPx=max(0.0, height * 0.5 - templateBox.heightPx * 0.5),
        widthPx=min(templateBox.widthPx, float(width)),
        heightPx=min(templateBox.heightPx, float(height)),
    )


def _clipBox(box: BBoxXYWH, width: int, height: int) -> BBoxXYWH:
    x = min(max(0.0, box.xPx), max(0.0, width - 1.0))
    y = min(max(0.0, box.yPx), max(0.0, height - 1.0))
    return BBoxXYWH(
        xPx=x,
        yPx=y,
        widthPx=max(1.0, min(box.widthPx, width - x)),
        heightPx=max(1.0, min(box.heightPx, height - y)),
    )


def _resolveArTrackRoot(value: str | Path | None) -> Path:
    root = (
        Path(value).expanduser().resolve()
        if value is not None
        else Path(__file__).resolve().parents[1] / "third_party" / "artrackv2"
    )
    if not (root / "lib" / "models" / "artrackv2_seq").is_dir():
        raise ModelError(f"ARTrackV2 source tree was not found: {root}")
    return root


def _activateVendorTree(root: Path) -> None:
    rootText = str(root)
    if rootText not in sys.path:
        sys.path.insert(0, rootText)


def _importTorch() -> ModuleType:
    try:
        import torch
    except ImportError as error:
        raise ModelError("ARTrackV2 requires PyTorch") from error
    return torch


def _requireRgb(rgb: NDArray[np.uint8]) -> None:
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or 0 in rgb.shape[:2]
    ):
        raise ProtocolError(
            "ARTrackV2 input rgb must have shape [H, W, 3] uint8, "
            f"actual={getattr(rgb, 'shape', None)}"
        )


__all__ = [
    "ARTrackBackend",
    "ARTrackPrediction",
    "ARTrackSession",
    "ARTrackTemplate",
]
