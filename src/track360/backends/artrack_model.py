"""ARTrackV2-B-256 runtime adapter.

The vendored implementation is intentionally kept behind this small adapter.  The
rest of the system only sees normalized RGB boxes and confidence values,
so replacing the model does not change the spherical controller contract.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter_ns
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from track360.core.config import ModelConfig
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


@runtime_checkable
class ARTrackSession(Protocol):
    supportsOnlineTemplates: bool

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
    def supportsOnlineTemplates(self) -> bool:
        return bool(self._session.supportsOnlineTemplates)

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
        # GPU geometry intentionally leaves ``rgb`` as a compatibility
        # placeholder.  Prefer the normalized device crop when available;
        # feeding that placeholder to the template encoder would initialize
        # ARTrack with an all-black target and destroy tracking IoU.
        deviceRgb = getattr(view, "deviceRgb", None)
        if deviceRgb is not None and callable(getattr(self._session, "encodeTemplateDevice", None)):
            try:
                encoded = self._session.encodeTemplateDevice(
                    deviceRgb,
                    bbox,
                    (view.spec.outputWidthPx, view.spec.outputHeightPx),
                )
                return replace(
                    encoded,
                    sourceHorizontalFovRad=view.spec.bfov.horizontalFovRad,
                    sourceVerticalFovRad=view.spec.bfov.verticalFovRad,
                )
            except (ModelError, ProtocolError):
                raise
            except Exception as error:
                raise ModelError(f"ARTrackV2 device template encoding failed: {error}") from error
        encoded = (
            self.encodeTemplate(_deviceRgbToNumpy(deviceRgb), bbox)
            if deviceRgb is not None
            else self.encodeTemplate(view.rgb, bbox)
        )
        return replace(
            encoded,
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
    ) -> tuple[ARTrackPrediction, ...]:
        self._requireOpen()
        images = tuple(rgbs)
        if priorBoxes is not None and len(priorBoxes) != len(images):
            raise ProtocolError("ARTrackV2 prior boxes must match the image batch")
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
                predictions = tuple(
                    self._session.inferBatch(
                        images, templateFeatures, imageFovs=imageFovs, priorBoxes=priorBoxes
                    )
                )
            elif imageFovs and callable(getattr(self._session, "inferBatchWithFovs", None)):
                predictions = tuple(
                    self._session.inferBatchWithFovs(images, templateFeatures, imageFovs)
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

    def inferDeviceBatch(
        self,
        deviceRgbs: Sequence[Any],
        imageSizes: Sequence[tuple[int, int]],
        templateFeatures: Sequence[object],
        imageFovs: Sequence[tuple[float, float]] | None = None,
    ) -> tuple[ARTrackPrediction, ...]:
        if len(deviceRgbs) != len(imageSizes):
            raise ProtocolError("device RGBs and image sizes must have equal length")
        if imageFovs is not None and len(imageFovs) != len(deviceRgbs):
            raise ProtocolError("ARTrackV2 device FOV metadata must match the image batch")
        inferDeviceBatch = getattr(self._session, "inferDeviceBatch", None)
        inferDeviceBatchWithFovs = getattr(self._session, "inferDeviceBatchWithFovs", None)
        if imageFovs and callable(inferDeviceBatchWithFovs):
            return tuple(
                inferDeviceBatchWithFovs(deviceRgbs, imageSizes, templateFeatures, imageFovs)
            )
        if callable(inferDeviceBatch):
            return tuple(inferDeviceBatch(deviceRgbs, imageSizes, templateFeatures))
        return self.inferBatch(
            tuple(_deviceRgbToNumpy(value) for value in deviceRgbs), templateFeatures
        )

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


class PyTorchARTrackV2Session:
    """Official ARTrackV2-B-256 PyTorch model with native crop/box mapping."""

    supportsOnlineTemplates = True

    def __init__(
        self,
        config: ModelConfig,
        *,
        artrackRoot: str | Path | None = None,
        fullViewSearch: bool = False,
    ) -> None:
        if config.backend != "pytorch":
            raise ModelError(f"PyTorch ARTrackV2 session cannot use backend={config.backend}")
        if config.variant.lower().replace("-", "_") not in {_VARIANT, "artrackv2_b_256"}:
            raise ModelError(
                f"unsupported ARTrackV2 variant: {config.variant}; expected={_VARIANT}"
            )
        self._torch = _importTorch()
        self._device = self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")
        self._precision = config.precision
        self._fullViewSearch = fullViewSearch
        self._closed = False
        self._lastProfile: dict[str, int | float | bool | str] = {}
        self._root = _resolveArTrackRoot(artrackRoot)
        self._weights = Path(config.weights).expanduser().resolve()
        if not self._weights.is_file():
            raise ModelError(f"ARTrackV2 checkpoint does not exist: {self._weights}")
        self._model = self._loadModel()

    @property
    def lastProfile(self) -> dict[str, int | float | bool | str]:
        return dict(self._lastProfile)

    def encodeTemplate(self, rgb: NDArray[np.uint8], bbox: BBoxXYWH) -> ARTrackTemplate:
        self._requireOpen()
        _requireRgb(rgb)
        crop, _, _ = _sampleTarget(rgb, bbox, _TEMPLATE_FACTOR, _TEMPLATE_SIZE)
        tensor = self._preprocess(crop)
        return ARTrackTemplate(tensor=tensor, bbox=bbox)

    def encodeTemplateDevice(
        self,
        deviceRgb: Any,
        bbox: BBoxXYWH,
        imageSize: tuple[int, int],
    ) -> ARTrackTemplate:
        """Encode a normalized GPU Geometry crop without using its host placeholder."""
        self._requireOpen()
        width, height = imageSize
        _requireDeviceRgb(deviceRgb, width, height)
        tensor, _, _ = _sampleNormalizedTarget(
            deviceRgb,
            bbox,
            _TEMPLATE_FACTOR,
            _TEMPLATE_SIZE,
            self._torch,
        )
        return ARTrackTemplate(tensor=tensor, bbox=bbox)

    def infer(
        self, rgb: NDArray[np.uint8], templateFeatures: Sequence[object]
    ) -> ARTrackPrediction:
        return self.inferBatch((rgb,), templateFeatures)[0]

    def inferBatch(
        self,
        rgbs: Sequence[NDArray[np.uint8]],
        templateFeatures: Sequence[object],
        *,
        imageFovs: Sequence[tuple[float, float]] | None = None,
        priorBoxes: Sequence[BBoxXYWH] | None = None,
    ) -> tuple[ARTrackPrediction, ...]:
        """Predict one box per image.

        The search crop is centered on the newest template's box by default, which
        suits views the controller already centers on the target.  ``priorBoxes``
        gives the previous box of each image instead, for tracking in a fixed image
        frame.
        """
        self._requireOpen()
        templates = tuple(item for item in templateFeatures if isinstance(item, ARTrackTemplate))
        if not templates:
            raise ProtocolError("ARTrackV2 template features are invalid")
        images = tuple(rgbs)
        if imageFovs is not None and len(imageFovs) != len(images):
            raise ProtocolError("ARTrackV2 image FOV metadata must match the image batch")
        if priorBoxes is not None and len(priorBoxes) != len(images):
            raise ProtocolError("ARTrackV2 prior boxes must match the image batch")
        states: list[BBoxXYWH] = []
        crops: list[NDArray[np.uint8]] = []
        resizeFactors: list[float] = []
        useFullView = self._fullViewSearch
        for index, image in enumerate(images):
            _requireRgb(image)
            # Each perspective view is an independent coordinate system. The
            # spherical controller centers its views on the current estimate, so
            # carrying a previous local box across viewIds would introduce drift.
            fov = imageFovs[index] if imageFovs is not None else None
            state = (
                priorBoxes[index]
                if priorBoxes is not None
                else _scaledPrior(
                    templates[-1], image.shape[1], image.shape[0],
                    fov[0] if fov is not None else None,
                    fov[1] if fov is not None else None,
                )
            )
            if useFullView:
                crop = np.ascontiguousarray(image)
                resizeFactor = _SEARCH_SIZE / float(image.shape[1])
                if image.shape[:2] != (_SEARCH_SIZE, _SEARCH_SIZE):
                    import cv2

                    crop = np.ascontiguousarray(cv2.resize(image, (_SEARCH_SIZE, _SEARCH_SIZE)))
            else:
                crop, resizeFactor, _ = _sampleTarget(image, state, _SEARCH_FACTOR, _SEARCH_SIZE)
            states.append(state)
            crops.append(crop)
            resizeFactors.append(resizeFactor)
        search = self._preprocessBatch(crops)
        template0 = templates[0].tensor
        template1 = templates[-1].tensor if len(templates) > 1 else template0
        template = self._torch.stack((template0, template1), dim=0)
        # Official ARTrackV2 expects [2, B, C, H, W] template batches.
        template = template[:, None].expand(2, len(images), -1, -1, -1)
        started = perf_counter_ns()
        with self._torch.inference_mode():
            output = self._model(template=template, search=search)
        elapsed = (perf_counter_ns() - started) // max(1, len(images))
        seqs = output.get("seqs")
        scores = output.get("score")
        if seqs is None or scores is None or seqs.shape[0] != len(images):
            raise ModelError("ARTrackV2 returned malformed prediction tensors")
        normalized = seqs[:, :4].float() / float(_BINS - 1) - 0.5
        predictions: list[ARTrackPrediction] = []
        for index, row in enumerate(normalized):
            x0, y0, x1, y1 = (float(value) for value in row.tolist())
            cx = (x0 + x1) * 0.5 * _SEARCH_SIZE / resizeFactors[index]
            cy = (y0 + y1) * 0.5 * _SEARCH_SIZE / resizeFactors[index]
            width = max(1.0, (x1 - x0) * _SEARCH_SIZE / resizeFactors[index])
            height = max(1.0, (y1 - y0) * _SEARCH_SIZE / resizeFactors[index])
            prior = states[index]
            half = 0.5 * _SEARCH_SIZE / resizeFactors[index]
            mapped = BBoxXYWH(
                xPx=cx + prior.xPx + 0.5 * prior.widthPx - half - 0.5 * width,
                yPx=cy + prior.yPx + 0.5 * prior.heightPx - half - 0.5 * height,
                widthPx=width,
                heightPx=height,
            )
            mapped = _clipBox(mapped, images[index].shape[1], images[index].shape[0])
            score = float(self._torch.sigmoid(scores[index].reshape(-1)[0]).item())
            predictions.append(ARTrackPrediction(mapped, score, score, score))
        self._lastProfile = {
            "cudaForward": int(elapsed),
            "batchSize": len(images),
            "device": str(self._device),
        }
        return tuple(predictions)

    def inferBatchWithFovs(
        self,
        rgbs: Sequence[NDArray[np.uint8]],
        templateFeatures: Sequence[object],
        imageFovs: Sequence[tuple[float, float]],
    ) -> tuple[ARTrackPrediction, ...]:
        return self.inferBatch(rgbs, templateFeatures, imageFovs=imageFovs)

    def inferDeviceBatch(
        self,
        deviceRgbs: Sequence[Any],
        imageSizes: Sequence[tuple[int, int]],
        templateFeatures: Sequence[object],
        *,
        imageFovs: Sequence[tuple[float, float]] | None = None,
    ) -> tuple[ARTrackPrediction, ...]:
        """Run the same official crop/decode path directly on normalized CUDA views."""
        self._requireOpen()
        templates = tuple(item for item in templateFeatures if isinstance(item, ARTrackTemplate))
        if not templates:
            raise ProtocolError("ARTrackV2 template features are invalid")
        if len(deviceRgbs) != len(imageSizes):
            raise ProtocolError("device RGBs and image sizes must have equal length")
        if imageFovs is not None and len(imageFovs) != len(deviceRgbs):
            raise ProtocolError("ARTrackV2 device FOV metadata must match the image batch")
        states: list[BBoxXYWH] = []
        crops: list[Any] = []
        resizeFactors: list[float] = []
        useFullView = self._fullViewSearch
        for deviceRgb, (width, height) in zip(deviceRgbs, imageSizes, strict=True):
            _requireDeviceRgb(deviceRgb, width, height)
            fov = imageFovs[len(states)] if imageFovs is not None else None
            state = _scaledPrior(
                templates[-1], width, height,
                fov[0] if fov is not None else None,
                fov[1] if fov is not None else None,
            )
            if useFullView:
                crop = deviceRgb
                resizeFactor = _SEARCH_SIZE / float(width)
                if (height, width) != (_SEARCH_SIZE, _SEARCH_SIZE):
                    crop = self._torch.nn.functional.interpolate(
                        crop.unsqueeze(0),
                        size=(_SEARCH_SIZE, _SEARCH_SIZE),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
            else:
                crop, resizeFactor, _ = _sampleNormalizedTarget(
                    deviceRgb,
                    state,
                    _SEARCH_FACTOR,
                    _SEARCH_SIZE,
                    self._torch,
                )
            states.append(state)
            crops.append(crop)
            resizeFactors.append(resizeFactor)
        search = self._torch.stack(crops, dim=0)
        template0 = templates[0].tensor
        template1 = templates[-1].tensor if len(templates) > 1 else template0
        template = self._torch.stack((template0, template1), dim=0)
        template = template[:, None].expand(2, len(deviceRgbs), -1, -1, -1)
        started = perf_counter_ns()
        with self._torch.inference_mode():
            output = self._model(template=template, search=search)
        elapsed = (perf_counter_ns() - started) // max(1, len(deviceRgbs))
        return self._decodeOutput(output, states, resizeFactors, imageSizes, elapsed)

    def inferDeviceBatchWithFovs(
        self,
        deviceRgbs: Sequence[Any],
        imageSizes: Sequence[tuple[int, int]],
        templateFeatures: Sequence[object],
        imageFovs: Sequence[tuple[float, float]],
    ) -> tuple[ARTrackPrediction, ...]:
        return self.inferDeviceBatch(
            deviceRgbs, imageSizes, templateFeatures, imageFovs=imageFovs
        )

    def close(self) -> None:
        if self._closed:
            return
        self._model = None
        if self._device.type == "cuda":
            self._torch.cuda.empty_cache()
        self._closed = True

    def _loadModel(self) -> Any:
        _activateVendorTree(self._root)
        try:
            from lib.config.artrackv2.config import cfg, update_config_from_file
            from lib.models.artrackv2 import build_artrackv2

            update_config_from_file(str(self._root / "artrackv2_256_full.yaml"))
            model = build_artrackv2(cfg, training=False).to(self._device).eval()
            try:
                checkpoint = self._torch.load(
                    self._weights,
                    map_location="cpu",
                    weights_only=True,
                )
            except Exception as safeError:
                # Official ARTrack archives include legacy training meters in
                # addition to ``net``.  PyTorch 2.6 rejects those objects under
                # weights_only=True; the checkpoint path is explicit and local,
                # so use the compatible loader after the safe attempt fails.
                try:
                    checkpoint = self._torch.load(
                        self._weights,
                        map_location="cpu",
                        weights_only=False,
                    )
                except Exception:
                    raise safeError
            state = (
                checkpoint.get("net", checkpoint.get("model", checkpoint))
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            if not isinstance(state, dict):
                raise ModelError("ARTrackV2 checkpoint has no state dictionary")
            incompatible = model.load_state_dict(state, strict=False)
            if incompatible.missing_keys:
                raise ModelError(
                    "ARTrackV2 checkpoint is missing model parameters: "
                    + ", ".join(incompatible.missing_keys[:8])
                )
            return model
        except ModelError:
            raise
        except Exception as error:
            raise ModelError(
                f"cannot construct ARTrackV2-B-256 from {self._weights}: {error}"
            ) from error

    def _preprocess(self, image: NDArray[np.uint8]) -> Any:
        return self._preprocessBatch((image,))[0]

    def _preprocessBatch(self, images: Sequence[NDArray[np.uint8]]) -> Any:
        array = np.stack(images).astype(np.float32) / 255.0
        tensor = self._torch.from_numpy(array).permute(0, 3, 1, 2).to(self._device)
        mean = self._torch.tensor([0.485, 0.456, 0.406], device=self._device).view(1, 3, 1, 1)
        std = self._torch.tensor([0.229, 0.224, 0.225], device=self._device).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def _decodeOutput(
        self,
        output: Any,
        states: Sequence[BBoxXYWH],
        resizeFactors: Sequence[float],
        imageSizes: Sequence[tuple[int, int]],
        elapsed: int,
    ) -> tuple[ARTrackPrediction, ...]:
        seqs = output.get("seqs")
        scores = output.get("score")
        if seqs is None or scores is None or seqs.shape[0] != len(states):
            raise ModelError("ARTrackV2 returned malformed prediction tensors")
        normalized = seqs[:, :4].float() / float(_BINS - 1) - 0.5
        predictions: list[ARTrackPrediction] = []
        for index, row in enumerate(normalized):
            x0, y0, x1, y1 = (float(value) for value in row.tolist())
            cx = (x0 + x1) * 0.5 * _SEARCH_SIZE / resizeFactors[index]
            cy = (y0 + y1) * 0.5 * _SEARCH_SIZE / resizeFactors[index]
            width = max(1.0, (x1 - x0) * _SEARCH_SIZE / resizeFactors[index])
            height = max(1.0, (y1 - y0) * _SEARCH_SIZE / resizeFactors[index])
            prior = states[index]
            half = 0.5 * _SEARCH_SIZE / resizeFactors[index]
            mapped = BBoxXYWH(
                xPx=cx + prior.xPx + 0.5 * prior.widthPx - half - 0.5 * width,
                yPx=cy + prior.yPx + 0.5 * prior.heightPx - half - 0.5 * height,
                widthPx=width,
                heightPx=height,
            )
            mapped = _clipBox(mapped, imageSizes[index][0], imageSizes[index][1])
            score = float(self._torch.sigmoid(scores[index].reshape(-1)[0]).item())
            predictions.append(ARTrackPrediction(mapped, score, score, score))
        self._lastProfile = {
            "cudaForward": int(elapsed),
            "batchSize": len(states),
            "device": str(self._device),
            "deviceInput": True,
        }
        return tuple(predictions)

    def _requireOpen(self) -> None:
        if self._closed:
            raise ProtocolError("ARTrackV2 session is closed")


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


def _sampleNormalizedTarget(
    normalizedRgb: Any,
    bbox: BBoxXYWH,
    factor: float,
    outputSize: int,
    torch: Any,
) -> tuple[Any, float, Any]:
    """Crop a normalized [C,H,W] tensor with the official square crop geometry."""
    height, width = int(normalizedRgb.shape[1]), int(normalizedRgb.shape[2])
    cropSize = max(1, math.ceil(math.sqrt(bbox.widthPx * bbox.heightPx) * factor))
    x1 = round(bbox.xPx + 0.5 * bbox.widthPx - 0.5 * cropSize)
    y1 = round(bbox.yPx + 0.5 * bbox.heightPx - 0.5 * cropSize)
    # grid_sample uses border padding, while the official path uses constant
    # zero pixels before normalization. Build the exact zero-padded square.
    left, right = max(0, -x1), max(x1 + cropSize - width, 0)
    top, bottom = max(0, -y1), max(y1 + cropSize - height, 0)
    fill = torch.tensor(
        [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        device=normalizedRgb.device,
        dtype=normalizedRgb.dtype,
    ).view(3, 1, 1)
    padded = fill.expand(3, cropSize + top + bottom, cropSize + left + right).clone()
    src = normalizedRgb[
        :,
        y1 + top : y1 + top + cropSize - top - bottom,
        x1 + left : x1 + left + cropSize - left - right,
    ]
    padded[:, top : top + src.shape[1], left : left + src.shape[2]] = src
    if cropSize == outputSize:
        return (
            padded,
            1.0,
            torch.ones((cropSize, cropSize), device=normalizedRgb.device, dtype=torch.bool),
        )
    resized = torch.nn.functional.interpolate(
        padded.unsqueeze(0), size=(outputSize, outputSize), mode="bilinear", align_corners=False
    ).squeeze(0)
    mask = torch.ones((cropSize, cropSize), device=normalizedRgb.device, dtype=torch.bool)
    mask = torch.nn.functional.interpolate(
        mask[None, None].float(), size=(outputSize, outputSize), mode="nearest"
    ).bool()[0, 0]
    return resized, outputSize / cropSize, mask


def _centeredPrior(templateBox: BBoxXYWH, width: int, height: int) -> BBoxXYWH:
    return BBoxXYWH(
        xPx=max(0.0, width * 0.5 - templateBox.widthPx * 0.5),
        yPx=max(0.0, height * 0.5 - templateBox.heightPx * 0.5),
        widthPx=min(templateBox.widthPx, float(width)),
        heightPx=min(templateBox.heightPx, float(height)),
    )


def _scaledPrior(
    template: ARTrackTemplate,
    width: int,
    height: int,
    targetHorizontalFovRad: float | None,
    targetVerticalFovRad: float | None,
) -> BBoxXYWH:
    """Center the newest template and optionally preserve angular extent.

    The public backend historically passed only RGB arrays, so target FOVs are
    optional.  When unavailable this still improves appearance/scale changes by
    using the newest online template instead of the immutable anchor.
    """
    box = template.bbox
    scaleX = scaleY = 1.0
    if (
        targetHorizontalFovRad is not None
        and targetVerticalFovRad is not None
        and template.sourceHorizontalFovRad is not None
        and template.sourceVerticalFovRad is not None
    ):
        import math

        scaleX = math.tan(0.5 * template.sourceHorizontalFovRad) / max(
            math.tan(0.5 * targetHorizontalFovRad), 1e-6
        )
        scaleY = math.tan(0.5 * template.sourceVerticalFovRad) / max(
            math.tan(0.5 * targetVerticalFovRad), 1e-6
        )
        scaleX = min(3.0, max(0.35, scaleX))
        scaleY = min(3.0, max(0.35, scaleY))
    return _centeredPrior(
        BBoxXYWH(
            xPx=box.xPx,
            yPx=box.yPx,
            widthPx=max(1.0, box.widthPx * scaleX),
            heightPx=max(1.0, box.heightPx * scaleY),
        ),
        width,
        height,
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


def _deviceRgbToNumpy(value: Any) -> NDArray[np.uint8]:
    _requireDeviceRgb(value)
    array = value.detach().float().cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return np.ascontiguousarray(
        np.clip((array * std + mean) * 255.0, 0.0, 255.0)
        .transpose(1, 2, 0)
        .round()
        .astype(np.uint8)
    )


def _requireDeviceRgb(value: Any, width: int | None = None, height: int | None = None) -> None:
    if not hasattr(value, "detach"):
        raise ProtocolError("ARTrackV2 device input must be a torch tensor")
    shape = tuple(getattr(value, "shape", ()))
    if len(shape) != 3 or shape[0] != 3:
        raise ProtocolError(f"ARTrackV2 device input must have shape [3,H,W], actual={shape}")
    if width is not None and height is not None and shape != (3, height, width):
        raise ProtocolError(
            "ARTrackV2 device input size does not match its view: "
            f"tensor={shape}, view={(3, height, width)}"
        )


def _resolveArTrackRoot(value: str | Path | None) -> Path:
    root = (
        Path(value).expanduser().resolve()
        if value is not None
        else Path(__file__).resolve().parents[1] / "third_party" / "artrackv2"
    )
    if not (root / "lib" / "models" / "artrackv2").is_dir():
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
    "PyTorchARTrackV2Session",
]
