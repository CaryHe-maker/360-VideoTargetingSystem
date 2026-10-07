"""ARTrackV2 sequence-level inference.

The released ARTrackV2-B-256 checkpoint is the sequence-level model: besides the
template and the search crop it expects the boxes of the previous frames (the
trajectory prompt) and an appearance feature it rewrites itself every frame.  This
session runs it the way the upstream tracker does (``lib/test/tracker/artrackv2_seq.py``).

Two pieces of state follow a target from frame to frame:

* the appearance feature, kept on the target's first template (``ARTrackTemplate.memory``),
  so a new template starts a new target;
* the trajectory, which the caller supplies with every image because only the caller
  knows how earlier boxes map into the current image.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
from numpy.typing import NDArray

from track360.backends.artrack_model import (
    _SEARCH_FACTOR,
    _SEARCH_SIZE,
    _TEMPLATE_FACTOR,
    _TEMPLATE_SIZE,
    ARTrackPrediction,
    ARTrackSession,
    ARTrackTemplate,
    PyTorchARTrackV2Session,
    _activateVendorTree,
    _centeredPrior,
    _clipBox,
    _importTorch,
    _requireRgb,
    _resolveArTrackRoot,
    _sampleTarget,
)
from track360.core.config import AppConfig, ModelConfig
from track360.core.errors import ModelError, ProtocolError
from track360.core.types import BBoxXYWH

_APPEARANCE_KEY = "appearance"
# Trajectory coordinates are relative to the search crop; outside this range they clamp.
_COORDINATE_MIN = -0.5
_COORDINATE_MAX = 1.5


class PyTorchARTrackV2SeqSession:
    """Official ARTrackV2-B-256 run as a sequence-level tracker."""

    supportsOnlineTemplates = False

    def __init__(self, config: ModelConfig, *, artrackRoot: str | Path | None = None) -> None:
        if config.backend != "pytorch":
            raise ModelError(f"PyTorch ARTrackV2 session cannot use backend={config.backend}")
        self._torch = _importTorch()
        self._device = self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")
        self._closed = False
        self._lastProfile: dict[str, int | float | bool | str] = {}
        self._root = _resolveArTrackRoot(artrackRoot)
        self._weights = Path(config.weights).expanduser().resolve()
        if not self._weights.is_file():
            raise ModelError(f"ARTrackV2 checkpoint does not exist: {self._weights}")
        self._model, modelConfig = self._loadModel()
        self._bins = int(modelConfig.BINS)
        self._range = int(modelConfig.RANGE)
        self._trajectoryLength = int(modelConfig.PRENUM)
        # Centers of the coordinate bins, as upstream decodes them.
        step = 2.0 / (self._bins * self._range)
        first = (-0.5 * self._range + 0.5) + 1.0 / (self._bins * self._range)
        self._binCenters = (
            first + step * self._torch.arange(self._bins * self._range, device=self._device)
        ).float()

    @property
    def lastProfile(self) -> dict[str, int | float | bool | str]:
        return dict(self._lastProfile)

    @property
    def trajectoryLength(self) -> int:
        """How many previous boxes the model reads with every search crop."""
        return self._trajectoryLength

    def encodeTemplate(self, rgb: NDArray[np.uint8], bbox: BBoxXYWH) -> ARTrackTemplate:
        self._requireOpen()
        _requireRgb(rgb)
        crop, _, _ = _sampleTarget(rgb, bbox, _TEMPLATE_FACTOR, _TEMPLATE_SIZE)
        return ARTrackTemplate(tensor=self._preprocess(crop), bbox=bbox)

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
        trajectories: Sequence[Sequence[BBoxXYWH]] | None = None,
    ) -> tuple[ARTrackPrediction, ...]:
        """Predict one box per image of the same target, in order.

        ``priorBoxes`` gives where the target is expected in each image; the search crop
        is centered on it.  ``trajectories`` gives the target's previous boxes in each
        image's coordinates, oldest first; without it the prior box stands in for all
        of them, as upstream does on the first frame.
        """
        del imageFovs
        self._requireOpen()
        templates = tuple(item for item in templateFeatures if isinstance(item, ARTrackTemplate))
        if not templates:
            raise ProtocolError("ARTrackV2 template features are invalid")
        images = tuple(rgbs)
        if priorBoxes is not None and len(priorBoxes) != len(images):
            raise ProtocolError("ARTrackV2 prior boxes must match the image batch")
        if trajectories is not None and len(trajectories) != len(images):
            raise ProtocolError("ARTrackV2 trajectories must match the image batch")
        # The appearance feature is updated by every forward pass, so images of one
        # target are processed one after the other.
        anchor = templates[0]
        predictions: list[ARTrackPrediction] = []
        elapsed = 0
        for index, image in enumerate(images):
            _requireRgb(image)
            prior = (
                priorBoxes[index]
                if priorBoxes is not None
                else _centeredPrior(anchor.bbox, image.shape[1], image.shape[0])
            )
            trajectory = trajectories[index] if trajectories is not None else None
            prediction, forwardNs = self._inferOne(image, anchor, prior, trajectory)
            predictions.append(prediction)
            elapsed += forwardNs
        self._lastProfile = {
            "cudaForward": int(elapsed // max(1, len(images))),
            "batchSize": len(images),
            "device": str(self._device),
            "sequenceModel": True,
        }
        return tuple(predictions)

    def close(self) -> None:
        if self._closed:
            return
        self._model = None
        if self._device.type == "cuda":
            self._torch.cuda.empty_cache()
        self._closed = True

    def _inferOne(
        self,
        image: NDArray[np.uint8],
        anchor: ARTrackTemplate,
        prior: BBoxXYWH,
        trajectory: Sequence[BBoxXYWH] | None,
    ) -> tuple[ARTrackPrediction, int]:
        torch = self._torch
        crop, resizeFactor, _ = _sampleTarget(image, prior, _SEARCH_FACTOR, _SEARCH_SIZE)
        search = self._preprocess(crop).unsqueeze(0)
        boxes = padTrajectory(trajectory or (prior,), self._trajectoryLength)
        tokens = torch.tensor(
            [trajectoryTokens(boxes, prior, resizeFactor, self._bins)],
            dtype=torch.float32,
            device=self._device,
        )
        appearance = anchor.memory.get(_APPEARANCE_KEY)
        started = perf_counter_ns()
        with torch.inference_mode():
            if appearance is None:
                appearance = self._model.backbone.patch_embed(anchor.tensor.unsqueeze(0))
            output = self._model(
                template=anchor.tensor[None, None],
                # Cloned: the backbone adds its embeddings to this tensor in place.
                dz_feat=appearance.clone(),
                search=search,
                seq_input=tokens,
            )
        elapsed = perf_counter_ns() - started
        anchor.memory[_APPEARANCE_KEY] = output["dz_feat"]

        # Upstream averages the arg-max bin with the expectation over all bins.
        hard = (output["seqs"][:, 0:4].float() + 0.5) / (self._bins - 1) - 0.5
        logits = output["feat"][0:4, :, 0 : self._bins * self._range]
        soft = (logits.softmax(-1) * self._binCenters).sum(dim=-1).permute(1, 0)
        x0, y0, x1, y1 = (float(value) for value in ((hard + soft) / 2.0)[0].tolist())
        scale = _SEARCH_SIZE / resizeFactor
        width = max(1.0, (x1 - x0) * scale)
        height = max(1.0, (y1 - y0) * scale)
        centerX = (x0 + x1) * 0.5 * scale
        centerY = (y0 + y1) * 0.5 * scale
        half = 0.5 * scale
        mapped = BBoxXYWH(
            xPx=centerX + prior.xPx + 0.5 * prior.widthPx - half - 0.5 * width,
            yPx=centerY + prior.yPx + 0.5 * prior.heightPx - half - 0.5 * height,
            widthPx=width,
            heightPx=height,
        )
        mapped = _clipBox(mapped, image.shape[1], image.shape[0])
        # The score head regresses the IoU of its own box directly.
        score = float(np.clip(float(output["score"].reshape(-1)[0].item()), 0.0, 1.0))
        return ARTrackPrediction(mapped, score, score, score), int(elapsed)

    def _loadModel(self) -> tuple[Any, Any]:
        _activateVendorTree(self._root)
        try:
            from lib.config.artrackv2_seq.config import cfg, update_config_from_file
            from lib.models.artrackv2_seq import build_artrackv2_seq

            update_config_from_file(str(self._root / "artrackv2_seq_256_full.yaml"))
            model = build_artrackv2_seq(cfg, training=False)
            try:
                checkpoint = self._torch.load(self._weights, map_location="cpu", weights_only=True)
            except Exception as safeError:
                # Official ARTrack archives include legacy training meters in addition to
                # ``net``; the checkpoint path is explicit and local, so fall back to the
                # compatible loader after the safe attempt fails.
                try:
                    checkpoint = self._torch.load(
                        self._weights, map_location="cpu", weights_only=False
                    )
                except Exception:
                    raise safeError from None
            state = (
                checkpoint.get("net", checkpoint.get("model", checkpoint))
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            if not isinstance(state, dict):
                raise ModelError("ARTrackV2 checkpoint has no state dictionary")
            # Strict: a sequence-level model that silently drops or lacks parameters
            # would run, but not as the model that was trained.
            model.load_state_dict(state, strict=True)
            # The coordinate embedding is declared with ``max_norm``: PyTorch rescales a
            # row in place the first time it is looked up, and the same table is the
            # output projection.  The checkpoint still holds rows above the limit (bins
            # far outside the crop), so predictions would depend on which coordinates
            # the process has seen before.  Looking every row up once settles the table.
            embedding = model.backbone.word_embeddings
            with self._torch.no_grad():
                embedding(self._torch.arange(embedding.num_embeddings))
            return model.to(self._device).eval(), cfg.MODEL
        except ModelError:
            raise
        except Exception as error:
            raise ModelError(
                f"cannot construct sequence-level ARTrackV2-B-256 from {self._weights}: {error}"
            ) from error

    def _preprocess(self, image: NDArray[np.uint8]) -> Any:
        array = image.astype(np.float32) / 255.0
        tensor = self._torch.from_numpy(array).permute(2, 0, 1).to(self._device)
        mean = self._torch.tensor([0.485, 0.456, 0.406], device=self._device).view(3, 1, 1)
        std = self._torch.tensor([0.229, 0.224, 0.225], device=self._device).view(3, 1, 1)
        return (tensor - mean) / std

    def _requireOpen(self) -> None:
        if self._closed:
            raise ProtocolError("ARTrackV2 session is closed")


def padTrajectory(trajectory: Sequence[BBoxXYWH], length: int) -> list[BBoxXYWH]:
    """Keep the newest ``length`` boxes; a shorter history is padded with its oldest."""
    boxes = list(trajectory)[-length:]
    return [boxes[0]] * (length - len(boxes)) + boxes


def trajectoryTokens(
    boxes: Sequence[BBoxXYWH], prior: BBoxXYWH, resizeFactor: float, bins: int
) -> list[float]:
    """Previous boxes as ``x0, y0, x1, y1`` bin coordinates of the search crop.

    The crop is centered on ``prior`` and scaled by ``resizeFactor``; coordinates are
    relative to the crop side, clamped to [-0.5, 1.5] and spread over ``2 * bins``
    bins, exactly as the upstream tracker builds its ``seq_input``.
    """
    priorCenterX = prior.xPx + 0.5 * prior.widthPx
    priorCenterY = prior.yPx + 0.5 * prior.heightPx
    middle = (_SEARCH_SIZE - 1) / 2.0
    tokens: list[float] = []
    for box in boxes:
        width = box.widthPx * resizeFactor
        height = box.heightPx * resizeFactor
        centerX = middle + (box.xPx + 0.5 * box.widthPx - priorCenterX) * resizeFactor
        centerY = middle + (box.yPx + 0.5 * box.heightPx - priorCenterY) * resizeFactor
        x0 = centerX - 0.5 * width
        y0 = centerY - 0.5 * height
        for value in (x0, y0, x0 + width, y0 + height):
            normalized = min(_COORDINATE_MAX, max(_COORDINATE_MIN, value / _SEARCH_SIZE))
            tokens.append((normalized + 0.5) * (bins - 1))
    return tokens


def createArtrackSession(config: AppConfig) -> ARTrackSession:
    """Load the ARTrackV2 session the configuration asks for."""
    if config.backendTuning.sequenceModel:
        return PyTorchARTrackV2SeqSession(config.model)
    return PyTorchARTrackV2Session(
        config.model, fullViewSearch=config.backendTuning.fullViewSearch
    )


__all__ = [
    "PyTorchARTrackV2SeqSession",
    "createArtrackSession",
    "padTrajectory",
    "trajectoryTokens",
]
