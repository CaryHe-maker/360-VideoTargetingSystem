"""CUDA perspective geometry for direct tracker device-tensor inference."""

from __future__ import annotations

from collections.abc import Sequence
from math import pi
from time import perf_counter_ns
from typing import Any

import numpy as np

from track360.core.errors import GeometryError
from track360.core.types import FramePacket, LocalView, ViewSpec
from track360.geometry.spherical_geometry import SphericalGeometryImpl


class GpuGeometryImpl(SphericalGeometryImpl):
    """Keep crop/resize/normalization on CUDA and expose tensors to ARTrackV2."""

    def __init__(self, boundarySamplesPerEdge: int = 65, *, profileEnabled: bool = False) -> None:
        super().__init__(boundarySamplesPerEdge=boundarySamplesPerEdge)
        try:
            import torch
        except ImportError as error:
            raise GeometryError("GPU Geometry requires PyTorch") from error
        if not torch.cuda.is_available():
            raise GeometryError("GPU Geometry requires CUDA")
        self._torch = torch
        self._device = torch.device("cuda")
        self._stream = torch.cuda.default_stream(self._device)
        self._frameIndex: int | None = None
        self._frameTensor: Any = None
        self._hostTensor: Any = None
        self._frameToDeviceNs = 0
        self._profileEnabled = profileEnabled
        self._localGridCache: dict[tuple[int, int], tuple[Any, Any]] = {}
        self._mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self._device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            [0.229, 0.224, 0.225], device=self._device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self._lastProfile: dict[str, int | float | bool | str] = {}
        self._closed = False

    @property
    def lastProfile(self) -> dict[str, int | float | bool | str]:
        return dict(self._lastProfile)

    def releaseFrame(self) -> None:
        """Drop per-frame CUDA storage as soon as all rounds have consumed it."""
        self._frameTensor = None
        self._hostTensor = None
        self._frameIndex = None

    def close(self) -> None:
        """Synchronize and release persistent CUDA tensors before context teardown."""
        if self._closed:
            return
        try:
            self._torch.cuda.synchronize(self._device)
        except RuntimeError:
            # Preserve an earlier asynchronous CUDA error while still dropping
            # references that would otherwise survive until interpreter shutdown.
            pass
        self.releaseFrame()
        self._localGridCache.clear()
        self._mean = None
        self._std = None
        self._stream = None
        self._closed = True

    def cropViews(self, frame: FramePacket, specs: Sequence[ViewSpec]) -> list[LocalView]:
        started = perf_counter_ns()
        result: list[LocalView] = []
        cropStarted = perf_counter_ns()
        with self._torch.cuda.stream(self._stream):
            self._ensureFrame(frame)
            for spec in specs:
                tensor = self._perspectiveCrop(spec)
                # Compatibility RGB is never consumed by the GPU backend.
                placeholder = np.zeros(
                    (spec.outputHeightPx, spec.outputWidthPx, 3),
                    dtype=np.uint8,
                )
                result.append(LocalView(spec=spec, rgb=placeholder, deviceRgb=tensor))
        # Establish a synchronous ownership boundary before ARTrackV2 consumes the tensors.
        self._stream.synchronize()
        self._synchronizeForProfile()
        gpuCropNs = perf_counter_ns() - cropStarted
        self._lastProfile = {
            "gpuGeometry": True,
            "frameToDevice": int(self._frameToDeviceNs),
            "gpuCrop": int(gpuCropNs),
            "gpuGeometryTotal": int(perf_counter_ns() - started),
            "viewCount": len(specs),
            "imageRoundTrips": 0,
            "gridCpuTransfers": 0,
            "gridDevice": str(self._device),
            "frameTensorDtype": str(self._frameTensor.dtype),
            "frameTensorDevice": str(self._frameTensor.device),
        }
        return result

    def _ensureFrame(self, frame: FramePacket) -> None:
        frameIndex = int(frame.frameIndex)
        if self._frameIndex == frameIndex and self._frameTensor is not None:
            return
        started = perf_counter_ns()
        host = self._torch.from_numpy(np.ascontiguousarray(frame.rgb))
        if not host.is_pinned():
            host = host.pin_memory()
        self._hostTensor = host
        self._frameTensor = (
            host.to(device=self._device, dtype=self._torch.float32, non_blocking=True)
            .permute(2, 0, 1)
            .contiguous()
        )
        self._frameIndex = frameIndex
        self._synchronizeForProfile()
        self._frameToDeviceNs = perf_counter_ns() - started

    def _perspectiveCrop(self, spec: ViewSpec) -> Any:
        height = int(self._frameTensor.shape[1])
        width = int(self._frameTensor.shape[2])
        localX, localY = self._localPixelGrid(spec.outputWidthPx, spec.outputHeightPx)
        torch = self._torch
        dtype = torch.float32
        yaw = torch.tensor(spec.bfov.center.yawRad, device=self._device, dtype=dtype)
        pitch = torch.tensor(spec.bfov.center.pitchRad, device=self._device, dtype=dtype)
        roll = torch.tensor(spec.bfov.rollRad, device=self._device, dtype=dtype)
        sinYaw, cosYaw = torch.sin(yaw), torch.cos(yaw)
        sinPitch, cosPitch = torch.sin(pitch), torch.cos(pitch)
        sinRoll, cosRoll = torch.sin(roll), torch.cos(roll)
        zero = torch.zeros((), device=self._device, dtype=dtype)
        forward = torch.stack((cosPitch * sinYaw, sinPitch, cosPitch * cosYaw))
        baseRight = torch.stack((cosYaw, zero, -sinYaw))
        baseUp = torch.stack((-sinPitch * sinYaw, cosPitch, -sinPitch * cosYaw))
        right = cosRoll * baseRight + sinRoll * baseUp
        up = -sinRoll * baseRight + cosRoll * baseUp
        horizontalScale = torch.tan(
            torch.tensor(spec.bfov.horizontalFovRad / 2.0, device=self._device, dtype=dtype)
        )
        verticalScale = torch.tan(
            torch.tensor(spec.bfov.verticalFovRad / 2.0, device=self._device, dtype=dtype)
        )
        horizontal = (2.0 * localX / spec.outputWidthPx - 1.0) * horizontalScale
        vertical = (1.0 - 2.0 * localY / spec.outputHeightPx) * verticalScale
        vectors = (
            forward[:, None, None]
            + right[:, None, None] * horizontal[None, :, :]
            + up[:, None, None] * vertical[None, :, :]
        )
        vectors = vectors / torch.linalg.vector_norm(vectors, dim=0, keepdim=True)
        sampleX = torch.remainder(
            (torch.atan2(vectors[0], vectors[2]) + pi) * width / (2.0 * pi) - 0.5,
            width,
        )
        sampleY = torch.clamp(
            (pi / 2.0 - torch.asin(torch.clamp(vectors[1], -1.0, 1.0)))
            * height
            / pi
            - 0.5,
            0.0,
            float(height - 1),
        )
        return self._normalizeSampledRgb(self._sampleCircularBilinear(sampleX, sampleY))

    def _localPixelGrid(self, width: int, height: int) -> tuple[Any, Any]:
        key = (width, height)
        cached = self._localGridCache.get(key)
        if cached is not None:
            return cached
        x = self._torch.arange(width, device=self._device, dtype=self._torch.float32) + 0.5
        y = self._torch.arange(height, device=self._device, dtype=self._torch.float32) + 0.5
        grid = self._torch.meshgrid(y, x, indexing="ij")
        cached = (grid[1], grid[0])
        self._localGridCache[key] = cached
        return cached

    def _sampleCircularBilinear(self, sampleX: Any, sampleY: Any) -> Any:
        torch = self._torch
        _, height, width = self._frameTensor.shape
        x = torch.remainder(sampleX, width)
        y = torch.clamp(sampleY, 0.0, float(height - 1))
        x0 = torch.floor(x).to(dtype=torch.long)
        y0 = torch.floor(y).to(dtype=torch.long)
        x1 = torch.remainder(x0 + 1, width)
        y1 = torch.clamp(y0 + 1, max=height - 1)
        wx = x - x0.to(dtype=x.dtype)
        wy = y - y0.to(dtype=y.dtype)
        topLeft = self._frameTensor[:, y0, x0]
        topRight = self._frameTensor[:, y0, x1]
        bottomLeft = self._frameTensor[:, y1, x0]
        bottomRight = self._frameTensor[:, y1, x1]
        top = topLeft * (1.0 - wx) + topRight * wx
        bottom = bottomLeft * (1.0 - wx) + bottomRight * wx
        return top * (1.0 - wy) + bottom * wy

    def _normalizeSampledRgb(self, sampled: Any) -> Any:
        rgb = sampled.round().clamp_(0.0, 255.0).div_(255.0).unsqueeze(0)
        return ((rgb - self._mean) / self._std).squeeze(0).contiguous()

    def _synchronizeForProfile(self) -> None:
        if self._profileEnabled:
            self._torch.cuda.synchronize(self._device)


__all__ = ["GpuGeometryImpl"]
