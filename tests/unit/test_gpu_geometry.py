import unittest

import numpy as np

from track360.core.types import BFoV, FrameIndex, FramePacket, SequenceId, ViewSpec
from track360.geometry import SphericalGeometryImpl, makeSphericalPoint
from track360.geometry.gpu_geometry import GpuGeometryImpl


class GpuGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest(f"PyTorch is unavailable: {error}") from error

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for GPU Geometry regression")

    def testPerspectiveCropsMatchCpuAndStayOnDevice(self) -> None:
        height, width = 96, 192
        x = np.arange(width, dtype=np.uint16)[None, :]
        y = np.arange(height, dtype=np.uint16)[:, None]
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = (x * 3 + y * 5) % 256
        rgb[..., 1] = (x * 7 + y * 2) % 256
        rgb[..., 2] = (x * 11 + y * 13) % 256
        frame = FramePacket(SequenceId("gpu-regression"), FrameIndex(0), 0, rgb)
        specs = (
            ViewSpec(0, BFoV(makeSphericalPoint(0.0, 0.0), 1.2, 1.0), 256, 256),
            ViewSpec(1, BFoV(makeSphericalPoint(3.12, 0.1), 1.4, 1.1), 256, 256),
            ViewSpec(2, BFoV(makeSphericalPoint(-1.0, 1.25), 1.0, 0.6), 256, 256),
        )
        cpu = SphericalGeometryImpl().cropViews(frame, specs)
        gpuGeometry = GpuGeometryImpl()
        try:
            gpu = gpuGeometry.cropViews(frame, specs)
            mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
            std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]

            for cpuView, gpuView in zip(cpu, gpu, strict=True):
                actual = gpuView.deviceRgb.detach().cpu().numpy()
                self.assertEqual(str(gpuView.deviceRgb.device), "cuda:0")
                self.assertEqual(str(gpuView.deviceRgb.dtype), "torch.float32")
                reconstructed = np.rint((actual * std + mean) * 255.0).clip(0, 255)
                pixelError = np.abs(reconstructed - cpuView.rgb.transpose(2, 0, 1))
                self.assertLessEqual(float(pixelError.max()), 1.0)
                self.assertEqual(float(np.percentile(pixelError, 99)), 0.0)

            self.assertEqual(gpuGeometry.lastProfile["imageRoundTrips"], 0)
            self.assertEqual(gpuGeometry.lastProfile["gridCpuTransfers"], 0)
            self.assertEqual(gpuGeometry.lastProfile["frameTensorDevice"], "cuda:0")
        finally:
            gpuGeometry.close()

    def testSameFrameReusesUploadAndReleaseDropsIt(self) -> None:
        rgb = np.full((32, 64, 3), 90, dtype=np.uint8)
        frame = FramePacket(SequenceId("gpu-lifecycle"), FrameIndex(0), 0, rgb)
        spec = ViewSpec(0, BFoV(makeSphericalPoint(0.0, 0.0), 1.0, 1.0), 32, 32)
        geometry = GpuGeometryImpl()
        try:
            geometry.cropViews(frame, [spec])
            firstTensor = geometry._frameTensor
            geometry.cropViews(frame, [spec])
            self.assertIs(geometry._frameTensor, firstTensor)
            geometry.releaseFrame()
            self.assertIsNone(geometry._frameTensor)
            self.assertIsNone(geometry._hostTensor)
            self.assertIsNone(geometry._frameIndex)
        finally:
            geometry.close()

    def testFrameChangeRefreshesUploadAndKeepsOutputFinite(self) -> None:
        import torch

        spec = ViewSpec(0, BFoV(makeSphericalPoint(2.9, 0.2), 1.3, 1.1), 48, 40)
        geometry = GpuGeometryImpl()
        try:
            frame0 = FramePacket(
                SequenceId("gpu-refresh"), FrameIndex(0), 0, np.zeros((40, 80, 3), dtype=np.uint8)
            )
            frame1 = FramePacket(
                SequenceId("gpu-refresh"),
                FrameIndex(1),
                1,
                np.full((40, 80, 3), 255, dtype=np.uint8),
            )
            first = geometry.cropViews(frame0, [spec])[0].deviceRgb
            second = geometry.cropViews(frame1, [spec])[0].deviceRgb
            self.assertTrue(bool(torch.isfinite(first).all()))
            self.assertTrue(bool(torch.isfinite(second).all()))
            self.assertFalse(bool(torch.equal(first, second)))
            self.assertEqual(geometry._frameIndex, 1)
        finally:
            geometry.close()

    def testGridCacheIsKeyedByOutputShape(self) -> None:
        geometry = GpuGeometryImpl()
        try:
            first = geometry._localPixelGrid(32, 16)
            second = geometry._localPixelGrid(32, 16)
            third = geometry._localPixelGrid(16, 32)
            self.assertIs(first, second)
            self.assertIsNot(first, third)
            self.assertEqual(set(geometry._localGridCache), {(32, 16), (16, 32)})
        finally:
            geometry.close()

    def testCloseIsIdempotentAndReleasesPersistentState(self) -> None:
        geometry = GpuGeometryImpl()
        geometry._localPixelGrid(16, 16)
        geometry.close()
        geometry.close()
        self.assertTrue(geometry._closed)
        self.assertEqual(geometry._localGridCache, {})
        self.assertIsNone(geometry._mean)
        self.assertIsNone(geometry._std)


if __name__ == "__main__":
    unittest.main()
