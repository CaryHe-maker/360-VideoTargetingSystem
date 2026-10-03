import math
import unittest

import numpy as np

from track360.core.errors import GeometryError
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    FramePacket,
    SequenceId,
    ViewSpec,
)
from track360.geometry import (
    BfovProjector,
    SphericalGeometryImpl,
    clampPitch,
    containsCircularX,
    erpPixelToSphericalPoint,
    localPixelsToUnitVectors,
    makeSphericalPoint,
    sphericalPointToErpPixel,
    splitSeamBox,
    unitVectorsToErpPixels,
    wrapYaw,
)
from track360.geometry.spherical_geometry import _fitBfovFromVectors


class GeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = SphericalGeometryImpl(boundarySamplesPerEdge=33)

    def testProjectionMathHandlesCornersAndWrap(self) -> None:
        self.assertAlmostEqual(wrapYaw(3.0 * math.pi), -math.pi)
        self.assertAlmostEqual(clampPitch(10.0), math.pi / 2.0)

        frameWidthPx = 8
        frameHeightPx = 4
        topLeft = erpPixelToSphericalPoint(0.0, 0.0, frameWidthPx, frameHeightPx)
        center = erpPixelToSphericalPoint(4.0, 2.0, frameWidthPx, frameHeightPx)
        bottomRight = erpPixelToSphericalPoint(8.0, 4.0, frameWidthPx, frameHeightPx)

        self.assertAlmostEqual(topLeft.yawRad, -math.pi)
        self.assertAlmostEqual(topLeft.pitchRad, math.pi / 2.0)
        self.assertAlmostEqual(center.yawRad, 0.0)
        self.assertAlmostEqual(center.pitchRad, 0.0)
        self.assertAlmostEqual(bottomRight.yawRad, -math.pi)
        self.assertAlmostEqual(bottomRight.pitchRad, -math.pi / 2.0)

        xPx, yPx = sphericalPointToErpPixel(
            makeSphericalPoint(0.0, 0.0),
            frameWidthPx,
            frameHeightPx,
        )
        self.assertAlmostEqual(xPx, 4.0)
        self.assertAlmostEqual(yPx, 2.0)

    def testCropViewsPreservesOrderAndRgbPixels(self) -> None:
        rgb = np.zeros((5, 5, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(5, dtype=np.uint8)
        rgb[..., 1] = np.arange(5, dtype=np.uint8)[:, np.newaxis]
        rgb[..., 2] = 7

        frame = FramePacket(
            sequenceId=SequenceId("sequence"),
            frameIndex=FrameIndex(0),
            timestampNs=0,
            rgb=rgb,
        )
        center = erpPixelToSphericalPoint(2.5, 2.5, 5, 5)
        bfov = BFoV(center=center, horizontalFovRad=0.4, verticalFovRad=0.4)
        spec1 = ViewSpec(viewId=7, bfov=bfov, outputWidthPx=1, outputHeightPx=1)
        spec2 = ViewSpec(viewId=2, bfov=bfov, outputWidthPx=1, outputHeightPx=1)

        views = self.geometry.cropViews(frame, [spec1, spec2])

        self.assertEqual([view.spec.viewId for view in views], [7, 2])
        self.assertTrue(np.array_equal(views[0].rgb[0, 0], rgb[2, 2]))

    def testSeamHelpersSplitAndContainCircularBoxes(self) -> None:
        bbox = BBoxXYWH(xPx=14.0, yPx=1.0, widthPx=4.0, heightPx=2.0)
        parts = splitSeamBox(bbox, 16)

        self.assertEqual(len(parts), 2)
        self.assertAlmostEqual(parts[0].xPx, 14.0)
        self.assertAlmostEqual(parts[0].widthPx, 2.0)
        self.assertAlmostEqual(parts[1].xPx, 0.0)
        self.assertAlmostEqual(parts[1].widthPx, 2.0)
        self.assertTrue(containsCircularX(1.0, bbox, 16))
        self.assertFalse(containsCircularX(6.0, bbox, 16))

    def testBfovToBboxContainsSampledBoundary(self) -> None:
        frameWidthPx = 360
        frameHeightPx = 180
        center = erpPixelToSphericalPoint(180.0, 60.0, frameWidthPx, frameHeightPx)
        bfov = BFoV(center=center, horizontalFovRad=1.35, verticalFovRad=1.0, rollRad=0.25)
        bbox = self.geometry.bfovToBbox(bfov, frameWidthPx, frameHeightPx)
        sampleX, sampleY = _sampleCanonicalBoundary(self.geometry.boundarySamplesPerEdge)
        vectors = localPixelsToUnitVectors(
            sampleX,
            sampleY,
            bfov,
            1,
            1,
        )
        erpX, erpY = unitVectorsToErpPixels(vectors, frameWidthPx, frameHeightPx)

        for xPx, yPx in zip(erpX, erpY, strict=True):
            self.assertTrue(containsCircularX(float(xPx), bbox, frameWidthPx))
            self.assertGreaterEqual(float(yPx), bbox.yPx - 1e-6)
            self.assertLessEqual(float(yPx), bbox.yPx + bbox.heightPx + 1e-6)

    def testBboxToBfovRoundTripContainsSampledBoundary(self) -> None:
        frameWidthPx = 16
        frameHeightPx = 8
        bbox = BBoxXYWH(xPx=13.5, yPx=1.0, widthPx=4.5, heightPx=3.0)
        bfov = self.geometry.bboxToBfov(bbox, frameWidthPx, frameHeightPx)
        roundTrip = self.geometry.bfovToBbox(bfov, frameWidthPx, frameHeightPx)
        sampleX, sampleY = _sampleBoxBoundary(bbox, self.geometry.boundarySamplesPerEdge)

        for xPx, yPx in zip(sampleX, sampleY, strict=True):
            self.assertTrue(containsCircularX(float(xPx), roundTrip, frameWidthPx))
            self.assertGreaterEqual(float(yPx), roundTrip.yPx - 1e-6)
            self.assertLessEqual(float(yPx), roundTrip.yPx + roundTrip.heightPx + 1e-6)

    def testBboxToBfovHandlesWrappedCameraAngles(self) -> None:
        frameWidthPx = 360
        frameHeightPx = 180
        bbox = BBoxXYWH(xPx=0.0, yPx=6.0, widthPx=174.0, heightPx=126.0)

        bfov = self.geometry.bboxToBfov(bbox, frameWidthPx, frameHeightPx)
        roundTrip = self.geometry.bfovToBbox(bfov, frameWidthPx, frameHeightPx)
        sampleX, sampleY = _sampleBoxBoundary(bbox, self.geometry.boundarySamplesPerEdge)

        self.assertLess(bfov.horizontalFovRad, np.pi)
        self.assertLess(bfov.verticalFovRad, np.pi)
        for xPx, yPx in zip(sampleX, sampleY, strict=True):
            self.assertTrue(containsCircularX(float(xPx), roundTrip, frameWidthPx))
            self.assertGreaterEqual(float(yPx), roundTrip.yPx - 1e-6)
            self.assertLessEqual(float(yPx), roundTrip.yPx + roundTrip.heightPx + 1e-6)

    def testLocalBoxToBfovUsesBoundarySamples(self) -> None:
        frameWidthPx = 360
        frameHeightPx = 180
        specCenter = erpPixelToSphericalPoint(180.0, 90.0, frameWidthPx, frameHeightPx)
        specBfov = BFoV(center=specCenter, horizontalFovRad=1.4, verticalFovRad=1.2, rollRad=0.3)
        spec = ViewSpec(viewId=1, bfov=specBfov, outputWidthPx=256, outputHeightPx=256)
        localBox = BBoxXYWH(xPx=180.0, yPx=10.0, widthPx=60.0, heightPx=180.0)

        bfov = self.geometry.localBoxToBfov(localBox, spec)
        bbox = self.geometry.bfovToBbox(bfov, frameWidthPx, frameHeightPx)
        sampleX, sampleY = _sampleBoxBoundary(localBox, self.geometry.boundarySamplesPerEdge)
        vectors = localPixelsToUnitVectors(
            sampleX,
            sampleY,
            spec.bfov,
            spec.outputWidthPx,
            spec.outputHeightPx,
        )
        erpX, erpY = unitVectorsToErpPixels(vectors, frameWidthPx, frameHeightPx)

        for xPx, yPx in zip(erpX, erpY, strict=True):
            self.assertTrue(containsCircularX(float(xPx), bbox, frameWidthPx))
            self.assertGreaterEqual(float(yPx), bbox.yPx - 1e-6)
            self.assertLessEqual(float(yPx), bbox.yPx + bbox.heightPx + 1e-6)

    def testDirectLocalBoundaryProjectionAvoidsSecondEnvelope(self) -> None:
        frameWidthPx = 360
        frameHeightPx = 180
        spec = ViewSpec(
            viewId=3,
            bfov=BFoV(
                center=erpPixelToSphericalPoint(300.0, 45.0, frameWidthPx, frameHeightPx),
                horizontalFovRad=1.8,
                verticalFovRad=1.4,
            ),
            outputWidthPx=256,
            outputHeightPx=256,
        )
        projection = self.geometry.projectLocalBoxBoundary(
            BBoxXYWH(170.0, 20.0, 70.0, 150.0),
            spec,
            frameWidthPx,
            frameHeightPx,
        )

        self.assertEqual(
            len(projection.erpBoundary),
            4 * self.geometry.boundarySamplesPerEdge,
        )
        self.assertLessEqual(
            projection.bbox.widthPx * projection.bbox.heightPx,
            projection.indirectBbox.widthPx * projection.indirectBbox.heightPx + 1e-6,
        )
        self.assertGreaterEqual(projection.envelopeInflation, 1.0 - 1e-9)
        for xPx, yPx in projection.erpBoundary:
            self.assertTrue(containsCircularX(xPx, projection.bbox, frameWidthPx))
            self.assertGreaterEqual(yPx, projection.bbox.yPx - 1e-6)
            self.assertLessEqual(yPx, projection.bbox.yPx + projection.bbox.heightPx + 1e-6)

    def testGeometryRejectsInvalidInputs(self) -> None:
        with self.assertRaises(GeometryError):
            SphericalGeometryImpl(boundarySamplesPerEdge=1)
        with self.assertRaises(GeometryError):
            BfovProjector(boundarySamplesPerEdge=1)
        with self.assertRaises(GeometryError):
            self.geometry.bboxToBfov(BBoxXYWH(xPx=0.0, yPx=-1.0, widthPx=2.0, heightPx=2.0), 8, 8)
        with self.assertRaises(GeometryError):
            self.geometry.localBoxToBfov(
                BBoxXYWH(xPx=0.0, yPx=0.0, widthPx=300.0, heightPx=10.0),
                ViewSpec(
                    viewId=0,
                    bfov=BFoV(
                        center=makeSphericalPoint(0.0, 0.0),
                        horizontalFovRad=0.5,
                        verticalFovRad=0.5,
                    ),
                    outputWidthPx=256,
                    outputHeightPx=256,
                ),
            )

    def testWideCircularSpanSaturatesInsteadOfAborting(self) -> None:
        # This reproduces the production failure mode (a span around 6.1356 rad)
        # when a projected boundary covers almost the complete yaw circle.
        angles = np.linspace(0.0, 2.0 * math.pi - 0.1475, 129, dtype=np.float64)
        vectors = np.column_stack(
            (np.sin(angles), np.zeros_like(angles), np.cos(angles))
        )

        bfov = _fitBfovFromVectors(vectors)

        self.assertGreater(bfov.horizontalFovRad, 0.0)
        self.assertLess(bfov.horizontalFovRad, math.pi)
        self.assertGreater(bfov.verticalFovRad, 0.0)
        self.assertLess(bfov.verticalFovRad, math.pi)


def _sampleBoxBoundary(bbox: BBoxXYWH, samplesPerEdge: int) -> tuple[np.ndarray, np.ndarray]:
    edge = np.linspace(0.0, 1.0, samplesPerEdge, dtype=np.float64)
    topX = bbox.xPx + edge * bbox.widthPx
    rightX = np.full_like(edge, bbox.xPx + bbox.widthPx)
    bottomX = bbox.xPx + (1.0 - edge) * bbox.widthPx
    leftX = np.full_like(edge, bbox.xPx)
    topY = np.full_like(edge, bbox.yPx)
    rightY = bbox.yPx + edge * bbox.heightPx
    bottomY = np.full_like(edge, bbox.yPx + bbox.heightPx)
    leftY = bbox.yPx + (1.0 - edge) * bbox.heightPx
    return (
        np.concatenate((topX, rightX, bottomX, leftX)),
        np.concatenate((topY, rightY, bottomY, leftY)),
    )


def _sampleCanonicalBoundary(samplesPerEdge: int) -> tuple[np.ndarray, np.ndarray]:
    edge = np.linspace(0.0, 1.0, samplesPerEdge, dtype=np.float64)
    topX = edge
    rightX = np.ones_like(edge)
    bottomX = 1.0 - edge
    leftX = np.zeros_like(edge)
    topY = np.zeros_like(edge)
    rightY = edge
    bottomY = np.ones_like(edge)
    leftY = 1.0 - edge
    return (
        np.concatenate((topX, rightX, bottomX, leftX)),
        np.concatenate((topY, rightY, bottomY, leftY)),
    )


if __name__ == "__main__":
    unittest.main()
