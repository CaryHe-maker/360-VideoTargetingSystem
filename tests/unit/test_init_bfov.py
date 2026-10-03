import math
import unittest
from pathlib import Path

import numpy as np

from track360.controller import TrackControllerImpl
from track360.core.config import loadConfig
from track360.core.errors import ConfigError, ProtocolError
from track360.core.types import BBoxXYWH, BFoV, FrameIndex, FramePacket, SequenceId
from track360.geometry import SphericalGeometryImpl
from track360.geometry.projection_math import makeSphericalPoint
from track360.runtime.track_video import _parseBfov, buildParser

ROOT = Path(__file__).resolve().parents[2]


class InitBfovTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = loadConfig(ROOT / "configs" / "default.yaml")
        self.geometry = SphericalGeometryImpl(
            boundarySamplesPerEdge=self.config.geometry.boundarySamplesPerEdge
        )
        self.frame = FramePacket(
            SequenceId("init"), FrameIndex(0), 0, np.zeros((480, 960, 3), dtype=np.uint8)
        )

    def testBfovInitializationCentersTheTemplateOnTheTarget(self) -> None:
        target = BFoV(makeSphericalPoint(math.radians(170.0), math.radians(20.0)), 0.4, 0.3)
        controller = TrackControllerImpl(self.geometry, self.config)

        plan = controller.buildInitialization(self.frame, initialBfov=target)
        result = controller.commitInitialization(plan)

        scale = self.config.backendTuning.templateFovScale
        self.assertEqual(plan.templateView.bfov.center, target.center)
        self.assertAlmostEqual(plan.templateView.bfov.horizontalFovRad, scale * 0.4)
        self.assertAlmostEqual(plan.templateView.bfov.verticalFovRad, scale * 0.3)
        # The template view is `scale` times the target, so the target fills 1/scale of it
        # along each tangent axis.
        expectedWidth = 256.0 * math.tan(0.2) / math.tan(scale * 0.2)
        self.assertAlmostEqual(plan.templateBox.widthPx, expectedWidth, places=6)
        self.assertEqual(result.bfov, target)
        self.assertEqual(result.bbox, self.geometry.bfovToBbox(target, 960, 480))

    def testBfovAndBoxInitializationAgreeForTheSameTarget(self) -> None:
        box = BBoxXYWH(400.0, 200.0, 60.0, 40.0)
        fromBox = TrackControllerImpl(self.geometry, self.config).buildInitialization(
            self.frame, box
        )
        bfov = self.geometry.bboxToBfov(box, 960, 480)
        fromBfov = TrackControllerImpl(self.geometry, self.config).buildInitialization(
            self.frame, initialBfov=bfov
        )

        self.assertEqual(fromBfov.templateView, fromBox.templateView)
        self.assertEqual(fromBfov.templateBox, fromBox.templateBox)

    def testInitializationRequiresExactlyOneTarget(self) -> None:
        target = BFoV(makeSphericalPoint(0.0, 0.0), 0.4, 0.3)
        controller = TrackControllerImpl(self.geometry, self.config)

        with self.assertRaisesRegex(ProtocolError, "exactly one"):
            controller.buildInitialization(self.frame)
        with self.assertRaisesRegex(ProtocolError, "exactly one"):
            controller.buildInitialization(
                self.frame, BBoxXYWH(1.0, 1.0, 5.0, 5.0), initialBfov=target
            )

    def testCommandLineParsesDegreesInThe360VotConvention(self) -> None:
        bfov = _parseBfov("-21.5, 10, 30, 20")

        self.assertAlmostEqual(bfov.center.yawRad, math.radians(-21.5))
        self.assertAlmostEqual(bfov.center.pitchRad, math.radians(10.0))
        self.assertAlmostEqual(bfov.horizontalFovRad, math.radians(30.0))
        self.assertAlmostEqual(bfov.verticalFovRad, math.radians(20.0))
        for invalid in ("1,2,3", "0,95,10,10", "0,0,0,10", "0,0,200,10", "a,b,c,d"):
            with self.subTest(value=invalid), self.assertRaises(ConfigError):
                _parseBfov(invalid)

    def testCommandLineRequiresExactlyOneInitialTarget(self) -> None:
        parser = buildParser()
        common = ["--input", "frames", "--output", "out.txt", "--config", "c.yaml"]

        self.assertEqual(parser.parse_args([*common, "--init-bfov", "0,0,10,10"]).init_box, None)
        with self.assertRaises(SystemExit):
            parser.parse_args(common)
        with self.assertRaises(SystemExit):
            parser.parse_args([*common, "--init-box", "1,2,3,4", "--init-bfov", "0,0,10,10"])


if __name__ == "__main__":
    unittest.main()
