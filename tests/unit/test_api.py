from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from track360.api import (
    PRESETS,
    Track360Tracker,
    bfovFromDegrees,
    boxFromPixels,
    presetPath,
    resolveConfig,
)
from track360.core.errors import ConfigError, OutputError
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    FramePacket,
    LocalObservation,
    LocalView,
    SequenceId,
    TrackResult,
    TrackStatus,
    ViewSpec,
)
from track360.geometry import makeSphericalPoint
from track360.runtime.track_video import EXIT_CONFIG
from track360.runtime.track_video import main as trackMain
from track360.visualization.demo import DemoRecorder


class PresetTest(unittest.TestCase):
    def testEveryPresetLoadsAndTheyDifferOnlyInLossHandling(self) -> None:
        for name in PRESETS:
            self.assertTrue(presetPath(name).is_file(), name)
        default = resolveConfig("default")
        loss = resolveConfig("loss_handling")
        self.assertFalse(default.backendTuning.lossHandling)
        self.assertTrue(loss.backendTuning.lossHandling)
        self.assertEqual(default.geometry, loss.geometry)
        self.assertEqual(default.model, loss.model)

    def testPrecisionAndWeightsOverrideThePreset(self) -> None:
        config = resolveConfig("default", precision="tf32", weights="some/weights.pth")
        self.assertEqual(config.model.precision, "tf32")
        self.assertEqual(config.model.weights.name, "weights.pth")
        self.assertTrue(config.model.weights.is_absolute())
        with self.assertRaises(ConfigError):
            resolveConfig("default", precision="fp8")
        with self.assertRaises(ConfigError):
            resolveConfig("fastest")

    def testATrackerIsBuiltWithoutLoadingTheWeights(self) -> None:
        tracker = Track360Tracker.fromPretrained("loss_handling", precision="tf32")
        self.assertTrue(tracker.config.backendTuning.lossHandling)
        self.assertEqual(tracker.config.model.precision, "tf32")
        with self.assertRaisesRegex(ConfigError, "exactly one"):
            tracker.track("unused")
        with self.assertRaisesRegex(ConfigError, "exactly one"):
            tracker.track("unused", initBox=(0, 0, 1, 1), initBfov=(0, 0, 10, 10))
        tracker.close()


class TargetParsingTest(unittest.TestCase):
    def testBfovIsGivenInDegrees(self) -> None:
        bfov = bfovFromDegrees((90.0, -30.0, 20.0, 40.0))
        self.assertAlmostEqual(bfov.center.yawRad, math.pi / 2.0)
        self.assertAlmostEqual(bfov.center.pitchRad, -math.pi / 6.0)
        self.assertAlmostEqual(bfov.horizontalFovRad, math.radians(20.0))
        self.assertAlmostEqual(bfov.verticalFovRad, math.radians(40.0))
        for bad in ((200.0, 0.0, 10.0, 10.0), (0.0, 0.0, 0.0, 10.0), (0.0, 0.0, 10.0)):
            with self.assertRaises(ConfigError):
                bfovFromDegrees(bad)

    def testBoxIsGivenInPixels(self) -> None:
        self.assertEqual(boxFromPixels(("10", 20, 30.5, 40)), BBoxXYWH(10.0, 20.0, 30.5, 40.0))
        with self.assertRaises(ConfigError):
            boxFromPixels((1, 2, 3))

    def testTheCommandRejectsARunThatWritesNothing(self) -> None:
        code = trackMain(["--input", "missing", "--init-box", "1,2,3,4"])
        self.assertEqual(code, EXIT_CONFIG)


class DemoRecorderTest(unittest.TestCase):
    def _frame(self, index: int) -> FramePacket:
        rgb = np.full((120, 240, 3), 40 + index, dtype=np.uint8)
        return FramePacket(SequenceId("s"), FrameIndex(index), index, rgb)

    def _result(self, index: int, xPx: float) -> TrackResult:
        return TrackResult(
            sequenceId=SequenceId("s"),
            frameIndex=FrameIndex(index),
            bbox=BBoxXYWH(xPx, 40.0, 40.0, 30.0),
            bfov=BFoV(makeSphericalPoint(0.0, 0.0), 0.3, 0.3),
            confidence=0.8,
            status=TrackStatus.TRACKING if index % 2 else TrackStatus.UNCERTAIN,
            valid=True,
        )

    def testWritesAVideoAndAGifWithTheBoxAcrossTheSeam(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video, gif = Path(directory) / "out" / "demo.mp4", Path(directory) / "demo.gif"
            recorder = DemoRecorder(video, gifPath=gif, heightPx=120, gifEvery=2, gifWidthPx=180)
            spec = ViewSpec(0, BFoV(makeSphericalPoint(0.0, 0.0), 1.0, 1.0), 64, 64)
            view = LocalView(spec, np.full((64, 64, 3), 90, dtype=np.uint8))
            box = LocalObservation(
                viewId=0,
                bbox=BBoxXYWH(20.0, 20.0, 24.0, 24.0),
                modelScore=0.8,
                appearanceScore=0.8,
                fusedScore=0.8,
                latencyNs=1,
            )
            canvases = []
            for index in range(6):
                frame = self._frame(index)
                recorder.recordLocalRgb(frame, (view,))
                recorder.recordBackendBoxes(frame, (view,), (box,))
                recorder.recordGeometryBoxes(frame, ())
                # The last frames put the box across the right edge of the panorama.
                result = self._result(index, 100.0 if index < 4 else 220.0)
                canvases.append(recorder._compose(frame, result))
                recorder.record(frame, result)
            recorder.close()

            self.assertGreater(video.stat().st_size, 0)
            self.assertGreater(gif.stat().st_size, 0)
            # Panorama 240 x 120 next to the 120 x 120 tracker view.
            self.assertEqual(canvases[0].shape, (120, 360, 3))
            wrapped = canvases[5][:, :240]
            background = 40 + 5
            # The box that runs off the right edge is drawn again from the left one.
            self.assertTrue((wrapped[60, :22] != background).any())
            self.assertTrue((wrapped[60, 218:] != background).any())
            self.assertTrue((canvases[0][60, :22] == 40).all())

    def testADemoNeedsSomewhereToGo(self) -> None:
        with self.assertRaises(OutputError):
            DemoRecorder(None)


if __name__ == "__main__":
    unittest.main()
