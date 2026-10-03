import tempfile
import unittest
from pathlib import Path

import numpy as np

from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    SequenceId,
    SphericalPoint,
    TrackResult,
    TrackStatus,
)
from track360.io.image_reader import readRgbImage
from track360.io.result_sink import FileResultSink
from track360.io.video_source import VideoFrameSource
from track360.visualization.png import writeRgbPng


class IoRuntimeTest(unittest.TestCase):
    def testVideoFrameSourceReadsDirectorySequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(2):
                writeRgbPng(
                    root / f"frame_{index:04d}.png",
                    np.full((4, 6, 3), index * 32, dtype=np.uint8),
                )

            source = VideoFrameSource()
            source.open(str(root))

            first = source.read()
            second = source.read()

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            assert first is not None and second is not None
            self.assertEqual(int(first.frameIndex), 0)
            self.assertEqual(int(second.frameIndex), 1)
            self.assertEqual(first.rgb.shape, (4, 6, 3))
            np.testing.assert_array_equal(
                readRgbImage(root / "frame_0000.png")[0, 0],
                np.array([0, 0, 0], dtype=np.uint8),
            )

    def testResultSinkWritesAtomicTextFile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.txt"
            sink = FileResultSink()
            sink.open(str(output))
            result0 = _result(0, 0.0)
            result1 = _result(1, 1.0)
            sink.write(result0)
            sink.write(result1)
            sink.finalize(2)

            self.assertTrue(output.exists())
            self.assertEqual(
                output.read_text(encoding="utf-8").splitlines(),
                ["0.000000,0.000000,2.000000,3.000000", "1.000000,1.000000,2.000000,3.000000"],
            )


def _result(frameIndex: int, offset: float) -> TrackResult:
    return TrackResult(
        sequenceId=SequenceId("sequence"),
        frameIndex=FrameIndex(frameIndex),
        bbox=BBoxXYWH(offset, offset, 2.0, 3.0),
        bfov=BFoV(
            center=SphericalPoint(1.0, 0.0, 0.0, 0.0, 0.0),
            horizontalFovRad=0.5,
            verticalFovRad=0.5,
        ),
        confidence=0.9,
        status=TrackStatus.TRACKING,
        valid=True,
    )
