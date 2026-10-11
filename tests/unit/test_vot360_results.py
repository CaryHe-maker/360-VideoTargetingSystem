import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from track360.core.errors import DecodeError, OutputError
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    SequenceId,
    TrackResult,
    TrackStatus,
)
from track360.geometry.projection_math import makeSphericalPoint
from track360.io.vot360_results import (
    ResultCollector,
    formatBboxLine,
    formatBfovLine,
    readResultFile,
    resultPaths,
    writeSequenceResults,
)


def _result(index: int, xPx: float = 100.0) -> TrackResult:
    return TrackResult(
        sequenceId=SequenceId("0001"),
        frameIndex=FrameIndex(index),
        bbox=BBoxXYWH(xPx, 50.0, 40.0, 20.0),
        bfov=BFoV(
            makeSphericalPoint(math.radians(-30.0), math.radians(12.5)),
            math.radians(20.0),
            math.radians(10.0),
        ),
        confidence=0.5,
        status=TrackStatus.TRACKING,
        valid=True,
    )


class Vot360ResultsTest(unittest.TestCase):
    def testBboxLineUsesPixelIndexCoordinates(self) -> None:
        self.assertEqual(
            formatBboxLine(BBoxXYWH(100.0, 50.0, 40.0, 20.0), 3840),
            "99.5000,49.5000,40.0000,20.0000",
        )

    def testSeamCrossingBoxIsWrittenWithNegativeX(self) -> None:
        self.assertEqual(
            formatBboxLine(BBoxXYWH(3820.0, 50.0, 40.0, 20.0), 3840),
            "-20.5000,49.5000,40.0000,20.0000",
        )
        # A box that ends exactly at the border does not cross it.
        self.assertEqual(
            formatBboxLine(BBoxXYWH(3800.0, 50.0, 40.0, 20.0), 3840),
            "3799.5000,49.5000,40.0000,20.0000",
        )

    def testBfovLineIsInDegreesWithRotation(self) -> None:
        self.assertEqual(
            formatBfovLine(_result(0).bfov), "-30.0000,12.5000,20.0000,10.0000,0.0000"
        )

    def testSequenceResultsUseTheToolkitDirectoryLayout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bboxPath, bfovPath = writeSequenceResults(
                root, "ours", "0001", [_result(0), _result(1, 3820.0)], 3840
            )

            self.assertEqual(bboxPath, root / "bbox" / "ours" / "0001.txt")
            self.assertEqual(bfovPath, root / "bfov" / "ours" / "0001.txt")
            self.assertEqual(resultPaths(root, "ours", "0001"), (bboxPath, bfovPath))
            self.assertFalse(list(root.rglob("*.partial")))
            np.testing.assert_array_equal(
                readResultFile(bboxPath),
                [[99.5, 49.5, 40.0, 20.0], [-20.5, 49.5, 40.0, 20.0]],
            )
            self.assertEqual(readResultFile(bfovPath).shape, (2, 5))
            with self.assertRaises(OutputError):
                writeSequenceResults(root, "ours", "0002", [], 3840)

    def testResultFilesMayBeWhitespaceSeparated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0001.txt"
            path.write_text("1 2 3 4\n5 6 7 8\n", encoding="utf-8")
            np.testing.assert_array_equal(readResultFile(path), [[1, 2, 3, 4], [5, 6, 7, 8]])

            path.write_text("1,2,3,4\n5,6,7\n", encoding="utf-8")
            with self.assertRaisesRegex(DecodeError, "ragged"):
                readResultFile(path)
            path.write_text("1,2,x,4\n", encoding="utf-8")
            with self.assertRaisesRegex(DecodeError, "invalid result line"):
                readResultFile(path)
            with self.assertRaises(DecodeError):
                readResultFile(Path(directory) / "missing.txt")

    def testCollectorChecksFrameOrderAndCount(self) -> None:
        collector = ResultCollector()
        collector.open("unused")
        collector.write(_result(0))
        with self.assertRaisesRegex(OutputError, "order mismatch"):
            collector.write(_result(2))
        collector.write(_result(1))
        collector.finalize(2)
        with self.assertRaisesRegex(OutputError, "count mismatch"):
            collector.finalize(3)
        self.assertEqual(len(collector.results), 2)


if __name__ == "__main__":
    unittest.main()
