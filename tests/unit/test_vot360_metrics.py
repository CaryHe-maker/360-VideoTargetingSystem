import tempfile
import unittest
from pathlib import Path

import numpy as np

from track360.core.errors import DecodeError, ProtocolError
from track360.core.types import BBoxXYWH
from track360.evaluation.vot360_metrics import evaluateVot360, loadTrackerResults
from track360.io.vot360_results import formatBboxLine

WIDTH = 3840
# A perfect box clears 20 of the 21 thresholds 0, 0.05, ..., 1.0 (IoU 1 is not > 1).
PERFECT_AUC = 20.0 / 21.0


def _truth(rows: list[list[float]], present: list[bool] | None = None):
    mask = np.ones(len(rows), dtype=bool) if present is None else np.asarray(present)
    return {"0001": (np.asarray(rows, dtype=np.float64), mask)}


def _results(rows: list[list[float]]):
    return {"0001": np.asarray(rows, dtype=np.float64)}


class Vot360MetricsTest(unittest.TestCase):
    def testPerfectBboxResults(self) -> None:
        rows = [[100.0, 50.0, 60.0, 40.0], [300.0, 80.0, 60.0, 40.0]]

        scores = evaluateVot360(_truth(rows), _results(rows), "bbox")

        self.assertEqual(scores.successName, "S_dual")
        self.assertAlmostEqual(scores.success, PERFECT_AUC)
        self.assertEqual(scores.precision, 1.0)
        self.assertEqual(scores.anglePrecision, 1.0)
        self.assertEqual((scores.sequenceCount, scores.frameCount), (1, 2))
        self.assertAlmostEqual(scores.perSequence["0001"].success, PERFECT_AUC)
        self.assertEqual(scores.successCurve.shape, (21,))
        self.assertEqual(scores.summary()["S_dual"], scores.success)

    def testAbsentFramesStayInTheDenominator(self) -> None:
        rows = [[100.0, 50.0, 60.0, 40.0], [0.0, 0.0, 0.0, 0.0]]
        results = [[100.0, 50.0, 60.0, 40.0], [5.0, 5.0, 10.0, 10.0]]

        scores = evaluateVot360(_truth(rows, [True, False]), _results(results), "bbox")

        self.assertAlmostEqual(scores.success, PERFECT_AUC / 2.0)
        self.assertEqual(scores.precision, 0.5)
        self.assertEqual(scores.anglePrecision, 0.5)

    def testPrecisionThresholdsAreTwentyPixelsAndThreeDegrees(self) -> None:
        truth = [[1000.0, 900.0, 100.0, 100.0]] * 3
        # Center offsets of 15, 25 and 40 pixels; 3 degrees is 32 pixels at 3840 wide.
        results = [
            [1015.0, 900.0, 100.0, 100.0],
            [1025.0, 900.0, 100.0, 100.0],
            [1040.0, 900.0, 100.0, 100.0],
        ]

        scores = evaluateVot360(_truth(truth), _results(results), "bbox")

        self.assertAlmostEqual(scores.precision, 1.0 / 3.0)
        self.assertAlmostEqual(scores.anglePrecision, 2.0 / 3.0)

    def testWrittenSeamCrossingBoxScoresAgainstBothAnnotationStyles(self) -> None:
        # The same target: 60 px wide, half of it past the seam.
        line = formatBboxLine(BBoxXYWH(WIDTH - 30.0 + 0.5, 50.5, 60.0, 40.0), WIDTH)
        written = [[float(value) for value in line.split(",")]]
        pastLeftBorder = [[-30.0, 50.0, 60.0, 40.0]]
        pastRightBorder = [[WIDTH - 30.0, 50.0, 60.0, 40.0]]

        self.assertEqual(written, pastLeftBorder)
        for truth in (pastLeftBorder, pastRightBorder):
            with self.subTest(truth=truth):
                scores = evaluateVot360(_truth(truth), _results(written), "bbox")
                self.assertAlmostEqual(scores.success, PERFECT_AUC)

    def testToolkitDualSuccessIgnoresPredictionsLeftOnTheRightSide(self) -> None:
        # Documents the toolkit behavior the writer works around.
        scores = evaluateVot360(
            _truth([[-30.0, 50.0, 60.0, 40.0]]),
            _results([[WIDTH - 30.0, 50.0, 60.0, 40.0]]),
            "bbox",
        )

        self.assertEqual(scores.success, 0.0)

    def testBfovScoresSphereSuccessAndAnglePrecision(self) -> None:
        truth = [[10.0, 5.0, 30.0, 20.0, 0.0]] * 2
        near = [[10.5, 5.0, 30.0, 20.0, 0.0], [10.5, 5.0, 30.0, 20.0, 0.0]]
        far = [[10.5, 5.0, 30.0, 20.0, 0.0], [10.0, 12.0, 30.0, 20.0, 0.0]]

        nearScores = evaluateVot360(_truth(truth), _results(near), "bfov")
        farScores = evaluateVot360(_truth(truth), _results(far), "bfov")

        self.assertEqual(nearScores.successName, "S_sphere")
        self.assertIsNone(nearScores.precision)
        self.assertIsNone(nearScores.normPrecision)
        self.assertEqual(nearScores.anglePrecision, 1.0)
        self.assertEqual(farScores.anglePrecision, 0.5)
        self.assertGreater(nearScores.success, farScores.success)
        self.assertGreater(farScores.success, 0.0)

    def testEverySequenceCountsEquallyInTheAggregate(self) -> None:
        box = [100.0, 50.0, 60.0, 40.0]
        miss = [2000.0, 900.0, 60.0, 40.0]
        truth = {
            "short": (np.asarray([box]), np.ones(1, dtype=bool)),
            "long": (np.asarray([box] * 9), np.ones(9, dtype=bool)),
        }
        results = {"short": np.asarray([box]), "long": np.asarray([miss] * 9)}

        scores = evaluateVot360(truth, results, "bbox")

        self.assertAlmostEqual(scores.success, PERFECT_AUC / 2.0)
        self.assertEqual(scores.precision, 0.5)
        self.assertEqual(scores.frameCount, 10)

    def testMismatchedInputsAreRejected(self) -> None:
        truth = _truth([[100.0, 50.0, 60.0, 40.0]] * 2)
        with self.assertRaisesRegex(ProtocolError, "result length"):
            evaluateVot360(truth, _results([[100.0, 50.0, 60.0, 40.0]]), "bbox")
        with self.assertRaisesRegex(ProtocolError, "need 5 columns"):
            evaluateVot360(truth, _results([[1.0, 2.0, 3.0, 4.0]] * 2), "bfov")
        with self.assertRaisesRegex(ProtocolError, "no ground truth"):
            evaluateVot360(truth, {"0002": np.zeros((2, 4))}, "bbox")
        with self.assertRaisesRegex(ProtocolError, "unsupported"):
            evaluateVot360(truth, _results([[1.0, 2.0, 3.0, 4.0, 0.0]] * 2), "rbbox")
        with self.assertRaisesRegex(ProtocolError, "no results"):
            evaluateVot360(truth, {}, "bbox")

    def testTrackerResultsLoadBySequenceName(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0002.txt").write_text("1,2,3,4\n", encoding="utf-8")
            (root / "0001.txt").write_text("5 6 7 8\n", encoding="utf-8")

            results = loadTrackerResults(root)

            self.assertEqual(list(results), ["0001", "0002"])
            np.testing.assert_array_equal(results["0001"], [[5.0, 6.0, 7.0, 8.0]])
            with self.assertRaises(DecodeError):
                loadTrackerResults(root / "missing")
        with tempfile.TemporaryDirectory() as empty, self.assertRaises(DecodeError):
            loadTrackerResults(empty)


if __name__ == "__main__":
    unittest.main()
