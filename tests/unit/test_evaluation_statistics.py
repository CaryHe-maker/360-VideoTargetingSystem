import unittest
from pathlib import Path

import numpy as np

from track360.core.errors import ProtocolError
from track360.datasets.tune_split import readSequenceFile
from track360.evaluation.bootstrap import bootstrapDifference, bootstrapRatio
from track360.evaluation.comparison import compareScores, hardRegressions
from track360.evaluation.loss_rate import (
    LOST_IOU_THRESHOLD,
    LOST_MIN_RUN_FRAMES,
    dualIou,
    lostFrameMask,
    sequenceLoss,
)
from track360.evaluation.vot360_metrics import evaluateVot360

ROOT = Path(__file__).resolve().parents[2]
WIDTH = 3840
HIT = [100.0, 50.0, 60.0, 40.0]
MISS = [2000.0, 900.0, 60.0, 40.0]


def _mask(pattern: str) -> np.ndarray:
    """``x``: IoU below the threshold, ``o``: tracked, ``-``: target absent."""
    iou = np.asarray([0.0 if char == "x" else 0.5 for char in pattern])
    present = np.asarray([char != "-" for char in pattern])
    return lostFrameMask(iou, present)


def _lost(pattern: str) -> str:
    return "".join("L" if value else "." for value in _mask(pattern))


def _scores(perSequence: dict[str, list[list[float]]]):
    truth = {
        name: (np.asarray([HIT] * len(rows)), np.ones(len(rows), bool))
        for name, rows in perSequence.items()
    }
    results = {name: np.asarray(rows) for name, rows in perSequence.items()}
    return evaluateVot360(truth, results, "bbox")


class LossRateTest(unittest.TestCase):
    def testDefinitionIsFiveConsecutiveFramesBelowOneTenth(self) -> None:
        self.assertEqual((LOST_IOU_THRESHOLD, LOST_MIN_RUN_FRAMES), (0.1, 5))

    def testRunsOfFiveOrMoreAreLostAndShorterDipsAreNot(self) -> None:
        self.assertEqual(_lost("oxxxxo"), "......")
        self.assertEqual(_lost("oxxxxxo"), ".LLLLL.")
        self.assertEqual(_lost("oxxxxxxxo"), ".LLLLLLL.")
        self.assertEqual(_lost("xxxxxoxxxxoxxxxxx"), "LLLLL......LLLLLL")

    def testRecoveryEndsALossAndTheNextOneNeedsAFullRunAgain(self) -> None:
        self.assertEqual(_lost("xxxxxxoxxo"), "LLLLLL....")

    def testAbsentFramesAreNeverLostAndDoNotInterruptARun(self) -> None:
        self.assertEqual(_lost("xx--xxxo"), "LL..LLL.")
        self.assertEqual(_lost("-----"), ".....")

    def testIouExactlyAtTheThresholdIsNotLost(self) -> None:
        iou = np.full(6, LOST_IOU_THRESHOLD)
        self.assertFalse(lostFrameMask(iou, np.ones(6, bool)).any())

    def testDualIouMatchesAGroundTruthWrittenPastEitherBorder(self) -> None:
        prediction = np.asarray([[-30.0, 50.0, 60.0, 40.0]])
        for truth in ([[-30.0, 50.0, 60.0, 40.0]], [[WIDTH - 30.0, 50.0, 60.0, 40.0]]):
            np.testing.assert_allclose(dualIou(np.asarray(truth), prediction, WIDTH), [1.0])

    def testSequenceLossCountsFramesAndTheFirstLostFrame(self) -> None:
        rows = [HIT] * 3 + [MISS] * 6 + [HIT]
        loss = sequenceLoss(
            np.asarray([HIT] * 10), np.ones(10, bool), np.asarray(rows), WIDTH
        )

        self.assertEqual((loss.frames, loss.lostFrames, loss.firstLostFrame), (10, 6, 3))
        self.assertAlmostEqual(loss.lossRate, 0.6)

    def testScoresCarryAFrameWeightedLossRate(self) -> None:
        scores = _scores({"short": [HIT] * 10, "long": [HIT] * 10 + [MISS] * 20})

        self.assertAlmostEqual(scores.lossRate, 20 / 40)
        self.assertEqual(scores.summary()["loss_rate"], scores.lossRate)
        self.assertEqual(scores.perSequence["short"].lostFrames, 0)
        self.assertIsNone(scores.perSequence["short"].firstLostFrame)
        self.assertEqual(scores.perSequence["long"].firstLostFrame, 10)
        self.assertEqual(scores.perSequence["long"].frames, 30)

    def testSphericalScoresHaveNoLossRate(self) -> None:
        rows = np.asarray([[10.0, 5.0, 30.0, 20.0, 0.0]] * 2)
        scores = evaluateVot360({"a": (rows, np.ones(2, bool))}, {"a": rows}, "bfov")

        self.assertIsNone(scores.lossRate)
        self.assertIsNone(scores.perSequence["a"].lostFrames)


class BootstrapTest(unittest.TestCase):
    def testIntervalContainsTheMeanAndNarrowsWithLessSpread(self) -> None:
        wide = bootstrapRatio([0.0, 1.0] * 10, samples=2000)
        narrow = bootstrapRatio([0.45, 0.55] * 10, samples=2000)

        self.assertAlmostEqual(wide.value, 0.5)
        self.assertTrue(wide.low < 0.5 < wide.high)
        self.assertLess(narrow.high - narrow.low, wide.high - wide.low)

    def testSameSeedGivesTheSameInterval(self) -> None:
        values = np.linspace(0.0, 1.0, 25)
        self.assertEqual(bootstrapRatio(values, seed=3), bootstrapRatio(values, seed=3))
        self.assertNotEqual(bootstrapRatio(values, seed=3), bootstrapRatio(values, seed=4))

    def testRatioOfSumsWeighsSequencesByTheirDenominator(self) -> None:
        interval = bootstrapRatio([0.0, 90.0], [10.0, 90.0], samples=500)
        self.assertAlmostEqual(interval.value, 0.9)

    def testPairedDifferenceSeesAConsistentSmallGain(self) -> None:
        base = np.linspace(0.0, 1.0, 25)
        consistent = bootstrapDifference(base, base + 0.02, samples=2000)
        # The same average gain, but scattered: half the sequences gain, half lose.
        scattered = bootstrapDifference(
            base, base + 0.02 + np.resize([0.4, -0.4], 25) - 0.016, samples=2000
        )

        self.assertAlmostEqual(consistent.value, 0.02)
        self.assertTrue(consistent.excludesZero)
        self.assertAlmostEqual(scattered.value, 0.02, places=6)
        self.assertFalse(scattered.excludesZero)

    def testInvalidInputsAreRejected(self) -> None:
        for call in (
            lambda: bootstrapRatio([]),
            lambda: bootstrapRatio([1.0], [1.0, 2.0]),
            lambda: bootstrapRatio([1.0], samples=0),
            lambda: bootstrapRatio([1.0], confidence=1.0),
            lambda: bootstrapDifference([1.0], [1.0, 2.0]),
        ):
            with self.assertRaises(ValueError):
                call()


class ComparisonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = _scores(
            {"a": [HIT] * 20, "b": [HIT] * 20, "c": [HIT] * 10 + [MISS] * 10, "only": [HIT] * 5}
        )
        self.candidate = _scores(
            {"a": [HIT] * 20, "b": [HIT] * 5 + [MISS] * 15, "c": [HIT] * 20, "other": [HIT] * 5}
        )
        self.comparison = compareScores(self.baseline, self.candidate, samples=500)

    def testComparesOnlyTheCommonSequences(self) -> None:
        self.assertEqual(self.comparison.sequences, ("a", "b", "c"))

    def testReportsScoresDifferencesAndLossRate(self) -> None:
        success = self.comparison.metric("S_dual")
        loss = self.comparison.metric("loss_rate")

        self.assertAlmostEqual(
            success.difference.value, success.candidate.value - success.baseline.value
        )
        self.assertAlmostEqual(loss.baseline.value, 10 / 60)
        self.assertAlmostEqual(loss.candidate.value, 15 / 60)
        self.assertTrue(success.difference.low <= success.difference.value)
        self.assertTrue(success.difference.value <= success.difference.high)

    def testChangesAreSortedLargestDropFirst(self) -> None:
        self.assertEqual([change.sequence for change in self.comparison.changes], ["b", "a", "c"])
        self.assertLess(self.comparison.changes[0].difference, 0.0)

    def testHardRegressionsFlagDropsBeyondTheTolerance(self) -> None:
        self.assertEqual(
            [change.sequence for change in hardRegressions(self.comparison, ["a", "b", "c"])],
            ["b"],
        )
        self.assertEqual(hardRegressions(self.comparison, ["a", "c"]), ())
        self.assertEqual(hardRegressions(self.comparison, ["b"], tolerance=0.9), ())

    def testAHardSequenceWithoutResultsIsAnError(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "only"):
            hardRegressions(self.comparison, ["a", "only"])

    def testSphericalScoresCannotBeCompared(self) -> None:
        rows = np.asarray([[10.0, 5.0, 30.0, 20.0, 0.0]] * 2)
        spherical = evaluateVot360({"a": (rows, np.ones(2, bool))}, {"a": rows}, "bfov")
        with self.assertRaisesRegex(ProtocolError, "BBox"):
            compareScores(spherical, spherical)


class HardRegressionListTest(unittest.TestCase):
    def testHardSequencesAreASubsetOfTheTuneSet(self) -> None:
        tune = readSequenceFile(ROOT / "configs" / "splits" / "360vos_tune.txt")
        hard = readSequenceFile(ROOT / "configs" / "splits" / "360vos_tune_hard.txt")

        self.assertEqual(len(hard), 5)
        self.assertEqual(len(set(hard)), len(hard))
        self.assertLessEqual(set(hard), set(tune))


if __name__ == "__main__":
    unittest.main()
