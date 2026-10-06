import tempfile
import unittest
from pathlib import Path

import numpy as np

from track360.core.errors import ProtocolError
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    SequenceId,
    TrackResult,
    TrackStatus,
)
from track360.evaluation.score_analysis import (
    auroc,
    bestThreshold,
    buildFrameTable,
    onsetProfile,
    spearman,
    summarize,
    thresholdTable,
)
from track360.geometry import makeSphericalPoint
from track360.io.vot360_results import readScoreFile, scorePath, writeSequenceResults

WIDTH = 3840
HIT = [100.0, 50.0, 60.0, 40.0]
MISS = [2000.0, 900.0, 60.0, 40.0]


def _table(rows: list[list[float]], scores: list[float], present: list[bool] | None = None):
    count = len(rows)
    mask = np.ones(count, bool) if present is None else np.asarray(present)
    return buildFrameTable(
        {"a": (np.asarray([HIT] * count), mask)},
        {"a": np.asarray(rows)},
        {"a": np.asarray(scores)},
        WIDTH,
    )


class ScoreAnalysisTest(unittest.TestCase):
    def testAurocIsOneHalfForNoSignalAndOneForPerfectSeparation(self) -> None:
        positive = np.asarray([True, True, False, False])
        self.assertEqual(auroc(np.asarray([0.9, 0.8, 0.2, 0.1]), positive), 1.0)
        self.assertEqual(auroc(np.asarray([0.1, 0.2, 0.8, 0.9]), positive), 0.0)
        self.assertEqual(auroc(np.asarray([0.5, 0.5, 0.5, 0.5]), positive), 0.5)
        self.assertIsNone(auroc(np.asarray([0.5, 0.6]), np.asarray([True, True])))

    def testSpearmanFollowsRankOrderAndNeedsVariation(self) -> None:
        values = np.asarray([0.1, 0.2, 0.3, 0.9])
        self.assertAlmostEqual(spearman(values, values**3), 1.0)
        self.assertAlmostEqual(spearman(values, -values), -1.0)
        self.assertIsNone(spearman(values, np.ones(4)))

    def testFrameTableSkipsTheFirstFrameAndAbsentFrames(self) -> None:
        table = _table(
            [HIT, HIT, MISS, HIT],
            [1.0, 0.6, 0.4, 0.7],
            present=[True, True, False, True],
        )

        self.assertEqual(table.frame.tolist(), [1, 3])
        self.assertEqual(table.score.tolist(), [0.6, 0.7])
        self.assertTrue(table.good.all())

    def testFrameTableMarksLossRunsAndRejectsMismatchedLengths(self) -> None:
        table = _table([HIT] * 3 + [MISS] * 6, [1.0] + [0.6] * 2 + [0.4] * 6)

        self.assertEqual(table.lost.tolist(), [False, False] + [True] * 6)
        with self.assertRaisesRegex(ProtocolError, "differ in length"):
            _table([HIT] * 3, [0.5] * 2)
        with self.assertRaisesRegex(ProtocolError, "no sequence"):
            buildFrameTable({}, {}, {}, WIDTH)

    def testThresholdTableCountsCaughtLossesAndFlaggedGoodFrames(self) -> None:
        table = _table(
            [HIT] * 5 + [MISS] * 6,
            [1.0, 0.6, 0.6, 0.45, 0.6] + [0.4] * 5 + [0.55],
        )

        row = thresholdTable(table, [0.5])[0]

        self.assertAlmostEqual(row.lostFlagged, 5 / 6)
        self.assertAlmostEqual(row.goodFlagged, 1 / 4)
        self.assertAlmostEqual(row.flaggedLost, 5 / 6)
        self.assertAlmostEqual(row.flagged, 6 / 10)

    def testBestThresholdSeparatesCleanGroups(self) -> None:
        table = _table([HIT] * 5 + [MISS] * 6, [1.0] + [0.6] * 4 + [0.4] * 6)

        best = bestThreshold(table)

        assert best is not None
        self.assertTrue(0.4 < best.threshold <= 0.6)
        self.assertEqual((best.lostFlagged, best.goodFlagged), (1.0, 0.0))
        self.assertIsNone(bestThreshold(_table([HIT] * 4, [1.0, 0.6, 0.6, 0.6])))

    def testOnsetProfileReadsScoresAroundTheFirstLostFrame(self) -> None:
        table = _table([HIT] * 4 + [MISS] * 6, [1.0, 0.7, 0.6, 0.5, 0.4, 0.3] + [0.2] * 4)

        profile = onsetProfile(table)

        self.assertEqual(profile[0], {"runs": 1, "median": 0.4})
        self.assertEqual(profile[-1]["median"], 0.5)
        self.assertEqual(profile[1]["median"], 0.3)
        self.assertNotIn(-20, profile)

    def testSummaryReportsSeparationWhenTheScoreTracksIou(self) -> None:
        informative = summarize(_table([HIT] * 6 + [MISS] * 6, [1.0] + [0.6] * 5 + [0.4] * 6))
        blind = summarize(_table([HIT] * 6 + [MISS] * 6, [1.0] + [0.5] * 11))

        self.assertEqual(informative["frames"], 11)
        self.assertEqual(informative["auroc"]["lostVsGood"], 1.0)
        self.assertGreater(informative["correlation"]["spearman"], 0.8)
        self.assertEqual(informative["scorePercentiles"]["lost"]["p50"], 0.4)
        self.assertEqual(informative["scorePercentiles"]["good"]["p50"], 0.6)
        self.assertEqual(blind["auroc"]["lostVsGood"], 0.5)
        self.assertIsNone(blind["correlation"]["spearman"])
        self.assertEqual(sum(item["frames"] for item in informative["scoreByIou"]), 11)


class ScoreFileTest(unittest.TestCase):
    def testResultsWriteOneConfidencePerFrame(self) -> None:
        bfov = BFoV(makeSphericalPoint(0.0, 0.0), 0.3, 0.2)
        results = [
            TrackResult(
                SequenceId("s"),
                FrameIndex(index),
                BBoxXYWH(10.0, 10.0, 20.0, 20.0),
                bfov,
                confidence,
                TrackStatus.TRACKING,
                True,
            )
            for index, confidence in enumerate((1.0, 0.4871234, 0.0))
        ]
        with tempfile.TemporaryDirectory() as directory:
            writeSequenceResults(directory, "ours", "0001", results, WIDTH)
            path = scorePath(directory, "ours", "0001")

            self.assertEqual(path, Path(directory) / "score" / "ours" / "0001.txt")
            np.testing.assert_allclose(readScoreFile(path), [1.0, 0.487123, 0.0])


if __name__ == "__main__":
    unittest.main()
