import tempfile
import unittest
from pathlib import Path

import numpy as np

from track360.core.errors import ModelError
from track360.runtime.candidate_judge import (
    SCALAR_COUNT,
    TOKEN_SIZE,
    CandidateJudge,
    judgeFeatures,
    writeJudges,
)

WIDTH = SCALAR_COUNT + TOKEN_SIZE


def _judge(sequences: list[str], qIntercept: float, split: float) -> dict[str, np.ndarray]:
    """Q: 0.5 * first scalar + qIntercept; P1: one tree splitting the first scalar."""
    qCoef = np.zeros(WIDTH)
    qCoef[0] = 0.5
    return {
        "sequences": np.asarray(sequences, dtype=str),
        "qLow": np.full(WIDTH, -10.0),
        "qHigh": np.full(WIDTH, 10.0),
        "qMean": np.zeros(WIDTH),
        "qScale": np.ones(WIDTH),
        "qCoef": qCoef,
        "qIntercept": np.asarray(qIntercept),
        "pTokenMean": np.zeros(TOKEN_SIZE),
        "pTokenScale": np.ones(TOKEN_SIZE),
        "pPcaMean": np.zeros(TOKEN_SIZE),
        "pPcaComponents": np.zeros((2, TOKEN_SIZE)),
        # Root 0 splits feature 0 at ``split``: left leaf -10 (p ~ 0), right leaf +10.
        "treeBaseline": np.asarray(0.0),
        "treeFeature": np.asarray([0, 0, 0]),
        "treeThreshold": np.asarray([split, 0.0, 0.0]),
        "treeLeft": np.asarray([1, 0, 0]),
        "treeRight": np.asarray([2, 0, 0]),
        "treeLeaf": np.asarray([0, 1, 1], dtype=np.uint8),
        "treeValue": np.asarray([0.0, -10.0, 10.0]),
        "treeRoots": np.asarray([0]),
    }


class CandidateJudgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / "judge.npz"
        writeJudges(
            self.path,
            {"fold0": _judge(["a", "b"], 0.4, 0.5), "all": _judge([], 0.0, 0.5)},
        )

    def tearDown(self) -> None:
        self._dir.cleanup()

    def testTheJudgeTrainedWithoutTheSequenceIsUsedAndScoresArePTimesQ(self) -> None:
        judge = CandidateJudge(self.path, threshold=0.3, prefilter=0.2)
        x = np.zeros((2, WIDTH))
        x[0, 0], x[1, 0] = 0.2, 0.8
        self.assertEqual(judge.select("a"), "fold0")
        scores = judge.score(x)
        self.assertAlmostEqual(scores[0], 0.0, places=4)  # P1 ~ 0 below the split
        self.assertAlmostEqual(scores[1], 0.8, places=4)  # P1 ~ 1, Q = 0.4 + 0.4
        self.assertEqual(judge.select("other"), "all")
        self.assertAlmostEqual(judge.score(x)[1], 0.4, places=4)
        self.assertEqual(judge.describe()["judges"], ["all", "fold0"])
        with self.assertRaises(ModelError):
            judge.score(np.zeros((1, 3)))

    def testAFileWithoutTheJudgeForAllIsRejected(self) -> None:
        writeJudges(self.path, {"fold0": _judge(["a"], 0.0, 0.5)})
        with self.assertRaises(ModelError):
            CandidateJudge(self.path, threshold=0.3)

    def testFeaturesFollowTheTrainingLayoutAndRounding(self) -> None:
        token = np.linspace(0.0, 1.0, TOKEN_SIZE, dtype=np.float32) + 1e-4
        x = judgeFeatures(
            searchScore=0.712345,
            recentredScore=0.5,
            similarity=None,
            scan="4x",
            diagnostics={
                "coordEntropy": [1.0, 2.0, 3.0, 6.0],
                "coordPeak": [0.4, 0.2, 0.9, 0.5],
                "hardSoftGap": [0.1, 0.3, 0.2, 0.0],
                "scoreFeature": token,
            },
        )
        self.assertEqual(x.shape, (WIDTH,))
        np.testing.assert_allclose(
            x[:SCALAR_COUNT], [0.7123, 0.5, 0.0, 0.0, 1.0, 3.0, 6.0, 0.2, 0.3], rtol=1e-6
        )
        np.testing.assert_array_equal(x[SCALAR_COUNT:], token.astype(np.float16))


if __name__ == "__main__":
    unittest.main()
