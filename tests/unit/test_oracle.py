import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from track360.core.types import BBoxXYWH, BFoV, ProjectedObservation
from track360.evaluation.oracle import OracleSettings, SequenceOracle, _angle, _offset
from track360.geometry import SphericalGeometryImpl, makeSphericalPoint

WIDTH, HEIGHT = 360, 180
TARGET = BFoV(makeSphericalPoint(0.0, 0.0), 0.2, 0.2)
# The target's ERP box: 20 x 20 pixels at the image centre (1 pixel per degree).
TRUTH_BOX = np.asarray([180.0 - 5.73, 90.0 - 5.73, 11.46, 11.46])


def _candidate(yaw: float, score: float = 0.5) -> ProjectedObservation:
    return ProjectedObservation(
        viewId=0,
        bfov=BFoV(makeSphericalPoint(yaw, 0.0), 0.2, 0.2),
        bbox=BBoxXYWH(180.0 + math.degrees(yaw) - 5.73, 84.27, 11.46, 11.46),
        modelScore=score,
        appearanceScore=score,
        motionScore=score,
        scaleScore=1.0,
        fusedScore=score,
        singleScore=score,
    )


class OracleTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.data = Path(self._dir.name) / "d.npz"
        np.savez(
            self.data,
            reach=np.linspace(0.0, 1.0, 11),
            score=np.full(5, 0.6),
            similarity=np.full(5, 0.4),
        )

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _oracle(self, shares: str, frames: int = 20) -> SequenceOracle:
        return SequenceOracle(
            settings=OracleSettings.parse(shares, self.data, seed=1),
            sequence="s",
            truth=[TARGET] * frames,
            truthBox=np.tile(TRUTH_BOX, (frames, 1)),
            present=np.ones(frames, dtype=bool),
            frameWidthPx=WIDTH,
            frameHeightPx=HEIGHT,
        )

    def testSharesAreParsedAndValidated(self) -> None:
        settings = OracleSettings.parse("a=0.5,E=1", self.data)
        self.assertEqual(dict(settings.shares), {"A": 0.5, "E": 1.0})
        with self.assertRaises(ValueError):
            OracleSettings.parse("F=0.5", self.data)
        with self.assertRaises(ValueError):
            OracleSettings.parse("A=1.5", self.data)

    def testAStretchCaughtIsDoubtedFromItsThirdFrameUntilTheBoxIsBack(self) -> None:
        oracle = self._oracle("A=1")
        miss = BBoxXYWH(10.0, 10.0, 10.0, 10.0)
        hit = BBoxXYWH(*TRUTH_BOX)
        self.assertGreater(oracle.boxIou(0, hit), 0.9)
        doubts = [oracle.forceDoubt(i, miss) for i in range(5)]
        self.assertEqual(doubts, [False, False, True, True, True])
        self.assertFalse(oracle.forceDoubt(5, hit))
        self.assertFalse(oracle.forceDoubt(6, miss))
        self.assertEqual(oracle.counts["A"], [2, 2])
        self.assertFalse(any(self._oracle("A=0").forceDoubt(i, miss) for i in range(5)))

    def testDrawsAreReproducibleAndFollowTheShare(self) -> None:
        first, second = self._oracle("E=0.3", 2000), self._oracle("E=0.3", 2000)
        wrong = _candidate(1.0)
        a = [first.vetoJump(i, wrong, None) for i in range(2000)]
        b = [second.vetoJump(i, wrong, None) for i in range(2000)]
        self.assertEqual(a, b)
        self.assertAlmostEqual(np.mean(a), 0.3, delta=0.04)

    def testOnlyWrongJumpsAreVetoed(self) -> None:
        oracle = self._oracle("E=1")
        self.assertFalse(oracle.vetoJump(0, _candidate(0.0), None))
        self.assertTrue(oracle.vetoJump(1, _candidate(1.0), None))
        # A jump on the target while the tracker is on it too is needless.
        self.assertTrue(oracle.vetoJump(2, _candidate(0.0), BBoxXYWH(*TRUTH_BOX)))

    def testAMissedCandidateOnTheTargetIsTaken(self) -> None:
        oracle = self._oracle("D=1")
        good, bad = _candidate(0.02), _candidate(1.0)
        self.assertIs(oracle.takeMissed(0, [bad, good]), good)
        self.assertIsNone(oracle.takeMissed(1, [bad]))

    def testASearchThatMissesIsMovedNearTheTarget(self) -> None:
        oracle = self._oracle("B=1")
        far = makeSphericalPoint(1.0, 0.3)
        moved = oracle.scanCentre(0, far, 0.3)
        self.assertLessEqual(_angle(moved, TARGET.center), 0.3 + 1e-9)
        near = makeSphericalPoint(0.1, 0.0)
        self.assertIs(oracle.scanCentre(1, near, 0.3), near)

    def testABoxIsAddedWhenTheTargetIsInViewAndNoCandidateIsOnIt(self) -> None:
        oracle = self._oracle("C=1")
        geometry = SphericalGeometryImpl(boundarySamplesPerEdge=9)
        extra = oracle.extraCandidate(0, TARGET.center, 0.5, (_candidate(1.0),), geometry)
        assert extra is not None
        self.assertTrue(oracle.onTarget(0, extra.bfov))
        self.assertAlmostEqual(extra.singleScore, 0.6)
        self.assertAlmostEqual(extra.appearanceSimilarity, 0.4)
        self.assertIsNone(
            oracle.extraCandidate(1, TARGET.center, 0.5, (_candidate(0.0),), geometry)
        )
        far = makeSphericalPoint(2.0, 0.0)
        self.assertIsNone(oracle.extraCandidate(2, far, 0.5, (), geometry))

    def testOffsetMovesByTheAngle(self) -> None:
        centre = makeSphericalPoint(0.4, 0.6)
        for bearing in (0.0, 1.0, 3.0):
            self.assertAlmostEqual(_angle(centre, _offset(centre, 0.25, bearing)), 0.25)


if __name__ == "__main__":
    unittest.main()
