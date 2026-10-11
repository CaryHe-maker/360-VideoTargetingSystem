import unittest
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from track360.backends import ARTrackBackend, ARTrackPrediction, TrackerBackendImpl
from track360.backends.artrack_model import ARTrackTemplate
from track360.backends.artrack_seq_session import padTrajectory, trajectoryTokens
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    LocalView,
    SphericalPoint,
    ViewSpec,
)

ROOT = Path(__file__).resolve().parents[2]
BINS = 400


class TrajectoryTokensTest(unittest.TestCase):
    def testTheBoxTheCropIsCenteredOnMapsToTheMiddleOfTheCrop(self) -> None:
        # A 4x crop of a 64x64 target is the 256 px search image itself: factor 1.
        prior = BBoxXYWH(96.0, 96.0, 64.0, 64.0)

        x0, y0, x1, y1 = trajectoryTokens([prior], prior, 1.0, BINS)

        # Crop coordinates 0.373..0.623 of the side, shifted by 0.5 and spread over bins - 1.
        self.assertAlmostEqual(x0, ((127.5 - 32.0) / 256.0 + 0.5) * (BINS - 1))
        self.assertAlmostEqual(x1, ((127.5 + 32.0) / 256.0 + 0.5) * (BINS - 1))
        self.assertAlmostEqual((x0, x1), (y0, y1))
        self.assertAlmostEqual(x1 - x0, 64.0 / 256.0 * (BINS - 1))

    def testOffsetsAndTheResizeFactorScaleWithTheCrop(self) -> None:
        prior = BBoxXYWH(100.0, 100.0, 100.0, 100.0)
        shifted = BBoxXYWH(140.0, 100.0, 100.0, 100.0)

        base = trajectoryTokens([prior], prior, 0.5, BINS)
        moved = trajectoryTokens([shifted], prior, 0.5, BINS)

        # 40 px in the image is 20 px in a crop resized by 0.5.
        self.assertAlmostEqual(moved[0] - base[0], 20.0 / 256.0 * (BINS - 1))
        self.assertAlmostEqual(moved[1], base[1])
        self.assertAlmostEqual(base[2] - base[0], 50.0 / 256.0 * (BINS - 1))

    def testCoordinatesFarOutsideTheCropClampToTheBinRange(self) -> None:
        prior = BBoxXYWH(96.0, 96.0, 64.0, 64.0)
        farLeft = BBoxXYWH(-5000.0, 96.0, 64.0, 64.0)
        farRight = BBoxXYWH(5000.0, 96.0, 64.0, 64.0)

        left = trajectoryTokens([farLeft], prior, 1.0, BINS)
        right = trajectoryTokens([farRight], prior, 1.0, BINS)

        self.assertEqual((left[0], left[2]), (0.0, 0.0))
        self.assertEqual((right[0], right[2]), (2.0 * (BINS - 1), 2.0 * (BINS - 1)))

    def testTokensListBoxesOldestFirstFourValuesEach(self) -> None:
        prior = BBoxXYWH(96.0, 96.0, 64.0, 64.0)
        boxes = [BBoxXYWH(96.0 + step, 96.0, 64.0, 64.0) for step in range(7)]

        tokens = trajectoryTokens(boxes, prior, 1.0, BINS)

        self.assertEqual(len(tokens), 28)
        self.assertEqual(tokens[0::4], sorted(tokens[0::4]))

    def testAShortHistoryIsPaddedWithItsOldestBoxAndALongOneKeepsTheNewest(self) -> None:
        boxes = [BBoxXYWH(float(index), 0.0, 10.0, 10.0) for index in range(3)]
        long = [BBoxXYWH(float(index), 0.0, 10.0, 10.0) for index in range(10)]

        self.assertEqual([box.xPx for box in padTrajectory(boxes, 7)], [0, 0, 0, 0, 0, 1, 2])
        self.assertEqual([box.xPx for box in padTrajectory(long, 7)], [3, 4, 5, 6, 7, 8, 9])


class _FakeSequenceSession:
    """A sequence-level session stand-in that records what the facade hands it."""

    trajectoryLength = 7

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def encodeTemplate(self, rgb, bbox):
        return ARTrackTemplate(rgb, bbox)

    def infer(self, rgb, templateFeatures):
        raise AssertionError("views planned with a prior must go through inferBatch")

    def inferBatch(
        self,
        rgbs,
        templateFeatures,
        *,
        imageFovs=None,
        priorBoxes=None,
        trajectories: Sequence[Sequence[BBoxXYWH]] | None = None,
    ):
        self.calls.append({"priors": priorBoxes, "trajectories": trajectories})
        return tuple(ARTrackPrediction(BBoxXYWH(8.0, 9.0, 20.0, 18.0), 0.8, 0.8, 0.8) for _ in rgbs)

    def close(self):
        return None


class SequenceBackendTest(unittest.TestCase):
    def testFacadeHandsTheViewsPriorAndTrajectoryToASequenceSession(self) -> None:
        point = SphericalPoint(1.0, 0.0, 0.0, 0.0, 0.0)
        prior = BBoxXYWH(24.0, 24.0, 16.0, 16.0)
        trajectory = tuple(BBoxXYWH(20.0 + step, 24.0, 16.0, 16.0) for step in range(7))
        spec = ViewSpec(0, BFoV(point, 1.0, 1.0), 64, 64, priorBox=prior, trajectory=trajectory)
        view = LocalView(spec, np.zeros((64, 64, 3), dtype=np.uint8))
        session = _FakeSequenceSession()
        backend = TrackerBackendImpl(ARTrackBackend(session))
        backend.initialize(view, BBoxXYWH(20.0, 20.0, 16.0, 16.0))

        backend.infer((view,))

        self.assertEqual(session.calls, [{"priors": (prior,), "trajectories": (trajectory,)}])
        backend.close()


if __name__ == "__main__":
    unittest.main()
