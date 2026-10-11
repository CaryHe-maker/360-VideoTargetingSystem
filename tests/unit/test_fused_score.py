import unittest
from contextlib import redirect_stderr
from io import StringIO
from math import pi
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from track360.controller import (
    MotionScore,
    backendScoreProbability,
    scoreViewCenterMotion,
    withScoreProbability,
)
from track360.core.errors import GeometryError
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    FramePacket,
    LocalObservation,
    LocalView,
    MotionState3D,
    SequenceId,
    ViewSpec,
)
from track360.geometry import makeSphericalPoint
from track360.runtime.driver import _projectObservation, _projectValidObservation


class FusedScoreRemappingTest(unittest.TestCase):
    def testTheScoreProbabilityIsTheScoreUpToRounding(self) -> None:
        rawScores = (0.0, 0.1, 0.4, 0.8, 0.95, 1.0)
        mapped = tuple(backendScoreProbability(score) for score in rawScores)

        self.assertEqual(mapped, tuple(sorted(mapped)))
        self.assertEqual(mapped[0], 0.0)
        self.assertEqual(mapped[-1], 1.0)
        for score, value in zip(rawScores, mapped, strict=True):
            self.assertAlmostEqual(value, score, places=12)
        # The round trip is not exact, which is why it is kept: the recorded
        # baselines were made with these very numbers.
        self.assertEqual(backendScoreProbability(0.9), 0.8999999999999999)
        self.assertEqual(backendScoreProbability(0.7), 0.7)

    def testCreatesNewObservationsWithoutOverwritingBackendScore(self) -> None:
        original = _observation(0.85)

        (remapped,) = withScoreProbability((original,))

        self.assertIsNot(remapped, original)
        self.assertEqual(original.fusedScore, 0.85)
        self.assertEqual(remapped.fusedScore, 0.85)
        self.assertAlmostEqual(remapped.appearanceProbability or 0.0, 0.85)
        self.assertEqual(remapped.bbox, original.bbox)
        self.assertEqual(remapped.appearanceScore, original.appearanceScore)

    def testViewCenterMotionFallsContinuouslyByPointOnePerThirtyDegrees(self) -> None:
        prediction = MotionState3D(
            position=(0.0, 0.0, 1.0),
            velocity=(0.0, 0.0, 0.0),
            confidence=1.0,
            reliability=0.25,
        )
        expected = {
            0.0: 1.0,
            30.0: 0.9,
            45.0: 0.85,
            60.0: 0.8,
            90.0: 0.7,
            180.0: 0.4,
        }

        for angleDeg, expectedScore in expected.items():
            with self.subTest(angleDeg=angleDeg):
                score = scoreViewCenterMotion(
                    makeSphericalPoint(angleDeg * pi / 180.0, 0.0),
                    prediction,
                )
                self.assertAlmostEqual(score.effectiveProbability, expectedScore)

    def testViewCenterMotionUsesLocalViewCenterNotDetectionCenter(self) -> None:
        prediction = MotionState3D(
            position=(0.0, 0.0, 1.0),
            velocity=(0.0, 0.0, 0.0),
            confidence=1.0,
        )

        centered = scoreViewCenterMotion(makeSphericalPoint(0.0, 0.0), prediction)
        sideView = scoreViewCenterMotion(makeSphericalPoint(pi / 3.0, 0.0), prediction)

        self.assertAlmostEqual(centered.effectiveProbability, 1.0)
        self.assertAlmostEqual(sideView.effectiveProbability, 0.8)

    def testProjectionPathScoresTheLocalViewCenter(self) -> None:
        predicted = MotionState3D(
            position=(0.0, 0.0, 1.0),
            velocity=(0.0, 0.0, 0.0),
            confidence=1.0,
        )
        viewCenter = makeSphericalPoint(pi / 3.0, 0.0)
        view = LocalView(
            ViewSpec(3, BFoV(viewCenter, 1.0, 1.0), 256, 256),
            np.zeros((256, 256, 3), dtype=np.uint8),
        )
        observation = _observation(0.90)
        frame = FramePacket(
            SequenceId("view-center-score"),
            FrameIndex(1),
            1,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        geometry = Mock()
        geometry.projectLocalBoxBoundary.return_value = SimpleNamespace(
            bfov=BFoV(makeSphericalPoint(-1.0, 0.2), 0.2, 0.2),
            bbox=BBoxXYWH(20.0, 20.0, 30.0, 30.0),
            erpBoundary=(),
            envelopeInflation=1.0,
        )
        viewScore = MotionScore(0.8, 0.8, 0.8, 0.5, (pi / 3.0) ** 2)

        target = "track360.runtime.driver.scoreViewCenterMotion"
        with patch(target, return_value=viewScore) as score:
            projected = _projectObservation(
                frame=frame,
                view=view,
                observation=observation,
                predictedMotion=predicted,
                geometry=geometry,
            )

        score.assert_called_once_with(view.spec.bfov.center, predicted)
        self.assertAlmostEqual(projected.motionScore, 0.8)
        # The motion prior is recorded; the score the controller uses is the backend's.
        self.assertAlmostEqual(projected.singleScore, 0.90)

    def testInvalidSphericalProjectionIsSkipped(self) -> None:
        view = LocalView(
            ViewSpec(7, BFoV(makeSphericalPoint(0.0, 0.0), 1.0, 1.0), 256, 256),
            np.zeros((256, 256, 3), dtype=np.uint8),
        )
        observation = _observation(0.90)
        frame = FramePacket(
            SequenceId("projection-skip"),
            FrameIndex(18),
            18,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        geometry = Mock()
        geometry.projectLocalBoxBoundary.side_effect = GeometryError(
            "horizontal BFoV span is invalid"
        )
        stderr = StringIO()

        with redirect_stderr(stderr):
            projected = _projectValidObservation(
                frame=frame,
                view=view,
                observation=observation,
                predictedMotion=None,
                geometry=geometry,
            )

        self.assertIsNone(projected)
        self.assertIn("sequence=projection-skip, frame=18, view=7", stderr.getvalue())


def _observation(score: float) -> LocalObservation:
    return LocalObservation(
        viewId=1,
        bbox=BBoxXYWH(10.0, 20.0, 30.0, 40.0),
        modelScore=0.88,
        appearanceScore=0.87,
        fusedScore=score,
        latencyNs=1,
    )


if __name__ == "__main__":
    unittest.main()
