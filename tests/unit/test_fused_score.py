import unittest
from contextlib import redirect_stderr
from io import StringIO
from math import pi
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from track360.controller import (
    BetaCalibration,
    MotionScore,
    ScoreCalibration,
    calibrateBackendFusedScore,
    calibrateLocalAppearanceProbabilities,
    composeSingleScore,
    scoreMotionConsistency,
    scoreViewCenterMotion,
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
from track360.runtime.driver import _projectObservation, _projectValidObservations

TEST_CALIBRATION = ScoreCalibration(
    format="track360.score-calibration.v1",
    checkpointSha256="1" * 64,
    manifestSha256="2" * 64,
    split="calibration",
    appearanceInput="presence_quality_product",
    appearance=BetaCalibration(alpha=1.5, beta=0.75, intercept=-0.2),
    appearanceWeight=0.8,
    motionWeight=0.2,
    candidateMinScore=0.35,
    fusionSourceMinConfidence=0.65,
)


class FusedScoreRemappingTest(unittest.TestCase):
    def testArtifactBetaCalibrationIsMonotonicAndKeepsEndpoints(self) -> None:
        rawScores = (0.0, 0.1, 0.4, 0.8, 0.95, 1.0)
        calibrated = tuple(
            calibrateBackendFusedScore(score, TEST_CALIBRATION) for score in rawScores
        )

        self.assertEqual(calibrated, tuple(sorted(calibrated)))
        self.assertEqual(calibrated[0], 0.0)
        self.assertEqual(calibrated[-1], 1.0)

    def testCreatesNewObservationsWithoutOverwritingBackendScore(self) -> None:
        original = _observation(0.85)

        (remapped,) = calibrateLocalAppearanceProbabilities(
            (original,), TEST_CALIBRATION
        )

        self.assertIsNot(remapped, original)
        self.assertEqual(original.fusedScore, 0.85)
        self.assertEqual(remapped.fusedScore, 0.85)
        self.assertAlmostEqual(
            remapped.appearanceProbability or 0.0,
            calibrateBackendFusedScore(0.85, TEST_CALIBRATION),
        )
        self.assertEqual(remapped.bbox, original.bbox)
        self.assertEqual(remapped.appearanceScore, original.appearanceScore)

    def testMotionScoreUsesPredictionCovarianceAndReliability(self) -> None:
        prediction = MotionState3D(
            position=(0.0, 0.0, 1.0),
            velocity=(0.0, 0.0, 0.0),
            confidence=0.9,
            horizontalSizeRad=0.4,
            verticalSizeRad=0.3,
            angularUncertaintyRad=0.03,
            scaleUncertainty=0.1,
            reliability=1.0,
            centerCovarianceRad2=((0.0009, 0.0), (0.0, 0.0009)),
            scaleCovarianceLog2=((0.01, 0.0), (0.0, 0.01)),
        )
        aligned = scoreMotionConsistency(
            BFoV(makeSphericalPoint(0.0, 0.0), 0.4, 0.3), prediction
        )
        displaced = scoreMotionConsistency(
            BFoV(makeSphericalPoint(0.35, 0.0), 0.4, 0.3), prediction
        )

        self.assertGreater(aligned.rawScore, displaced.rawScore)
        self.assertGreater(aligned.effectiveProbability, displaced.effectiveProbability)

        unreliable = scoreMotionConsistency(
            BFoV(makeSphericalPoint(0.35, 0.0), 0.4, 0.3),
            MotionState3D(
                position=prediction.position,
                velocity=prediction.velocity,
                confidence=0.9,
            ),
        )
        self.assertAlmostEqual(unreliable.effectiveProbability, 0.5)

    def testSingleScoreUsesArtifactWeights(self) -> None:
        self.assertAlmostEqual(composeSingleScore(0.8, 0.2, TEST_CALIBRATION), 0.68)

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
                scoreCalibration=TEST_CALIBRATION,
            )

        score.assert_called_once_with(view.spec.bfov.center, predicted)
        self.assertAlmostEqual(projected.motionScore, 0.8)

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
            projected = _projectValidObservations(
                frame=frame,
                views=(view,),
                observations=(observation,),
                predictedMotion=None,
                geometry=geometry,
                scoreCalibration=TEST_CALIBRATION,
            )

        self.assertEqual(projected, ())
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
