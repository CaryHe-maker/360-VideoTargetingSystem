import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from track360.controller import (
    SphericalMotionEstimator,
    TrackControllerImpl,
    TrackStateMachine,
    ViewPlanner,
)
from track360.controller.state_evaluator import fuseStateScore, motionAgreement
from track360.controller.state_model import TrackMode, TransitionReason
from track360.controller.view_planner import localBoxOfBfov
from track360.core.config import loadConfig
from track360.core.errors import ConfigError, ProtocolError
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    FramePacket,
    ProjectedObservation,
    ResultSource,
    SequenceId,
    TrackStatus,
    ViewProjection,
    ViewSpec,
)
from track360.geometry import SphericalGeometryImpl, makeSphericalPoint

ROOT = Path(__file__).resolve().parents[2]
INITIAL_BOX = BBoxXYWH(150.0, 70.0, 40.0, 50.0)


def _frame(index: int, *, sequence: str = "s", shape: tuple[int, int] = (180, 360)) -> FramePacket:
    return FramePacket(
        SequenceId(sequence),
        FrameIndex(index),
        index * 1_000_000_000,
        np.zeros((*shape, 3), dtype=np.uint8),
    )


def _observation(
    score: float,
    *,
    viewId: int = 0,
    xPx: float = 150.0,
    widthPx: float = 40.0,
    heightPx: float = 50.0,
) -> ProjectedObservation:
    return ProjectedObservation(
        viewId=viewId,
        bfov=BFoV(makeSphericalPoint(0.0, 0.0), 0.35, 0.25),
        bbox=BBoxXYWH(xPx, 70.0, widthPx, heightPx),
        modelScore=score,
        appearanceScore=score,
        motionScore=score,
        scaleScore=score,
        fusedScore=score,
        localBox=BBoxXYWH(96.0, 88.0, 64.0, 80.0),
    )


class MotionEstimatorTest(unittest.TestCase):
    def testWrapsSphericalDirectionAcrossTheSeam(self) -> None:
        estimator = SphericalMotionEstimator()
        estimator.initialize(makeSphericalPoint(math.pi - 0.05, 0.0), 0)
        updated = estimator.update(makeSphericalPoint(-math.pi + 0.05, 0.0), 1_000_000_000, 0.9)
        predicted = estimator.predict(2_000_000_000)

        self.assertGreater(predicted.confidence, 0.0)
        self.assertGreater(updated.confidence, 0.0)
        self.assertTrue(-math.pi <= predicted.position[0] <= math.pi)
        with self.assertRaises(ProtocolError):
            estimator.predict(500_000_000)

    def testHistoryContainsMeasurementsNotPredictions(self) -> None:
        estimator = SphericalMotionEstimator(windowLength=3)
        estimator.initialize(makeSphericalPoint(math.pi - 0.05, 0.0), 0)
        estimator.predict(500_000_000)
        self.assertEqual(len(estimator.samples), 1)
        estimator.update(makeSphericalPoint(-math.pi + 0.05, 0.0), 1_000_000_000, 0.9)
        self.assertEqual(len(estimator.samples), 2)
        prediction = estimator.predictDetailed(2_000_000_000)
        self.assertGreater(prediction.angularUncertaintyRad, 0.0)

    def testPredictionHasStableTangentBasisAtPole(self) -> None:
        estimator = SphericalMotionEstimator(windowLength=3)
        estimator.initialize(makeSphericalPoint(0.0, math.pi / 2.0 - 0.10), 0)
        estimator.update(makeSphericalPoint(0.0, math.pi / 2.0), 1_000_000_000, 0.9)
        prediction = estimator.predictDetailed(2_000_000_000)
        self.assertTrue(math.isfinite(prediction.center.yawRad))
        self.assertTrue(math.isfinite(prediction.center.pitchRad))
        self.assertLess(prediction.center.pitchRad, math.pi / 2.0)

    def testPredictionExtrapolatesScaleInLogSpace(self) -> None:
        estimator = SphericalMotionEstimator(windowLength=3)
        estimator.resetFromMeasurement(
            makeSphericalPoint(0.0, 0.0),
            0,
            0,
            1.0,
            horizontalSizeRad=0.10,
            verticalSizeRad=0.20,
        )
        estimator.recordMeasurement(
            frameIndex=1,
            timestampNs=1_000_000_000,
            point=makeSphericalPoint(0.0, 0.0),
            confidence=1.0,
            horizontalSizeRad=0.20,
            verticalSizeRad=0.40,
        )

        prediction = estimator.predictDetailed(2_000_000_000)

        self.assertAlmostEqual(prediction.horizontalSizeRad, 0.40, places=6)
        self.assertAlmostEqual(prediction.verticalSizeRad, 0.80, places=6)

    def testSingleAnchorProvidesReducedNonNeutralMotionReliability(self) -> None:
        estimator = SphericalMotionEstimator(windowLength=3, minSamplesForVelocity=2)
        estimator.resetFromMeasurement(
            makeSphericalPoint(0.0, 0.0),
            0,
            0,
            1.0,
            horizontalSizeRad=0.40,
            verticalSizeRad=0.30,
        )

        prediction = estimator.predictDetailed(1_000_000_000)

        self.assertEqual(prediction.sampleCount, 1)
        self.assertIn("insufficient_motion_samples", prediction.degradedReasons)
        self.assertGreater(prediction.reliability, 0.0)
        self.assertLess(prediction.reliability, prediction.confidence)


class StateMachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = loadConfig(ROOT / "configs" / "default.yaml")

    def _machine(self, **tuning: object) -> TrackStateMachine:
        state = TrackStateMachine(
            replace(self.config.backendTuning, uncertainScore=0.6, lostAfterFrames=3, **tuning)
        )
        state.initialize()
        return state

    def testAFrameBelowTheUncertainScoreIsNotTrustedAndOneGoodFrameRestoresTracking(self) -> None:
        state = self._machine()
        modes = []
        mode = TrackMode.TRACKING
        for score in (0.8, 0.59, 0.9, 0.6):
            mode = state.transition(mode, score, measurementAccepted=True).nextMode
            modes.append(mode)

        self.assertEqual(
            modes,
            [TrackMode.TRACKING, TrackMode.UNCERTAIN, TrackMode.TRACKING, TrackMode.TRACKING],
        )
        self.assertEqual(state.untrustedFrames, 0)

    def testEnoughUntrustedFramesInARowMeanTheTargetIsLost(self) -> None:
        state = self._machine()
        mode = TrackMode.TRACKING
        decisions = []
        for score in (0.5, 0.4, 0.0, 0.3):
            decision = state.transition(mode, score, measurementAccepted=score > 0.0)
            mode = decision.nextMode
            decisions.append(decision)

        self.assertEqual(
            [decision.nextMode for decision in decisions],
            [TrackMode.UNCERTAIN, TrackMode.UNCERTAIN, TrackMode.LOST, TrackMode.LOST],
        )
        self.assertEqual(decisions[1].reason, TransitionReason.WEAK_MEASUREMENT)
        self.assertEqual(decisions[2].reason, TransitionReason.HARD_MISS)
        self.assertFalse(decisions[2].acceptMeasurement)
        self.assertEqual(state.untrustedFrames, 4)
        state.reset()
        self.assertEqual(state.untrustedFrames, 0)
        recovered = state.transition(TrackMode.LOST, 0.7, measurementAccepted=True)
        self.assertEqual(recovered.nextMode, TrackMode.TRACKING)

    def testStateScoreIsAWeightedMeanAndLeavesOutAMissingAppearanceScore(self) -> None:
        tuning = replace(
            self.config.backendTuning,
            stateBackendWeight=0.5,
            stateAppearanceWeight=0.3,
            stateMotionWeight=0.2,
        )
        self.assertAlmostEqual(fuseStateScore(0.8, 0.4, 1.0, tuning), 0.4 + 0.12 + 0.2)
        # Without an appearance score the other two keep their proportion.
        self.assertAlmostEqual(fuseStateScore(0.8, None, 1.0, tuning), (0.4 + 0.2) / 0.7)
        with self.assertRaises(ConfigError):
            replace(tuning, stateBackendWeight=0.0, stateMotionWeight=0.0)

    def testMotionScoreFallsWithDistanceFromThePredictionAndWithSizeChange(self) -> None:
        estimator = SphericalMotionEstimator(windowLength=3, minSamplesForVelocity=2)
        estimator.resetFromMeasurement(
            makeSphericalPoint(0.0, 0.0), 0, 0, 1.0, horizontalSizeRad=0.2, verticalSizeRad=0.2
        )
        prediction = estimator.predictDetailed(1_000_000_000)

        def score(yaw: float, size: float) -> float:
            box = BFoV(makeSphericalPoint(yaw, 0.0), size, size)
            return motionAgreement(box, prediction, 1.0, 0.3)

        self.assertAlmostEqual(score(0.0, 0.2), 1.0)
        # One target size away with an offset scale of one target size: exp(-1/2).
        self.assertAlmostEqual(score(0.2, 0.2), math.exp(-0.5), places=5)
        self.assertGreater(score(0.1, 0.2), score(0.3, 0.2))
        # Twice as large: the log size ratio is log 2.
        self.assertAlmostEqual(score(0.0, 0.4), math.exp(-0.5 * (math.log(2.0) / 0.3) ** 2))


class ViewPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = loadConfig(ROOT / "configs" / "default.yaml")
        self.center = makeSphericalPoint(0.3, -0.2)

    def _planner(self, **tuning: object) -> ViewPlanner:
        return ViewPlanner(
            self.config.geometry,
            self.config.tracking,
            replace(self.config.backendTuning, **tuning),
        )

    def testAlignedViewIsSquareAndFourTimesTheMeanTargetSize(self) -> None:
        planner = self._planner()
        view = planner.searchView(self.center, math.radians(12.0), math.radians(3.0))

        # On the image plane the target spans tan(6 deg) x tan(1.5 deg) half-extents.
        meanHalfSize = math.sqrt(math.tan(math.radians(6.0)) * math.tan(math.radians(1.5)))
        self.assertEqual(view.bfov.horizontalFovRad, view.bfov.verticalFovRad)
        self.assertAlmostEqual(math.tan(view.bfov.horizontalFovRad / 2.0), 4.0 * meanHalfSize)
        self.assertEqual(view.bfov.center, self.center)
        # The prior keeps the target's aspect and makes the backend's 4x crop the view.
        prior = view.priorBox
        assert prior is not None
        self.assertAlmostEqual(4.0 * math.sqrt(prior.widthPx * prior.heightPx), 256.0, places=5)
        self.assertLessEqual(math.ceil(4.0 * math.sqrt(prior.widthPx * prior.heightPx)), 256)
        self.assertAlmostEqual(prior.xPx + prior.widthPx / 2.0, 128.0)
        self.assertGreater(prior.widthPx / prior.heightPx, 3.9)

    def testAlignedViewFollowsSmallTargetsBelowTheGeometryMinimum(self) -> None:
        planner = self._planner(sphericalSearch=False)
        small = planner.searchView(self.center, math.radians(2.0), math.radians(2.0))
        tiny = planner.searchView(self.center, math.radians(0.2), math.radians(0.2))
        large = planner.searchView(self.center, math.radians(60.0), math.radians(60.0))

        self.assertAlmostEqual(math.degrees(small.bfov.horizontalFovRad), 8.0, places=1)
        self.assertLess(small.bfov.horizontalFovRad, self.config.geometry.minFovRad)
        self.assertAlmostEqual(math.degrees(tiny.bfov.horizontalFovRad), 2.0)
        self.assertAlmostEqual(math.degrees(large.bfov.horizontalFovRad), 90.0)
        # The FOV limit cut the view short: the backend's 4x crop pads past the view.
        assert large.priorBox is not None
        self.assertGreater(4.0 * large.priorBox.widthPx, 256.0)
        # Towards 180 degrees the tangent extent diverges; the prior stays finite.
        huge = planner.searchView(self.center, math.pi, math.pi)
        bound = planner.searchView(self.center, math.radians(150.0), math.radians(150.0))
        assert huge.priorBox is not None and bound.priorBox is not None
        self.assertAlmostEqual(math.degrees(huge.bfov.horizontalFovRad), 90.0)
        self.assertEqual(huge.priorBox, bound.priorBox)
        self.assertLess(huge.priorBox.widthPx, 1000.0)

    def testSearchRegionsPastTheSwitchPointAreSampledSpherically(self) -> None:
        planner = self._planner()
        # The switch point is a search region of 120 degrees: a 30 degree target.
        below = planner.searchView(self.center, math.radians(29.0), math.radians(29.0))
        view = planner.searchView(self.center, math.radians(35.0), math.radians(35.0))

        self.assertIs(below.projection, ViewProjection.PERSPECTIVE)
        self.assertIs(view.projection, ViewProjection.SPHERICAL)
        # Four times the target on each axis, linear in angle: no FOV limit applies.
        self.assertAlmostEqual(math.degrees(view.bfov.horizontalFovRad), 140.0)
        self.assertAlmostEqual(math.degrees(view.bfov.verticalFovRad), 140.0)
        self.assertEqual((view.outputWidthPx, view.outputHeightPx), (256, 256))
        assert view.priorBox is not None
        self.assertAlmostEqual(view.priorBox.widthPx, 64.0)
        self.assertAlmostEqual(view.priorBox.xPx + view.priorBox.widthPx / 2.0, 128.0)
        self.assertIs(
            self._planner(sphericalSearch=False)
            .searchView(self.center, math.radians(35.0), math.radians(35.0))
            .projection,
            ViewProjection.PERSPECTIVE,
        )

    def testSphericalViewIsCutAtTheWholeSphereWithIsotropicPixels(self) -> None:
        planner = self._planner()
        # A 60 x 60 degree target asks for 240 degrees: the latitude span stops at 180.
        tall = planner.searchView(self.center, math.radians(60.0), math.radians(60.0))
        whole = planner.searchView(self.center, math.radians(179.0), math.radians(179.0))

        self.assertAlmostEqual(math.degrees(tall.bfov.horizontalFovRad), 240.0)
        self.assertEqual(tall.outputWidthPx, 256)
        self.assertLess(tall.outputHeightPx, 256)
        self.assertLessEqual(tall.bfov.verticalFovRad, math.pi)
        for view in (tall, whole):
            self.assertAlmostEqual(
                view.outputWidthPx / view.bfov.horizontalFovRad,
                view.outputHeightPx / view.bfov.verticalFovRad,
            )
            prior = view.priorBox
            assert prior is not None
            # The backend's 4x crop around the prior is the 256 px search image.
            self.assertAlmostEqual(4.0 * math.sqrt(prior.widthPx * prior.heightPx), 256.0)
            self.assertLessEqual(prior.widthPx, view.outputWidthPx)
            self.assertLessEqual(prior.heightPx, view.outputHeightPx)
        # The whole sphere at this scale is about 128 x 64 px.
        self.assertLess(whole.bfov.horizontalFovRad, 2.0 * math.pi)
        self.assertAlmostEqual(whole.outputWidthPx / whole.outputHeightPx, 2.0, delta=0.05)

    def testSphericalViewIsWidenedForElongatedTargets(self) -> None:
        view = self._planner().searchView(
            self.center, math.radians(170.0), math.radians(8.0)
        )

        # Four times the mean size is 147.5 degrees, less than the target is wide.
        self.assertAlmostEqual(math.degrees(view.bfov.horizontalFovRad), 212.5, delta=1.0)
        self.assertGreater(view.outputWidthPx, 256)
        assert view.priorBox is not None
        self.assertLess(view.priorBox.widthPx, view.outputWidthPx)

    def testLargeTemplatesAreSampledSphericallyToo(self) -> None:
        target = BFoV(self.center, math.radians(100.0), math.radians(60.0))
        view = self._planner().templateView(target)
        small = self._planner().templateView(BFoV(self.center, 0.2, 0.1))

        self.assertIs(view.projection, ViewProjection.SPHERICAL)
        self.assertIs(small.projection, ViewProjection.PERSPECTIVE)
        box = view.priorBox
        assert box is not None
        self.assertAlmostEqual(box.widthPx / box.heightPx, 100.0 / 60.0)
        # The view is templateFovScale times the mean target size across 256 px.
        self.assertAlmostEqual(2.5 * math.sqrt(box.widthPx * box.heightPx), 256.0)
        self.assertGreaterEqual(box.xPx, 0.0)
        self.assertLessEqual(box.yPx + box.heightPx, view.outputHeightPx)

    def testAlignedTemplateViewIsSquareSoTheTargetKeepsItsAspect(self) -> None:
        target = BFoV(self.center, math.radians(12.0), math.radians(3.0))
        aligned = self._planner().templateBfov(target)

        self.assertEqual(aligned.horizontalFovRad, aligned.verticalFovRad)
        self.assertEqual(aligned.center, target.center)


class LocalBoxOfBfovTest(unittest.TestCase):
    def setUp(self) -> None:
        self.view = ViewSpec(
            0, BFoV(makeSphericalPoint(0.3, -0.2), math.radians(40.0), math.radians(40.0)), 256, 256
        )

    def testABfovAtTheViewCenterIsCenteredAndSizedOnTheImagePlane(self) -> None:
        box = localBoxOfBfov(
            self.view, BFoV(self.view.bfov.center, math.radians(10.0), math.radians(5.0))
        )

        self.assertAlmostEqual(box.xPx + box.widthPx / 2.0, 128.0)
        self.assertAlmostEqual(box.yPx + box.heightPx / 2.0, 128.0)
        self.assertAlmostEqual(
            box.widthPx, 256.0 * math.tan(math.radians(5.0)) / math.tan(math.radians(20.0))
        )
        self.assertAlmostEqual(
            box.heightPx, 256.0 * math.tan(math.radians(2.5)) / math.tan(math.radians(20.0))
        )

    def testOffsetsFollowTheViewAxesAndAgreeWithTheGeometryBackProjection(self) -> None:
        geometry = SphericalGeometryImpl()
        local = BBoxXYWH(150.0, 60.0, 40.0, 30.0)
        box = localBoxOfBfov(self.view, geometry.localBoxToBfov(local, self.view))

        self.assertAlmostEqual(box.xPx + box.widthPx / 2.0, 170.0, delta=1.0)
        self.assertAlmostEqual(box.yPx + box.heightPx / 2.0, 75.0, delta=1.0)
        self.assertAlmostEqual(box.widthPx, 40.0, delta=2.0)
        self.assertAlmostEqual(box.heightPx, 30.0, delta=2.0)

    def testInASphericalViewTheBoxIsLinearInAngleAndInvertsTheGeometry(self) -> None:
        view = ViewSpec(
            0,
            BFoV(makeSphericalPoint(0.3, -0.2), math.radians(240.0), math.radians(120.0)),
            256,
            128,
            projection=ViewProjection.SPHERICAL,
        )
        geometry = SphericalGeometryImpl()
        local = BBoxXYWH(150.0, 30.0, 60.0, 40.0)

        bfov = geometry.localBoxToBfov(local, view)
        box = localBoxOfBfov(view, bfov)

        # 256 px across 240 degrees: 60 px is 56.25 degrees of longitude, which at the
        # box's latitude of 13.125 degrees is that much narrower on the sphere.
        self.assertAlmostEqual(
            math.degrees(bfov.horizontalFovRad), 56.25 * math.cos(math.radians(13.125))
        )
        self.assertAlmostEqual(math.degrees(bfov.verticalFovRad), 37.5)
        for actual, expected in zip(
            (box.xPx, box.yPx, box.widthPx, box.heightPx),
            (local.xPx, local.yPx, local.widthPx, local.heightPx),
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected, places=6)
        # A direction behind the view center still has a finite place in the view.
        behind = localBoxOfBfov(view, BFoV(makeSphericalPoint(0.3 - 3.0, 0.2), 0.2, 0.2))
        self.assertLess(behind.xPx + behind.widthPx / 2.0, 0.0)

    def testADirectionBehindTheViewLandsFarOutsideWithFiniteCoordinates(self) -> None:
        behind = makeSphericalPoint(0.3 - math.pi, 0.2)
        box = localBoxOfBfov(self.view, BFoV(behind, 0.2, 0.2))

        self.assertTrue(all(math.isfinite(v) for v in (box.xPx, box.yPx)))
        center = (box.xPx + box.widthPx / 2.0, box.yPx + box.heightPx / 2.0)
        self.assertTrue(abs(center[0] - 128.0) > 256.0 or abs(center[1] - 128.0) > 256.0)


class ControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = loadConfig(ROOT / "configs" / "default.yaml")
        self.geometry = SphericalGeometryImpl(
            boundarySamplesPerEdge=self.config.geometry.boundarySamplesPerEdge
        )

    def _controller(self, config=None, **kwargs: object) -> TrackControllerImpl:
        controller = TrackControllerImpl(self.geometry, config or self.config, **kwargs)
        controller.commitInitialization(controller.buildInitialization(_frame(0), INITIAL_BOX))
        return controller

    def testPlansOneViewAroundThePredictionAndCommitsOrderedFrames(self) -> None:
        controller = TrackControllerImpl(self.geometry, self.config)
        initPlan = controller.buildInitialization(_frame(0), BBoxXYWH(150.0, 70.0, 60.0, 80.0))
        self.assertEqual(initPlan.stateRevision, 0)
        self.assertLess(initPlan.templateBox.widthPx, 256.0)
        self.assertGreaterEqual(initPlan.templateBox.xPx, 0.0)
        initial = controller.commitInitialization(initPlan)

        plan = controller.beginFrame(_frame(1))
        self.assertEqual(plan.stateRevision, 1)
        self.assertAlmostEqual(plan.view.bfov.center.yawRad, initial.bfov.center.yawRad)
        self.assertAlmostEqual(plan.view.bfov.center.pitchRad, initial.bfov.center.pitchRad)
        # The default view is square: four times the mean target size, 90 degrees at most.
        meanHalfSize = math.sqrt(
            math.tan(initial.bfov.horizontalFovRad / 2.0)
            * math.tan(initial.bfov.verticalFovRad / 2.0)
        )
        # A target this large needs a search region past the 120 degree switch point.
        self.assertGreater(2.0 * math.atan(4.0 * meanHalfSize), math.radians(120.0))
        self.assertIs(plan.view.projection, ViewProjection.SPHERICAL)
        self.assertIs(initPlan.templateView.projection, ViewProjection.SPHERICAL)
        self.assertIsNone(initPlan.templateView.priorBox)
        self.assertIsNotNone(plan.view.priorBox)

        result = controller.consume(plan, _observation(0.95))
        self.assertTrue(result.valid)
        self.assertEqual(result.status, TrackStatus.TRACKING)
        self.assertEqual(result.resultSource, ResultSource.OBSERVED_CONFIRMED)
        self.assertEqual(result.bbox, _observation(0.95).bbox)
        assert controller.lastStateObservation is not None
        # The published confidence is the backend's score; the state score also counts
        # how well the box agrees with the motion prediction.
        self.assertAlmostEqual(result.confidence, 0.95)
        self.assertAlmostEqual(controller.lastStateObservation.backendScore, 0.95)
        self.assertLessEqual(controller.lastStateObservation.stateScore, 0.95)

        nextPlan = controller.beginFrame(_frame(2))
        self.assertEqual(nextPlan.stateRevision, 2)

    def testRejectsStalePlansAndForeignObservations(self) -> None:
        controller = self._controller()
        plan = controller.beginFrame(_frame(1))
        with self.assertRaisesRegex(ProtocolError, "already awaiting"):
            controller.beginFrame(_frame(1))
        with self.assertRaisesRegex(ProtocolError, "planned view"):
            controller.consume(plan, _observation(0.9, viewId=99))

        controller.consume(plan, _observation(0.9))
        with self.assertRaisesRegex(ProtocolError, "pending plan"):
            controller.consume(plan, _observation(0.9))
        with self.assertRaisesRegex(ProtocolError, "frame index"):
            controller.beginFrame(_frame(3))

    def testReturnsPredictionWhenTheViewYieldsNoBox(self) -> None:
        controller = self._controller()
        result = controller.consume(controller.beginFrame(_frame(1)), None)

        self.assertFalse(result.valid)
        # A frame without a box has a state score of zero: it is not trusted.
        self.assertEqual(result.status, TrackStatus.UNCERTAIN)
        self.assertEqual(result.resultSource, ResultSource.MOTION_PREDICTED)
        self.assertGreater(result.bbox.widthPx, 0.0)

    def testRepeatedMissesAreUncertainFirstAndLostAfterEnoughOfThem(self) -> None:
        controller = self._controller()
        statuses = [
            controller.consume(controller.beginFrame(_frame(index)), None).status
            for index in range(1, 7)
        ]
        patience = self.config.backendTuning.lostAfterFrames

        self.assertEqual(statuses[: patience - 1], [TrackStatus.UNCERTAIN] * (patience - 1))
        self.assertEqual(statuses[patience - 1 :], [TrackStatus.LOST] * (7 - patience))
        # One trusted box brings the track back.
        back = controller.consume(controller.beginFrame(_frame(7)), _observation(0.95))
        self.assertEqual(back.status, TrackStatus.TRACKING)

    def testDefaultConfigAcceptsALowScoredBox(self) -> None:
        controller = self._controller()
        result = controller.consume(controller.beginFrame(_frame(1)), _observation(0.10))

        self.assertTrue(result.valid)
        self.assertEqual(result.resultSource, ResultSource.OBSERVED_CONFIRMED)

    def testAFrameWithoutABoxHoldsTheBoxOfALargeTarget(self) -> None:
        largeBox = BBoxXYWH(100.0, 40.0, 150.0, 100.0)  # more than 10% of the frame
        controller = TrackControllerImpl(self.geometry, self.config)
        controller.commitInitialization(controller.buildInitialization(_frame(0), largeBox))
        held = controller.consume(controller.beginFrame(_frame(1)), None)

        self.assertTrue(held.valid)
        self.assertEqual(held.bbox, largeBox)
        # A small target is not held: its frame without a box is not a valid result.
        small = self._controller()
        self.assertFalse(small.consume(small.beginFrame(_frame(1)), None).valid)

    def testFallbackAdvancesProtocolAndAllowsNextFrame(self) -> None:
        controller = self._controller()
        controller.beginFrame(_frame(1))
        fallback = controller.commitFallback(_frame(1), reason="GeometryError")

        self.assertFalse(fallback.valid)
        self.assertEqual(fallback.frameIndex, FrameIndex(1))
        self.assertEqual(controller.lastPipelineProfile["reason"], "GeometryError")
        self.assertEqual(controller.beginFrame(_frame(2)).frameIndex, FrameIndex(2))

    def testPlansCarryTheLastSevenTargetBoxesInViewPixels(self) -> None:
        controller = self._controller()
        first = controller.beginFrame(_frame(1))

        # Before any measurement the history is the initial target, seven times,
        # and the view is centered on it.
        self.assertEqual(len(first.view.trajectory), 7)
        self.assertEqual(len(set(first.view.trajectory)), 1)
        box = first.view.trajectory[0]
        self.assertAlmostEqual(
            box.xPx + box.widthPx / 2.0, first.view.outputWidthPx / 2.0, places=6
        )
        self.assertAlmostEqual(
            box.yPx + box.heightPx / 2.0, first.view.outputHeightPx / 2.0, places=6
        )
        assert first.view.priorBox is not None
        self.assertAlmostEqual(box.widthPx, first.view.priorBox.widthPx, places=5)

        moved = BFoV(makeSphericalPoint(0.2, 0.0), 0.35, 0.25)
        controller.consume(first, replace(_observation(0.9), bfov=moved))
        second = controller.beginFrame(_frame(2))

        # The newest entry is the box just committed; the older ones are unchanged
        # on the sphere, so in the new view they sit to one side of it.
        self.assertEqual(len(second.view.trajectory), 7)
        newest, oldest = second.view.trajectory[-1], second.view.trajectory[0]
        self.assertEqual(len(set(second.view.trajectory[:-1])), 1)
        self.assertNotAlmostEqual(
            newest.xPx + newest.widthPx / 2.0, oldest.xPx + oldest.widthPx / 2.0, places=1
        )


if __name__ == "__main__":
    unittest.main()
