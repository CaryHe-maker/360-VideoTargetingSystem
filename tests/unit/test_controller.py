import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from track360.controller import (
    SphericalMotionEstimator,
    TemplatePolicy,
    TrackControllerImpl,
    TrackStateMachine,
    ViewPlanner,
    scoreMotionConsistency,
)
from track360.controller.state_model import ScoreGroup, TransitionReason
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
    TemplateCommandKind,
    TrackStatus,
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
        motionScore = scoreMotionConsistency(
            BFoV(makeSphericalPoint(0.0, 0.0), 0.40, 0.30),
            prediction.motionState,
        )

        self.assertEqual(prediction.sampleCount, 1)
        self.assertIn("insufficient_motion_samples", prediction.degradedReasons)
        self.assertGreater(prediction.reliability, 0.0)
        self.assertLess(prediction.reliability, prediction.confidence)
        self.assertGreater(motionScore.effectiveProbability, 0.5)


class StateMachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = loadConfig(ROOT / "configs" / "default.yaml")

    def testUsesScoreGroupThresholds(self) -> None:
        state = TrackStateMachine(self.config.tracking)
        state.initialize()
        first = state.update(0.5, True, True)
        second = state.update(0.5, True, True)
        uncertain = state.update(0.4, True, True)
        recovered = state.update(0.9, True, True)

        self.assertEqual(first.status, TrackStatus.TRACKING)
        self.assertEqual(second.status, TrackStatus.TRACKING)
        self.assertEqual(uncertain.status, TrackStatus.UNCERTAIN)
        self.assertTrue(recovered.accepted)
        self.assertEqual(recovered.status, TrackStatus.TRACKING)

    def testKeepsHardMissInUncertain(self) -> None:
        state = TrackStateMachine(self.config.tracking)
        state.initialize()
        state.update(0.8, True, True)
        state.update(0.8, True, True)
        state.update(0.8, True, True)

        hardMiss = state.update(0.0, False, False)

        self.assertEqual(hardMiss.status, TrackStatus.UNCERTAIN)
        self.assertEqual(hardMiss.reason, TransitionReason.HARD_MISS)

    def testScoreGroupUsesWarmupAndRollingOrderStatistics(self) -> None:
        group = ScoreGroup()
        self.assertIsNone(group.thresholds())
        group.append(0.2)
        self.assertIsNone(group.thresholds())
        group.append(0.8)
        self.assertAlmostEqual(group.thresholds()[0], 0.5)
        self.assertAlmostEqual(group.thresholds()[1], 0.32)
        for value in (0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.95):
            group.append(value)
        self.assertEqual(group.thresholds(), (0.6, 0.3))
        self.assertEqual(len(group.values), 10)


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

    def testViewIsCenteredAndThreeTimesTheTargetOnEachAxis(self) -> None:
        view = self._planner(alignedSearch=False).searchView(
            self.center, math.radians(20.0), math.radians(15.0)
        )

        self.assertEqual(view.bfov.center, self.center)
        self.assertAlmostEqual(view.bfov.horizontalFovRad, math.radians(60.0))
        self.assertAlmostEqual(view.bfov.verticalFovRad, math.radians(45.0))
        self.assertEqual(
            (view.outputWidthPx, view.outputHeightPx),
            (self.config.geometry.viewWidthPx, self.config.geometry.viewHeightPx),
        )

    def testViewFovIsClampedToGeometryLimitsAndCaps(self) -> None:
        legacy = self._planner(alignedSearch=False)
        small = legacy.searchView(self.center, math.radians(2.0), math.radians(3.0))
        large = legacy.searchView(self.center, math.radians(50.0), math.radians(50.0))
        uncapped = self._planner(
            alignedSearch=False, viewHorizontalFovCapRad=None, viewVerticalFovCapRad=None
        ).searchView(self.center, math.radians(50.0), math.radians(50.0))

        self.assertAlmostEqual(small.bfov.horizontalFovRad, self.config.geometry.minFovRad)
        self.assertAlmostEqual(small.bfov.verticalFovRad, self.config.geometry.minFovRad)
        self.assertAlmostEqual(large.bfov.horizontalFovRad, math.radians(90.0))
        self.assertAlmostEqual(large.bfov.verticalFovRad, math.radians(90.0))
        self.assertAlmostEqual(uncapped.bfov.horizontalFovRad, self.config.geometry.maxFovRad)


    def testAlignedViewIsSquareAndFourTimesTheMeanTargetSize(self) -> None:
        planner = self._planner(alignedSearch=True)
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
        legacy = self._planner(alignedSearch=False)
        self.assertIsNone(legacy.searchView(self.center, 0.2, 0.1).priorBox)

    def testAlignedViewFollowsSmallTargetsBelowTheGeometryMinimum(self) -> None:
        planner = self._planner(alignedSearch=True)
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

    def testAlignedTemplateViewIsSquareSoTheTargetKeepsItsAspect(self) -> None:
        target = BFoV(self.center, math.radians(12.0), math.radians(3.0))
        legacy = self._planner(alignedSearch=False).templateBfov(target)
        aligned = self._planner(alignedSearch=True).templateBfov(target)

        self.assertAlmostEqual(legacy.horizontalFovRad, math.radians(30.0))
        self.assertAlmostEqual(legacy.verticalFovRad, self.config.geometry.minFovRad)
        self.assertEqual(aligned.horizontalFovRad, aligned.verticalFovRad)
        self.assertEqual(aligned.center, target.center)

    def testAlignedSearchExcludesFullViewSearch(self) -> None:
        with self.assertRaisesRegex(ConfigError, "cannot both be enabled"):
            replace(self.config.backendTuning, alignedSearch=True, fullViewSearch=True)


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

    def testADirectionBehindTheViewLandsFarOutsideWithFiniteCoordinates(self) -> None:
        behind = makeSphericalPoint(0.3 - math.pi, 0.2)
        box = localBoxOfBfov(self.view, BFoV(behind, 0.2, 0.2))

        self.assertTrue(all(math.isfinite(v) for v in (box.xPx, box.yPx)))
        center = (box.xPx + box.widthPx / 2.0, box.yPx + box.heightPx / 2.0)
        self.assertTrue(abs(center[0] - 128.0) > 256.0 or abs(center[1] - 128.0) > 256.0)


class ControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = loadConfig(ROOT / "configs" / "default.yaml")
        self.gatedConfig = replace(
            self.config,
            backendTuning=replace(self.config.backendTuning, acceptAnyCandidate=False),
        )
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
        self.assertEqual(plan.templateCommand.expectedRevision, 1)
        self.assertEqual(plan.templateCommand.kind, TemplateCommandKind.KEEP)
        self.assertAlmostEqual(plan.view.bfov.center.yawRad, initial.bfov.center.yawRad)
        self.assertAlmostEqual(plan.view.bfov.center.pitchRad, initial.bfov.center.pitchRad)
        # The default view is square: four times the mean target size, 90 degrees at most.
        meanHalfSize = math.sqrt(
            math.tan(initial.bfov.horizontalFovRad / 2.0)
            * math.tan(initial.bfov.verticalFovRad / 2.0)
        )
        self.assertEqual(plan.view.bfov.horizontalFovRad, plan.view.bfov.verticalFovRad)
        self.assertAlmostEqual(
            plan.view.bfov.horizontalFovRad,
            min(2.0 * math.atan(4.0 * meanHalfSize), math.radians(90.0)),
        )
        self.assertIsNotNone(plan.view.priorBox)

        result = controller.consume(plan, _observation(0.95))
        self.assertTrue(result.valid)
        self.assertEqual(result.status, TrackStatus.TRACKING)
        self.assertEqual(result.resultSource, ResultSource.OBSERVED_CONFIRMED)
        self.assertEqual(result.bbox, _observation(0.95).bbox)
        assert controller.lastStateObservation is not None
        self.assertAlmostEqual(controller.lastStateObservation.stateScore, 0.95)

        nextPlan = controller.beginFrame(_frame(2))
        self.assertEqual(nextPlan.stateRevision, 2)
        self.assertEqual(nextPlan.templateCommand.expectedRevision, 2)

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
        self.assertEqual(result.status, TrackStatus.TRACKING)
        self.assertEqual(result.resultSource, ResultSource.MOTION_PREDICTED)
        self.assertGreater(result.bbox.widthPx, 0.0)

    def testRepeatedMissesStayUncertainAndNeverBecomeLost(self) -> None:
        controller = self._controller()
        for index in range(1, 6):
            result = controller.consume(controller.beginFrame(_frame(index)), None)
            self.assertNotEqual(result.status, TrackStatus.LOST)
        self.assertEqual(result.status, TrackStatus.UNCERTAIN)

    def testDefaultConfigAcceptsALowScoredBox(self) -> None:
        controller = self._controller()
        result = controller.consume(controller.beginFrame(_frame(1)), _observation(0.10))

        self.assertTrue(result.valid)
        self.assertEqual(result.resultSource, ResultSource.OBSERVED_CONFIRMED)

    def testGatedConfigRejectsABoxBelowCandidateMinScore(self) -> None:
        controller = self._controller(self.gatedConfig)
        weak = controller.consume(controller.beginFrame(_frame(1)), _observation(0.10, xPx=20.0))

        self.assertFalse(weak.valid)
        self.assertEqual(weak.resultSource, ResultSource.OBSERVED_WEAK_BLEND)
        self.assertAlmostEqual(weak.bbox.xPx, 20.0)

        strong = controller.consume(controller.beginFrame(_frame(2)), _observation(0.90))
        self.assertTrue(strong.valid)

    def testWeakFirstObservationBootstrapsVelocityBeforeThirdFrame(self) -> None:
        estimator = SphericalMotionEstimator(windowLength=3, minSamplesForVelocity=2)
        controller = self._controller(self.gatedConfig, motionEstimator=estimator)

        weak = controller.consume(controller.beginFrame(_frame(1)), _observation(0.10))
        self.assertFalse(weak.valid)
        self.assertEqual(len(estimator.samples), 2)

        controller.beginFrame(_frame(2))
        prediction = estimator.predictDetailed(_frame(2).timestampNs)
        self.assertEqual(prediction.sampleCount, 2)
        self.assertNotIn("insufficient_motion_samples", prediction.degradedReasons)

    def testRejectedMeasurementHoldsTheBoxOfALargeTarget(self) -> None:
        largeBox = BBoxXYWH(100.0, 40.0, 150.0, 100.0)  # more than 10% of the frame

        def track(config):
            controller = TrackControllerImpl(self.geometry, config)
            controller.commitInitialization(controller.buildInitialization(_frame(0), largeBox))
            return controller.consume(
                controller.beginFrame(_frame(1)), _observation(0.10, xPx=20.0)
            )

        held = track(self.gatedConfig)
        released = track(
            replace(
                self.gatedConfig,
                backendTuning=replace(self.gatedConfig.backendTuning, holdWeakBox=False),
            )
        )

        self.assertTrue(held.valid)
        self.assertEqual(held.bbox, largeBox)
        self.assertFalse(released.valid)
        self.assertAlmostEqual(released.bbox.xPx, 20.0)

    def testFallbackAdvancesProtocolAndAllowsNextFrame(self) -> None:
        controller = self._controller()
        controller.beginFrame(_frame(1))
        fallback = controller.commitFallback(_frame(1), backendRevision=1, reason="GeometryError")

        self.assertFalse(fallback.valid)
        self.assertEqual(fallback.frameIndex, FrameIndex(1))
        self.assertEqual(controller.lastPipelineProfile["reason"], "GeometryError")
        self.assertEqual(controller.beginFrame(_frame(2)).frameIndex, FrameIndex(2))

    def testConfidentObservationSchedulesARecentTemplateUpdate(self) -> None:
        controller = self._controller(
            replace(
                self.config,
                backendTuning=replace(self.config.backendTuning, sequenceModel=False),
            )
        )
        controller.consume(controller.beginFrame(_frame(1)), _observation(0.95))
        controller.consume(controller.beginFrame(_frame(2)), _observation(0.95))

        command = controller.beginFrame(_frame(3)).templateCommand
        self.assertEqual(command.kind, TemplateCommandKind.UPDATE_RECENT)
        self.assertEqual(command.viewId, 0)
        self.assertEqual(command.localBox, _observation(0.95).localBox)
        self.assertEqual(command.expectedRevision, 3)

    def testPlansCarryTheLastSevenTargetBoxesInViewPixels(self) -> None:
        controller = self._controller()
        first = controller.beginFrame(_frame(1))

        # Before any measurement the history is the initial target, seven times,
        # and the view is centered on it.
        self.assertEqual(len(first.view.trajectory), 7)
        self.assertEqual(len(set(first.view.trajectory)), 1)
        box = first.view.trajectory[0]
        self.assertAlmostEqual(box.xPx + box.widthPx / 2.0, 128.0, places=6)
        self.assertAlmostEqual(box.yPx + box.heightPx / 2.0, 128.0, places=6)
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


class TemplatePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = loadConfig(ROOT / "configs" / "default.yaml")
        self.geometry = SphericalGeometryImpl()

    def _decide(self, score: float, stableFrames: int, **tuning: object):
        # Template updates belong to the frame-level model; see the last test.
        tuning = {"sequenceModel": False, **tuning}
        config = replace(
            self.config, backendTuning=replace(self.config.backendTuning, **tuning)
        )
        controller = TrackControllerImpl(self.geometry, config)
        controller.commitInitialization(controller.buildInitialization(_frame(0), INITIAL_BOX))
        controller.consume(controller.beginFrame(_frame(1)), _observation(score))
        return TemplatePolicy(config.tracking, config.backendTuning).decide(
            TrackStatus.TRACKING, stableFrames, controller.lastStateObservation
        )

    def testKeepsTheAnchorWhenOnlineTemplatesAreDisabled(self) -> None:
        period = self.config.tracking.stableFramesBeforeUpdate
        decision = self._decide(0.99, 2 * period, onlineTemplate=False)

        self.assertEqual(decision.kind, TemplateCommandKind.KEEP)
        self.assertIsNone(decision.viewId)
        self.assertIsNone(decision.localBox)

    def testWeakObservationsNeverRefreshATemplate(self) -> None:
        belowThreshold = self.config.backendTuning.templateMinConfidence - 0.01
        self.assertEqual(self._decide(belowThreshold, 2).kind, TemplateCommandKind.KEEP)

    def testSequenceModelTakesNoTemplateUpdates(self) -> None:
        self.assertTrue(self.config.backendTuning.sequenceModel)
        self.assertEqual(
            self._decide(0.99, 2, sequenceModel=True).kind, TemplateCommandKind.KEEP
        )

    def testRecentRefreshesEveryOtherFrameAndStableOncePerPeriod(self) -> None:
        period = self.config.tracking.stableFramesBeforeUpdate
        self.assertEqual(self._decide(0.9, 1).kind, TemplateCommandKind.KEEP)
        self.assertEqual(self._decide(0.9, 2).kind, TemplateCommandKind.UPDATE_RECENT)
        self.assertEqual(self._decide(0.9, period).kind, TemplateCommandKind.UPDATE_STABLE)


if __name__ == "__main__":
    unittest.main()
