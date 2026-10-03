import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from track360.controller import (
    DecisionGate,
    RecoveryPlanner,
    SphericalMotionEstimator,
    TrackControllerImpl,
    scoreMotionConsistency,
)
from track360.controller.state_model import TrackMode
from track360.core.config import DecisionGateConfig, loadConfig
from track360.core.protocols import FrameCommitted, MoreViewsRequired
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    FramePacket,
    ProjectedObservation,
    ResultSource,
    SequenceId,
    TrackStatus,
)
from track360.geometry import SphericalGeometryImpl, makeSphericalPoint

ROOT = Path(__file__).resolve().parents[2]


def _candidate(viewId: int, yawRad: float, score: float = 0.9) -> ProjectedObservation:
    bfov = BFoV(makeSphericalPoint(yawRad, 0.0), 0.35, 0.25)
    return ProjectedObservation(
        viewId=viewId,
        bfov=bfov,
        bbox=BBoxXYWH(20.0 + viewId * 120.0, 60.0, 25.0, 20.0),
        modelScore=score,
        appearanceScore=score,
        motionScore=score,
        scaleScore=score,
        fusedScore=score,
        localBox=BBoxXYWH(80.0, 80.0, 30.0, 30.0),
    )


def _boxedCandidate(
    viewId: int,
    xPx: float,
    score: float,
    *,
    widthPx: float = 100.0,
) -> ProjectedObservation:
    bfov = BFoV(makeSphericalPoint(0.0, 0.0), 0.35, 0.25)
    return ProjectedObservation(
        viewId=viewId,
        bfov=bfov,
        bbox=BBoxXYWH(xPx, 40.0, widthPx, 80.0),
        modelScore=score,
        appearanceScore=score,
        motionScore=score,
        scaleScore=score,
        fusedScore=score,
        localBox=BBoxXYWH(50.0, 50.0, 80.0, 80.0),
    )


def _scoredCandidate(
    viewId: int,
    xPx: float,
    yawRad: float,
    *,
    appearanceProbability: float,
    singleScore: float,
) -> ProjectedObservation:
    return ProjectedObservation(
        viewId=viewId,
        bfov=BFoV(makeSphericalPoint(yawRad, 0.0), 0.20, 0.15),
        bbox=BBoxXYWH(xPx, 50.0, 40.0, 40.0),
        modelScore=appearanceProbability,
        appearanceScore=appearanceProbability,
        motionScore=singleScore,
        scaleScore=1.0,
        fusedScore=singleScore,
        localBox=BBoxXYWH(50.0, 50.0, 40.0, 40.0),
        backendFusedScore=appearanceProbability,
        appearanceProbability=appearanceProbability,
        singleScore=singleScore,
    )


class ControllerV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.config = loadConfig(ROOT / "configs" / "default.yaml")
        self.geometry = SphericalGeometryImpl(
            boundarySamplesPerEdge=self.config.geometry.boundarySamplesPerEdge
        )

    def testDisjointCandidatesAreNotMergedIntoOneUnionBox(self) -> None:
        gate = DecisionGate(DecisionGateConfig(0.25, 0.15), self.config.tracking)
        aggregate = gate.aggregate(
            (_candidate(0, 0.0), _candidate(1, 2.2)),
            self.geometry,
            360,
            180,
        )
        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        self.assertEqual(len(aggregate.sourceViewIds), 1)
        self.assertEqual(aggregate.clusterCount, 2)
        self.assertLess(aggregate.bbox.widthPx, 100.0)

    def testTrackingUsesTwoRoundsAndCommitsOneResult(self) -> None:
        frame0 = FramePacket(
            SequenceId("v2"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init)
        frame1 = FramePacket(
            SequenceId("v2"),
            FrameIndex(1),
            1_000_000_000,
            frame0.rgb,
        )
        firstPlan = controller.beginFrame(frame1)
        firstStep = controller.consume(firstPlan, ())
        self.assertIsInstance(firstStep, MoreViewsRequired)
        assert isinstance(firstStep, MoreViewsRequired)
        self.assertEqual(firstStep.plan.transactionId, firstPlan.transactionId)
        self.assertEqual(firstStep.plan.attemptIndex, 1)
        self.assertEqual(firstPlan.templateCommand.expectedRevision, 1)
        self.assertEqual(firstStep.plan.templateCommand.expectedRevision, 2)
        secondStep = controller.consume(firstStep.plan, ())
        self.assertIsInstance(secondStep, FrameCommitted)
        assert isinstance(secondStep, FrameCommitted)
        self.assertEqual(secondStep.result.frameIndex, FrameIndex(1))
        self.assertFalse(secondStep.result.valid)
        self.assertEqual(secondStep.result.status, TrackStatus.TRACKING)

    def testPipelineUsesRound1AsProvisionalPredictionWithoutCommittingHistory(self) -> None:
        estimator = SphericalMotionEstimator(windowLength=4, minSamplesForVelocity=2)
        frame0 = FramePacket(
            SequenceId("pipeline-provisional"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(
            self.geometry,
            self.config,
            motionEstimator=estimator,
        )
        controller.commitInitialization(
            controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        )
        frame1 = FramePacket(
            SequenceId("pipeline-provisional"),
            FrameIndex(1),
            1_000_000_000,
            frame0.rgb,
        )
        firstPlan = controller.beginFrame(frame1)
        firstCenter = firstPlan.views[0].bfov.center.yawRad
        step = controller.consume(
            firstPlan,
            (_boxedCandidate(firstPlan.views[0].viewId, firstCenter + 0.20, 0.95),),
        )

        self.assertIsInstance(step, MoreViewsRequired)
        assert isinstance(step, MoreViewsRequired)
        self.assertIn("round1_provisional", controller._planned.prediction.degradedReasons)
        self.assertEqual(len(estimator.samples), 1)
        committed = controller.consume(step.plan, ())
        self.assertIsInstance(committed, FrameCommitted)
        # The provisional update itself never mutates history. The existing fallback may select
        # the R1 observation as the frame's final measurement, at which point it is committed once.
        self.assertEqual(len(estimator.samples), 2)
        self.assertEqual(estimator.samples[-1].frameIndex, FrameIndex(1))

    def testEscalationClampsInvalidProvisionalFovToGeometryBounds(self) -> None:
        frame0 = FramePacket(
            SequenceId("provisional-fov-bounds"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(self.geometry, self.config)
        controller.commitInitialization(
            controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        )
        frame1 = FramePacket(
            SequenceId("provisional-fov-bounds"),
            FrameIndex(1),
            1_000_000_000,
            frame0.rgb,
        )
        firstPlan = controller.beginFrame(frame1)
        original = controller._provisionalRoundPrediction
        controller._provisionalRoundPrediction = lambda planned, evaluation: replace(
            original(planned, evaluation),
            horizontalSizeRad=4.0,
            verticalSizeRad=4.0,
        )

        step = controller.consume(
            firstPlan,
            (_boxedCandidate(firstPlan.views[0].viewId, 20.0, 0.1),),
        )

        self.assertIsInstance(step, MoreViewsRequired)
        assert isinstance(step, MoreViewsRequired)
        assert controller._planned is not None
        self.assertEqual(
            controller._planned.predictedBfov.horizontalFovRad,
            self.config.geometry.maxFovRad,
        )
        self.assertEqual(
            controller._planned.predictedBfov.verticalFovRad,
            self.config.geometry.maxFovRad,
        )

    def testFallbackAdvancesProtocolAndAllowsNextFrame(self) -> None:
        frame0 = FramePacket(
            SequenceId("controller-fallback"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(self.geometry, self.config)
        controller.commitInitialization(
            controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        )
        frame1 = FramePacket(
            SequenceId("controller-fallback"),
            FrameIndex(1),
            1_000_000_000,
            frame0.rgb,
        )
        controller.beginFrame(frame1)
        fallback = controller.commitFallback(
            frame1,
            backendRevision=1,
            reason="GeometryError",
        )

        self.assertFalse(fallback.valid)
        self.assertEqual(fallback.frameIndex, FrameIndex(1))
        self.assertEqual(controller.lastPipelineProfile["reason"], "GeometryError")

        frame2 = FramePacket(
            SequenceId("controller-fallback"),
            FrameIndex(2),
            2_000_000_000,
            frame0.rgb,
        )
        nextPlan = controller.beginFrame(frame2)
        self.assertEqual(nextPlan.frameIndex, FrameIndex(2))

    def testSecondRoundFusesCandidatesFromBothRounds(self) -> None:
        frame0 = FramePacket(
            SequenceId("cumulative-round2"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init)
        frame1 = FramePacket(
            SequenceId("cumulative-round2"),
            FrameIndex(1),
            1_000_000_000,
            frame0.rgb,
        )

        firstPlan = controller.beginFrame(frame1)
        firstObservation = _scoredCandidate(
            0,
            20.0,
            0.6,
            appearanceProbability=0.85,
            singleScore=0.85,
        )
        secondStep = controller.consume(firstPlan, (firstObservation,))
        assert isinstance(secondStep, MoreViewsRequired)
        self.assertEqual(len(secondStep.plan.views), 4)
        direction = np.sum(
            np.asarray(
                [
                    (view.bfov.center.x, view.bfov.center.y, view.bfov.center.z)
                    for view in secondStep.plan.views
                ]
            ),
            axis=0,
        )
        direction /= np.linalg.norm(direction)
        np.testing.assert_allclose(
            direction,
            np.asarray(
                (
                    firstObservation.bfov.center.x,
                    firstObservation.bfov.center.y,
                    firstObservation.bfov.center.z,
                )
            ),
            atol=1e-10,
        )
        refinedViewId = secondStep.plan.views[0].viewId
        final = controller.consume(secondStep.plan, (_boxedCandidate(refinedViewId, 20.0, 0.85),))

        assert isinstance(final, FrameCommitted)
        self.assertTrue(final.result.valid)
        assert controller.lastStateObservation is not None
        self.assertTrue(controller.lastStateObservation.selectedIsFused)
        self.assertEqual(controller.lastStateObservation.sourceViewIds, (0, refinedViewId))
        self.assertEqual(controller.lastStateObservation.candidateCount, 2)
        self.assertEqual(controller.lastStateObservation.overlapThreshold, 0.70)

    def testControllerUsesExactlyTwoRounds(self) -> None:
        frame0 = FramePacket(
            SequenceId("cumulative-round3"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init)

        frame1 = FramePacket(
            SequenceId("cumulative-round3"),
            FrameIndex(1),
            1_000_000_000,
            frame0.rgb,
        )
        plan = controller.beginFrame(frame1)
        step = controller.consume(plan, ())
        assert isinstance(step, MoreViewsRequired)
        committed = controller.consume(step.plan, ())
        assert isinstance(committed, FrameCommitted)
        self.assertEqual(committed.result.status, TrackStatus.TRACKING)
        self.assertEqual(committed.result.valid, False)

    def testSecondRoundKeepsSingleScoreSemantics(self) -> None:
        frame0 = FramePacket(
            SequenceId("appearance-round3"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init)

        frame1 = FramePacket(
            SequenceId("appearance-round3"),
            FrameIndex(1),
            1_000_000_000,
            frame0.rgb,
        )
        first = controller.beginFrame(frame1)
        second = controller.consume(first, ())
        assert isinstance(second, MoreViewsRequired)
        committed = controller.consume(second.plan, ())
        assert isinstance(committed, FrameCommitted)
        self.assertEqual(committed.result.status, TrackStatus.TRACKING)

        frame2 = FramePacket(
            SequenceId("appearance-round3"), FrameIndex(2), 2_000_000_000, frame0.rgb
        )
        round1 = controller.beginFrame(frame2)
        self.assertFalse(round1.appearanceOnlyScoring)
        round2 = controller.consume(
            round1,
            (_scoredCandidate(0, 10.0, -1.0, appearanceProbability=0.10, singleScore=0.89),),
        )
        assert isinstance(round2, MoreViewsRequired)
        self.assertFalse(round2.plan.appearanceOnlyScoring)
        final = controller.consume(
            round2.plan,
            (
                _scoredCandidate(
                    round2.plan.views[0].viewId,
                    250.0,
                    1.0,
                    appearanceProbability=0.85,
                    singleScore=0.10,
                ),
            ),
        )

        assert isinstance(final, FrameCommitted)
        assert controller.lastStateObservation is not None
        self.assertFalse(controller.lastStateObservation.appearanceOnlyScoring)
        self.assertEqual(controller.lastStateObservation.representativeViewId, 0)
        self.assertAlmostEqual(controller.lastStateObservation.stateScore, 0.89)

    def testMotionHistoryContainsMeasurementsNotPredictions(self) -> None:
        estimator = SphericalMotionEstimator(windowLength=3)
        estimator.initialize(makeSphericalPoint(math.pi - 0.05, 0.0), 0)
        estimator.predict(500_000_000)
        self.assertEqual(len(estimator.samples), 1)
        estimator.update(
            makeSphericalPoint(-math.pi + 0.05, 0.0),
            1_000_000_000,
            0.9,
        )
        self.assertEqual(len(estimator.samples), 2)
        prediction = estimator.predictDetailed(2_000_000_000)
        self.assertGreater(prediction.angularUncertaintyRad, 0.0)

    def testMotionPredictionHasStableTangentBasisAtPole(self) -> None:
        estimator = SphericalMotionEstimator(windowLength=3)
        estimator.initialize(makeSphericalPoint(0.0, math.pi / 2.0 - 0.10), 0)
        estimator.update(
            makeSphericalPoint(0.0, math.pi / 2.0),
            1_000_000_000,
            0.9,
        )
        prediction = estimator.predictDetailed(2_000_000_000)
        self.assertTrue(math.isfinite(prediction.center.yawRad))
        self.assertTrue(math.isfinite(prediction.center.pitchRad))
        self.assertLess(prediction.center.pitchRad, math.pi / 2.0)

    def testMotionPredictionExtrapolatesScaleInLogSpace(self) -> None:
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

    def testWeakFirstObservationBootstrapsVelocityBeforeThirdFrame(self) -> None:
        estimator = SphericalMotionEstimator(windowLength=3, minSamplesForVelocity=2)
        frame0 = FramePacket(
            SequenceId("motion-bootstrap"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(
            self.geometry,
            self.config,
            motionEstimator=estimator,
        )
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init)

        frame1 = FramePacket(
            SequenceId("motion-bootstrap"),
            FrameIndex(1),
            1_000_000_000,
            frame0.rgb,
        )
        firstPlan = controller.beginFrame(frame1)
        secondStep = controller.consume(firstPlan, ())
        assert isinstance(secondStep, MoreViewsRequired)
        bootstrapViewId = secondStep.plan.views[0].viewId
        committed = controller.consume(
            secondStep.plan,
            (_boxedCandidate(bootstrapViewId, 20.0, 0.10),),
        )
        assert isinstance(committed, FrameCommitted)
        self.assertFalse(committed.result.valid)
        self.assertEqual(len(estimator.samples), 2)

        frame2 = FramePacket(
            SequenceId("motion-bootstrap"),
            FrameIndex(2),
            2_000_000_000,
            frame0.rgb,
        )
        controller.beginFrame(frame2)
        prediction = estimator.predictDetailed(frame2.timestampNs)
        motionScore = scoreMotionConsistency(
            BFoV(estimator.samples[-1].center, 0.35, 0.25),
            prediction.motionState,
        )

        self.assertEqual(prediction.sampleCount, 2)
        self.assertNotIn("insufficient_motion_samples", prediction.degradedReasons)
        self.assertGreater(prediction.reliability, 0.0)
        self.assertNotAlmostEqual(motionScore.effectiveProbability, 0.5)

    def testLostGlobalPlanContainsEquatorAndPolarCubeFaces(self) -> None:
        planner = RecoveryPlanner(
            self.config.geometry,
            self.config.tracking,
            self.config.recovery,
        )
        box = BBoxXYWH(150.0, 70.0, 40.0, 50.0)
        fallback = self.geometry.bboxToBfov(box, 360, 180)
        views = planner.buildViews(
            self.config.recovery.globalSearchInterval,
            360,
            180,
            box,
            box,
            fallback,
            None,
            TrackStatus.LOST,
        )
        self.assertEqual(len(views), 10)
        self.assertEqual(sum("cubemap" in item.role for item in views), 6)
        self.assertEqual(
            {item.role for item in views},
            {
                "round1_cubemap_front",
                "round1_cubemap_right",
                "round1_cubemap_back",
                "round1_cubemap_left",
                "round1_cubemap_up",
                "round1_cubemap_down",
                "round1_left_top",
                "round1_right_top",
                "round1_left_bottom",
                "round1_right_bottom",
            },
        )
        self.assertTrue(
            all(
                math.isclose(item.spec.bfov.horizontalFovRad, math.radians(120.0))
                and math.isclose(item.spec.bfov.verticalFovRad, math.radians(120.0))
                for item in views
            )
        )

    def testRefinementUsesClusterCentersAndLostUsesOneCombinedAttempt(self) -> None:
        planner = RecoveryPlanner(
            self.config.geometry,
            self.config.tracking,
            self.config.recovery,
        )
        box = BBoxXYWH(150.0, 70.0, 40.0, 50.0)
        fallback = self.geometry.bboxToBfov(box, 360, 180)
        first = planner.buildViews(
            1,
            360,
            180,
            box,
            box,
            fallback,
            None,
            TrackStatus.TRACKING,
            attemptIndex=0,
        )
        second = planner.buildViews(
            1,
            360,
            180,
            box,
            box,
            fallback,
            None,
            TrackStatus.TRACKING,
            attemptIndex=1,
            viewIdStart=4,
            searchSeedCenter=makeSphericalPoint(0.0, 0.0),
        )
        lost = planner.buildViews(
            1,
            360,
            180,
            box,
            box,
            fallback,
            None,
            TrackStatus.LOST,
            attemptIndex=0,
        )

        self.assertEqual((len(first), len(second), len(lost)), (4, 4, 10))
        self.assertEqual(
            tuple(item.spec.viewId for item in (*first, *second)),
            tuple(range(8)),
        )
        self.assertTrue(all(item.role.startswith("round1_") for item in lost))

    def testSourceConfidenceFloorLeavesLowSourcesAsSingles(self) -> None:
        frame0 = FramePacket(
            SequenceId("floor"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init)
        frame1 = FramePacket(SequenceId("floor"), FrameIndex(1), 1_000_000_000, frame0.rgb)
        first = controller.beginFrame(frame1)
        lowPair = (_boxedCandidate(0, 0.0, 0.70), _boxedCandidate(1, 0.0, 0.70))
        step = controller.consume(first, lowPair)

        self.assertIsInstance(step, MoreViewsRequired)
        assert isinstance(step, MoreViewsRequired)
        self.assertEqual(step.plan.attemptIndex, 1)
        final = controller.consume(
            step.plan,
            tuple(_boxedCandidate(view.viewId, 0.0, 0.70) for view in step.plan.views),
        )
        self.assertIsInstance(final, FrameCommitted)
        assert isinstance(final, FrameCommitted)
        self.assertTrue(final.result.valid)
        self.assertEqual(final.result.status, TrackStatus.TRACKING)
        assert controller.lastStateObservation is not None
        self.assertFalse(controller.lastStateObservation.selectedIsFused)
        self.assertTrue(controller.lastStateObservation.selectedSourceConfidencePassed)
        self.assertEqual(len(controller.lastStateObservation.sourceViewIds), 1)
        self.assertAlmostEqual(controller.lastStateObservation.stateScore, 0.70)

    def testOverlapThresholdSevenTenthsAllowsFirstRoundFusion(self) -> None:
        frame0 = FramePacket(
            SequenceId("overlap"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init)
        frame1 = FramePacket(SequenceId("overlap"), FrameIndex(1), 1_000_000_000, frame0.rgb)
        first = controller.beginFrame(frame1)
        pair = (_boxedCandidate(0, 0.0, 0.85), _boxedCandidate(1, 29.0, 0.85))
        result = controller.consume(first, pair)

        self.assertIsInstance(result, MoreViewsRequired)
        assert isinstance(result, MoreViewsRequired)
        final = controller.consume(
            result.plan,
            (_boxedCandidate(result.plan.views[0].viewId, 0.0, 0.85),),
        )
        self.assertIsInstance(final, FrameCommitted)
        assert isinstance(final, FrameCommitted)
        self.assertTrue(final.result.valid)
        assert controller.lastStateObservation is not None
        self.assertTrue(controller.lastStateObservation.selectedIsFused)
        self.assertEqual(controller.lastStateObservation.selectedOverlapRate, 1.0)

    def testUncertainSecondRoundFusesCandidatesFromBothRounds(self) -> None:
        frame0 = FramePacket(
            SequenceId("uncertain-cumulative"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init)

        for frameIndex in range(1, 4):
            frame = FramePacket(
                SequenceId("uncertain-cumulative"),
                FrameIndex(frameIndex),
                frameIndex * 1_000_000_000,
                frame0.rgb,
            )
            first = controller.beginFrame(frame)
            second = controller.consume(first, ())
            assert isinstance(second, MoreViewsRequired)
            committed = controller.consume(second.plan, ())
            assert isinstance(committed, FrameCommitted)
        self.assertEqual(controller.status, TrackStatus.UNCERTAIN)

        frame4 = FramePacket(
            SequenceId("uncertain-cumulative"),
            FrameIndex(4),
            4_000_000_000,
            frame0.rgb,
        )
        first = controller.beginFrame(frame4)
        self.assertEqual(len(first.views), 4)
        second = controller.consume(first, (_boxedCandidate(first.views[0].viewId, 20.0, 0.85),))
        assert isinstance(second, MoreViewsRequired)
        refinedViewId = second.plan.views[0].viewId
        final = controller.consume(
            second.plan,
            (_boxedCandidate(refinedViewId, 30.0, 0.85, widthPx=80.0),),
        )

        assert isinstance(final, FrameCommitted)
        self.assertTrue(final.result.valid)
        self.assertAlmostEqual(final.result.bbox.xPx, 20.0)
        self.assertAlmostEqual(final.result.bbox.widthPx, 100.0)
        assert controller.lastStateObservation is not None
        self.assertTrue(controller.lastStateObservation.selectedIsFused)
        self.assertEqual(
            controller.lastStateObservation.sourceViewIds,
            (first.views[0].viewId, refinedViewId),
        )
        self.assertEqual(controller.lastStateObservation.candidateCount, 2)

    def testTrackingFusionUsesBestSourceGeometry(self) -> None:
        frame0 = FramePacket(
            SequenceId("wide-fusion"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init)
        frame1 = FramePacket(
            SequenceId("wide-fusion"), FrameIndex(1), 1_000_000_000, frame0.rgb
        )
        plan = controller.beginFrame(frame1)

        result = controller.update(
            plan,
            (
                _boxedCandidate(0, 0.0, 0.99, widthPx=100.0),
                _boxedCandidate(1, 10.0, 0.99, widthPx=100.0),
            ),
        )

        self.assertAlmostEqual(result.bbox.xPx, 0.0)
        self.assertAlmostEqual(result.bbox.widthPx, 100.0)
        self.assertAlmostEqual(result.bfov.horizontalFovRad, 0.35)
        self.assertTrue(result.valid)

    def testTrackingFusionUsesBestSourceGeometryAcrossErpSeam(self) -> None:
        frame0 = FramePacket(
            SequenceId("seam-intersection"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init)
        frame1 = FramePacket(
            SequenceId("seam-intersection"),
            FrameIndex(1),
            1_000_000_000,
            frame0.rgb,
        )
        plan = controller.beginFrame(frame1)

        result = controller.update(
            plan,
            (
                _boxedCandidate(0, 340.0, 0.95, widthPx=40.0),
                _boxedCandidate(1, 350.0, 0.95, widthPx=30.0),
            ),
        )

        self.assertAlmostEqual(result.bbox.xPx, 340.0)
        self.assertAlmostEqual(result.bbox.widthPx, 40.0)
        self.assertTrue(result.valid)

    def testRuntimeStaysUncertainAndUsesTwoType1Rounds(self) -> None:
        frame0 = FramePacket(
            SequenceId("routes"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init)

        frame1 = FramePacket(SequenceId("routes"), FrameIndex(1), 1_000_000_000, frame0.rgb)
        first = controller.beginFrame(frame1)
        second = controller.consume(first, ())
        assert isinstance(second, MoreViewsRequired)
        uncertainResult = controller.consume(second.plan, ())
        assert isinstance(uncertainResult, FrameCommitted)
        self.assertEqual(uncertainResult.result.status, TrackStatus.TRACKING)

        frame2 = FramePacket(SequenceId("routes"), FrameIndex(2), 2_000_000_000, frame0.rgb)
        secondTrackingFirst = controller.beginFrame(frame2)
        self.assertEqual(len(secondTrackingFirst.views), 4)
        secondTrackingSecond = controller.consume(secondTrackingFirst, ())
        assert isinstance(secondTrackingSecond, MoreViewsRequired)
        secondTrackingResult = controller.consume(secondTrackingSecond.plan, ())
        assert isinstance(secondTrackingResult, FrameCommitted)
        self.assertEqual(secondTrackingResult.result.status, TrackStatus.TRACKING)

        frame3 = FramePacket(SequenceId("routes"), FrameIndex(3), 3_000_000_000, frame0.rgb)
        uncertainFirst = controller.beginFrame(frame3)
        self.assertEqual(len(uncertainFirst.views), 4)
        uncertainSecond = controller.consume(uncertainFirst, ())
        assert isinstance(uncertainSecond, MoreViewsRequired)
        uncertainResult = controller.consume(uncertainSecond.plan, ())
        assert isinstance(uncertainResult, FrameCommitted)
        self.assertEqual(uncertainResult.result.status, TrackStatus.UNCERTAIN)

        frame4 = FramePacket(SequenceId("routes"), FrameIndex(4), 4_000_000_000, frame0.rgb)
        uncertainFirst = controller.beginFrame(frame4)
        self.assertEqual(len(uncertainFirst.views), 4)
        self.assertTrue(
            all(
                math.isclose(view.bfov.horizontalFovRad, self.config.geometry.maxFovRad)
                and math.isclose(view.bfov.verticalFovRad, self.config.geometry.maxFovRad)
                for view in uncertainFirst.views
            )
        )
        uncertainSecond = controller.consume(uncertainFirst, ())
        assert isinstance(uncertainSecond, MoreViewsRequired)
        self.assertEqual(len(uncertainSecond.plan.views), 4)
        self.assertTrue(
            all(
                math.isclose(view.bfov.horizontalFovRad, self.config.geometry.maxFovRad)
                and math.isclose(view.bfov.verticalFovRad, self.config.geometry.maxFovRad)
                for view in uncertainSecond.plan.views
            )
        )
        uncertainFinal = controller.consume(uncertainSecond.plan, ())
        assert isinstance(uncertainFinal, FrameCommitted)
        self.assertEqual(uncertainFinal.result.status, TrackStatus.UNCERTAIN)

        frame5 = FramePacket(SequenceId("routes"), FrameIndex(5), 5_000_000_000, frame0.rgb)
        uncertainFirst = controller.beginFrame(frame5)
        self.assertEqual(len(uncertainFirst.views), 4)
        uncertainSecond = controller.consume(uncertainFirst, ())
        assert isinstance(uncertainSecond, MoreViewsRequired)
        self.assertEqual(len(uncertainSecond.plan.views), 4)
        uncertainFinal = controller.consume(uncertainSecond.plan, ())
        assert isinstance(uncertainFinal, FrameCommitted)
        self.assertEqual(uncertainFinal.result.status, TrackStatus.UNCERTAIN)

    def testExplicitLostComponentCanStillReacquire(self) -> None:
        frame0 = FramePacket(
            SequenceId("recover"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init)
        controller._mode = TrackMode.LOST

        frame1 = FramePacket(SequenceId("recover"), FrameIndex(1), 1_000_000_000, frame0.rgb)
        lostFirst = controller.beginFrame(frame1)
        self.assertEqual(len(lostFirst.views), 10)
        pending = controller.consume(
            lostFirst,
            tuple(
                _boxedCandidate(
                    view.viewId,
                    20.0 if index == 0 else 30.0 if index == 6 else 200.0,
                    0.96 if index == 6 else 0.95 if index == 0 else 0.10,
                    widthPx=80.0 if index == 6 else 100.0,
                )
                for index, view in enumerate(lostFirst.views)
            ),
        )
        assert isinstance(pending, FrameCommitted)
        self.assertEqual(pending.result.status, TrackStatus.TRACKING)
        self.assertTrue(pending.result.valid)
        self.assertEqual(pending.result.resultSource, ResultSource.OBSERVED_REACQUIRED)
        self.assertAlmostEqual(pending.result.bbox.xPx, 30.0)
        self.assertAlmostEqual(pending.result.bbox.widthPx, 80.0)

    def testRuntimeDoesNotEnterLostAndKeepsUsingUncertainPlan(self) -> None:
        frame0 = FramePacket(
            SequenceId("suppress-lost"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(self.geometry, self.config)
        init = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 40.0, 50.0))
        controller.commitInitialization(init)

        for frameIndex in range(1, 5):
            frame = FramePacket(
                SequenceId("suppress-lost"),
                FrameIndex(frameIndex),
                frameIndex * 1_000_000_000,
                frame0.rgb,
            )
            plan = controller.beginFrame(frame)
            while True:
                step = controller.consume(plan, ())
                if isinstance(step, MoreViewsRequired):
                    plan = step.plan
                    continue
                break
            self.assertNotEqual(step.result.status, TrackStatus.LOST)
        self.assertEqual(step.result.status, TrackStatus.UNCERTAIN)

        nextFrame = FramePacket(
            SequenceId("suppress-lost"),
            FrameIndex(5),
            5_000_000_000,
            frame0.rgb,
        )
        self.assertEqual(len(controller.beginFrame(nextFrame).views), 4)


if __name__ == "__main__":
    unittest.main()
