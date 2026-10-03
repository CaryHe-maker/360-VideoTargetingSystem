import math
import unittest
from pathlib import Path

from track360.controller import (
    FrameAggregate,
    FusionBoxMode,
    Fusor,
    RecoveryPlanner,
    TemplatePolicy,
    ViewSpecType1,
)
from track360.controller.state_model import ScoreGroup
from track360.core.config import loadConfig
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    ProjectedObservation,
    TemplateCommandKind,
    TrackStatus,
)
from track360.geometry import SphericalGeometryImpl, makeSphericalPoint

ROOT = Path(__file__).resolve().parents[2]


def _observation(
    viewId: int,
    yaw: float,
    score: float,
    x: float = 30.0,
    width: float = 100.0,
    height: float = 60.0,
) -> ProjectedObservation:
    return ProjectedObservation(
        viewId=viewId,
        bfov=BFoV(makeSphericalPoint(yaw, 0.0), 0.5, 0.4),
        bbox=BBoxXYWH(x, 60.0, width, height),
        modelScore=score,
        appearanceScore=score,
        motionScore=score,
        scaleScore=score,
        fusedScore=score,
        localBox=BBoxXYWH(70.0, 70.0, 40.0, 40.0),
        singleScore=score,
    )


class ControllerPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = loadConfig(ROOT / "configs" / "default.yaml")
        # Gated two-round behavior: every backendTuning switch at its "off" value.
        self.legacyConfig = loadConfig(ROOT / "configs" / "tests" / "legacy_off.yaml")
        self.geometry = SphericalGeometryImpl(
            boundarySamplesPerEdge=self.config.geometry.boundarySamplesPerEdge
        )

    def testViewSpecType1ClampsSmallPredictedViewsToThirtyDegrees(self) -> None:
        views = ViewSpecType1(
            makeSphericalPoint(0.0, 0.0),
            math.radians(5.0),
            math.radians(8.0),
        )

        self.assertEqual(len(views), 4)
        self.assertTrue(
            all(
                math.isclose(item.bfov.horizontalFovRad, math.radians(30.0))
                and math.isclose(item.bfov.verticalFovRad, math.radians(30.0))
                for item in views
            )
        )
        self.assertTrue(
            all(math.isclose(abs(item.bfov.center.yawRad), math.radians(10.0)) for item in views)
        )

    def testViewSpecType1UsesThreeTimesPredictedExtentOnEachAxis(self) -> None:
        views = ViewSpecType1(
            makeSphericalPoint(0.0, 0.0),
            math.radians(20.0),
            math.radians(15.0),
        )

        self.assertTrue(
            all(
                math.isclose(item.bfov.horizontalFovRad, math.radians(60.0))
                and math.isclose(item.bfov.verticalFovRad, math.radians(45.0))
                for item in views
            )
        )

    def testFusorReturnsTheBestFusedCandidate(self) -> None:
        result = Fusor(self.geometry).fuse(
            (
                _observation(0, 0.0, 0.90, 30.0),
                _observation(1, 0.01, 0.90, 40.0),
                _observation(2, 2.0, 0.79, 220.0),
            ),
            frameWidthPx=360,
            frameHeightPx=180,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.fused)
        self.assertEqual(result.sourceViewIds, (0, 1))
        self.assertGreater(result.confidence, 0.80)

    def testFusorCanChooseMinimumUnionBox(self) -> None:
        result = Fusor(self.geometry, boxMode=FusionBoxMode.MIN_UNION).fuse(
            (
                _observation(0, 0.0, 0.90, 30.0),
                _observation(1, 0.01, 0.90, 40.0),
            ),
            frameWidthPx=360,
            frameHeightPx=180,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.fused)
        self.assertAlmostEqual(result.bbox.xPx, 30.0)
        self.assertAlmostEqual(result.bbox.widthPx, 110.0)

    def testFusorBestSourceKeepsFusedConfidenceButUsesBestSourceShape(self) -> None:
        first = _observation(0, 0.0, 0.91, 30.0, width=60.0, height=40.0)
        second = _observation(1, 0.01, 0.90, 35.0, width=80.0, height=50.0)

        result = Fusor(self.geometry, boxMode=FusionBoxMode.BEST_SOURCE).fuse(
            (first, second),
            frameWidthPx=360,
            frameHeightPx=180,
        )

        assert result is not None
        self.assertTrue(result.fused)
        self.assertEqual(result.representativeViewId, first.viewId)
        self.assertEqual(result.bbox, first.bbox)
        self.assertGreater(result.confidence, first.singleScore or first.fusedScore)

    def testFusorReferenceAdaptiveBoxCoversAllSizeBranches(self) -> None:
        observations = (
            _observation(0, 0.0, 0.90, 20.0, width=100.0, height=80.0),
            _observation(1, 0.01, 0.90, 40.0, width=100.0, height=80.0),
        )
        fusor = Fusor(self.geometry, boxMode=FusionBoxMode.REFERENCE_ADAPTIVE)

        intersection = fusor.fuse(
            observations,
            frameWidthPx=360,
            frameHeightPx=180,
            referenceBoxAreaPx=4_800.0,
        )
        middle = fusor.fuse(
            observations,
            frameWidthPx=360,
            frameHeightPx=180,
            referenceBoxAreaPx=7_000.0,
        )
        union = fusor.fuse(
            observations,
            frameWidthPx=360,
            frameHeightPx=180,
            referenceBoxAreaPx=12_000.0,
        )

        assert intersection is not None
        assert middle is not None
        assert union is not None
        self.assertTrue(intersection.fused)
        self.assertAlmostEqual(intersection.bbox.xPx, 40.0)
        self.assertAlmostEqual(intersection.bbox.widthPx, 80.0)
        self.assertAlmostEqual(intersection.bbox.heightPx, 80.0)
        self.assertGreater(middle.bbox.widthPx, intersection.bbox.widthPx)
        self.assertLess(middle.bbox.widthPx, 120.0)
        self.assertAlmostEqual(middle.bbox.heightPx, 80.0)
        self.assertAlmostEqual(union.bbox.xPx, 20.0)
        self.assertAlmostEqual(union.bbox.widthPx, 120.0)
        self.assertAlmostEqual(union.bbox.heightPx, 80.0)

    def testFusorFullAgreementDoesNotSaturateConfidence(self) -> None:
        result = Fusor(self.geometry).fuse(
            (
                _observation(0, 0.0, 0.80),
                _observation(1, 0.0, 0.80),
            ),
            frameWidthPx=360,
            frameHeightPx=180,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.fused)
        self.assertAlmostEqual(result.confidence, 0.83)
        self.assertLess(result.confidence, 1.0)

    def testFusorScoresContainedBoxesWithIou(self) -> None:
        result = Fusor(self.geometry).fuse(
            (
                _observation(0, 0.0, 0.80, width=100.0),
                _observation(1, 0.0, 0.80, width=80.0),
            ),
            frameWidthPx=360,
            frameHeightPx=180,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.fused)
        self.assertAlmostEqual(result.overlapRate or 0.0, 1.0)
        self.assertAlmostEqual(result.confidence, 0.824)

    def testFusorDoesNotFuseBelowSourceConfidenceThreshold(self) -> None:
        result = Fusor(self.geometry).fuse(
            (
                _observation(0, 0.0, 0.90),
                _observation(1, 0.0, 0.79),
            ),
            frameWidthPx=360,
            frameHeightPx=180,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.fused)
        self.assertEqual(result.sourceViewIds, (0,))
        self.assertAlmostEqual(result.confidence, 0.90)

    def testFusorCapsFusionGainAtThreeHundredths(self) -> None:
        result = Fusor(self.geometry, sourceMinConfidence=0.0).fuse(
            (
                _observation(0, 0.0, 0.50),
                _observation(1, 0.0, 0.50),
            ),
            frameWidthPx=360,
            frameHeightPx=180,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.fused)
        self.assertAlmostEqual(result.confidence, 0.53)

    def testTemplatePolicyAlwaysKeepsFrameZeroAnchor(self) -> None:
        box = BBoxXYWH(10.0, 10.0, 20.0, 20.0)
        aggregate = FrameAggregate(
            bfov=BFoV(makeSphericalPoint(0.0, 0.0), 0.3, 0.3),
            bbox=box,
            confidence=0.99,
            decisionScore=0.99,
            sourceViewIds=(0, 1),
            representativeViewId=0,
            localBox=box,
            supported=True,
        )

        decision = TemplatePolicy(
            self.legacyConfig.tracking, self.legacyConfig.backendTuning
        ).decide(
            TrackStatus.TRACKING,
            self.legacyConfig.tracking.stableFramesBeforeUpdate * 2,
            aggregate,
        )

        self.assertEqual(decision.kind, TemplateCommandKind.KEEP)
        self.assertIsNone(decision.viewId)
        self.assertIsNone(decision.localBox)

    def testFusorUsesCircularIntersectionAtTheSeam(self) -> None:
        result = Fusor(self.geometry).fuse(
            (
                _observation(0, 0.0, 0.95, 350.0),
                _observation(1, 0.0, 0.95, 355.0),
            ),
            frameWidthPx=360,
            frameHeightPx=180,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.fused)
        self.assertGreater(result.overlapRate or 0.0, 0.70)
        self.assertGreater(result.bbox.widthPx, 0.0)

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
        thresholds = group.thresholds()
        self.assertIsNotNone(thresholds)
        assert thresholds is not None
        self.assertEqual(thresholds, (0.6, 0.3))
        self.assertEqual(len(group.values), 10)

    def testPlannerBuildsDynamicTrackingCornersAndTwelveViewLostRoute(self) -> None:
        planner = RecoveryPlanner(
            self.config.geometry,
            self.config.tracking,
            self.config.recovery,
        )
        box = BBoxXYWH(150.0, 70.0, 40.0, 50.0)
        fallback = self.geometry.bboxToBfov(box, 360, 180)
        primary = planner.buildViews(
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
        refined = planner.buildViews(
            1,
            360,
            180,
            box,
            box,
            fallback,
            None,
            TrackStatus.TRACKING,
            attemptIndex=1,
            searchSeedCenter=makeSphericalPoint(0.0, 0.0),
        )
        self.assertEqual((len(primary), len(refined)), (4, 4))
        expectedHorizontalFov = min(
            3.0 * fallback.horizontalFovRad,
            self.config.geometry.maxFovRad,
        )
        expectedVerticalFov = min(
            3.0 * fallback.verticalFovRad,
            self.config.geometry.maxFovRad,
        )
        self.assertTrue(
            all(
                math.isclose(item.spec.bfov.horizontalFovRad, expectedHorizontalFov)
                and math.isclose(item.spec.bfov.verticalFovRad, expectedVerticalFov)
                for item in (*primary, *refined)
            )
        )
        self.assertEqual(tuple(item.spec.viewId for item in refined), (0, 1, 2, 3))
        self.assertEqual(
            tuple(item.role for item in refined),
            (
                "round2_left_top",
                "round2_right_top",
                "round2_left_bottom",
                "round2_right_bottom",
            ),
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
        )
        self.assertEqual(len(lost), 10)
        self.assertTrue(
            all(
                item.spec.bfov.horizontalFovRad == self.config.geometry.maxFovRad
                for item in lost
            )
        )
        self.assertEqual(sum("cubemap" in item.role for item in lost), 6)
        self.assertEqual(sum("left_top" in item.role for item in lost), 1)


if __name__ == "__main__":
    unittest.main()
