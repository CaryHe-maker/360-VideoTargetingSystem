import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from track360.backends import ARTrackBackend, ARTrackPrediction, TrackerBackendImpl
from track360.backends.appearance import histogramSimilarity, targetCrop
from track360.backends.artrack_model import ARTrackTemplate
from track360.controller import TrackControllerImpl, ViewPlanner
from track360.core.config import loadConfig
from track360.core.errors import ConfigError
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    FramePacket,
    LocalView,
    ProjectedObservation,
    SequenceId,
    TemplateCommand,
    TemplateCommandKind,
    ViewProjection,
    ViewSpec,
)
from track360.geometry import SphericalGeometryImpl, makeSphericalPoint

ROOT = Path(__file__).resolve().parents[2]
INITIAL_BOX = BBoxXYWH(170.0, 80.0, 20.0, 20.0)


def _frame(index: int) -> FramePacket:
    return FramePacket(
        SequenceId("s"), FrameIndex(index), index * 1_000_000_000, np.zeros((180, 360, 3), np.uint8)
    )


def _observation(
    score: float, similarity: float | None, *, yaw: float = 0.0, viewId: int = 0
) -> ProjectedObservation:
    return ProjectedObservation(
        viewId=viewId,
        bfov=BFoV(makeSphericalPoint(yaw, 0.0), 0.35, 0.35),
        bbox=BBoxXYWH(170.0 + math.degrees(yaw), 80.0, 20.0, 20.0),
        modelScore=score,
        appearanceScore=score,
        motionScore=score,
        scaleScore=1.0,
        fusedScore=score,
        localBox=BBoxXYWH(100.0, 100.0, 60.0, 60.0),
        singleScore=score,
        appearanceSimilarity=similarity,
    )


class LossHandlingControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        config = loadConfig(ROOT / "configs" / "default.yaml")
        # State score = mean of the backend score and the appearance similarity; a
        # frame below 0.6 is not trusted and three of them in a row mean lost.
        self.config = replace(
            config,
            backendTuning=replace(
                config.backendTuning,
                lossHandling=True,
                stateBackendWeight=0.5,
                stateAppearanceWeight=0.5,
                stateMotionWeight=0.0,
                uncertainScore=0.6,
                lostAfterFrames=4,
            ),
        )
        self.controller = TrackControllerImpl(SphericalGeometryImpl(), self.config)
        self.controller.commitInitialization(
            self.controller.buildInitialization(_frame(0), INITIAL_BOX)
        )

    def _step(self, index: int, observation, candidates=()):
        plan = self.controller.beginFrame(_frame(index))
        return plan, self.controller.consume(plan, observation, candidates)

    def testALowScoreOrAnUnlikeBoxIsSuspectAndAGoodFrameClearsIt(self) -> None:
        self._step(1, _observation(0.9, 0.7))
        self.assertFalse(self.controller.lastFrameSuspect)
        self._step(2, _observation(0.9, 0.1))
        self.assertTrue(self.controller.lastFrameSuspect)
        self._step(3, _observation(0.2, 0.7))
        self.assertEqual(self.controller.suspectFrames, 2)
        # The box is still followed: doubt alone does not stop the tracker.
        _, result = self._step(4, _observation(0.9, 0.1, yaw=0.05))
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.bfov.center.yawRad, 0.05)
        self._step(5, _observation(0.9, 0.7))
        self.assertEqual(self.controller.suspectFrames, 0)
        self.assertFalse(self.controller.lastFrameSuspect)

    def testScanViewsStartAfterEnoughDoubtfulFramesAroundTheLastTrustedPlace(self) -> None:
        self._step(1, _observation(0.9, 0.7))
        plans = []
        for index in range(2, 8):
            plan, _ = self._step(index, _observation(0.9, 0.1, yaw=0.1 * index))
            plans.append(plan)

        # Four doubtful frames must have passed before a plan carries scan views.
        self.assertEqual([len(plan.scanViews) for plan in plans], [0, 0, 0, 0, 4, 4])
        first = plans[4].scanViews[0]
        # The nearest scan view looks at where the target was last trusted (yaw 0), not at
        # where the doubted box wandered to.
        self.assertLess(abs(first.bfov.center.yawRad), 0.5)
        self.assertIsNotNone(first.priorBox)
        self.assertEqual([view.viewId for view in plans[4].scanViews], [1, 2, 3, 4])
        self.assertNotEqual(plans[4].scanViews, plans[5].scanViews)

    def testAClearlyBetterCandidateRestartsTheTrackThere(self) -> None:
        for index in range(1, 6):
            self._step(index, _observation(0.9, 0.1))
        weak = _observation(0.9, 0.2, yaw=1.0, viewId=1)
        unsure = _observation(0.3, 0.9, yaw=1.5, viewId=2)
        strong = _observation(0.8, 0.8, yaw=2.0, viewId=3)

        _, kept = self._step(6, _observation(0.9, 0.1), (weak, unsure))
        self.assertFalse(self.controller.lastFrameReacquired)
        self.assertAlmostEqual(kept.bfov.center.yawRad, 0.0)

        _, jumped = self._step(7, _observation(0.9, 0.1), (weak, unsure, strong))
        self.assertTrue(self.controller.lastFrameReacquired)
        self.assertFalse(self.controller.lastFrameSuspect)
        self.assertAlmostEqual(jumped.bfov.center.yawRad, 2.0)
        self.assertEqual(self.controller.suspectFrames, 0)
        after = self.controller.beginFrame(_frame(8))
        self.assertEqual(after.scanViews, ())
        self.assertAlmostEqual(after.view.bfov.center.yawRad, 2.0, places=3)
        # The trajectory starts over at the new place.
        self.assertEqual(len(set(after.view.trajectory)), 1)

    def testACandidateMustBeatTheTrackedBoxByAMargin(self) -> None:
        for index in range(1, 6):
            self._step(index, _observation(0.4, 0.55))
        self._step(6, _observation(0.4, 0.55), (_observation(0.8, 0.6, yaw=2.0, viewId=1),))
        self.assertFalse(self.controller.lastFrameReacquired)

    def testLossHandlingIsOffByDefaultAndItsSettingsAreValidated(self) -> None:
        config = loadConfig(ROOT / "configs" / "default.yaml")
        self.assertFalse(config.backendTuning.lossHandling)
        controller = TrackControllerImpl(SphericalGeometryImpl(), config)
        controller.commitInitialization(controller.buildInitialization(_frame(0), INITIAL_BOX))
        for index in range(1, 8):
            plan = controller.beginFrame(_frame(index))
            self.assertEqual(plan.scanViews, ())
            controller.consume(plan, _observation(0.1, 0.0))
            self.assertFalse(controller.lastFrameSuspect)
        with self.assertRaises(ConfigError):
            replace(config.backendTuning, lostAfterFrames=0)
        with self.assertRaises(ConfigError):
            replace(config.backendTuning, uncertainScore=2.0)


class ScanViewsTest(unittest.TestCase):
    def setUp(self) -> None:
        config = loadConfig(ROOT / "configs" / "default.yaml")
        self.planner = ViewPlanner(config.geometry, config.tracking, config.backendTuning)

    def testScanCoversTheSphereNearestFirstAndWrapsAround(self) -> None:
        lastSeen = BFoV(makeSphericalPoint(1.0, 0.3), math.radians(20.0), math.radians(20.0))
        views, cursor = self.planner.scanViews(lastSeen, 0, 6)
        self.assertEqual(cursor, 6)
        forward = np.asarray((lastSeen.center.x, lastSeen.center.y, lastSeen.center.z))
        centers = np.asarray([(v.bfov.center.x, v.bfov.center.y, v.bfov.center.z) for v in views])
        distances = np.round(np.arccos(np.clip(centers @ forward, -1.0, 1.0)), 9).tolist()
        self.assertEqual(distances, sorted(distances))
        # An 80 degree view every 40 degrees: the nearest is within half a step.
        self.assertLess(distances[0], math.radians(21.0))
        self.assertAlmostEqual(math.degrees(views[0].bfov.horizontalFovRad), 70.4, delta=1.0)

        total, seen, cursor = 0, [], 0
        while True:
            batch, cursor = self.planner.scanViews(lastSeen, cursor, 7)
            seen += [view.bfov.center for view in batch]
            total += len(batch)
            if cursor < 7 and total > 7:
                break
        directions = np.asarray([(c.x, c.y, c.z) for c in seen])
        # Every direction on the sphere is within one step of some scan view.
        rng = np.random.default_rng(0)
        probes = rng.normal(size=(500, 3))
        probes /= np.linalg.norm(probes, axis=1, keepdims=True)
        nearest = np.arccos(np.clip(probes @ directions.T, -1.0, 1.0)).min(axis=1)
        self.assertLess(float(nearest.max()), math.radians(40.0))

    def testLargeTargetsAreScannedWithSphericalViews(self) -> None:
        lastSeen = BFoV(makeSphericalPoint(0.0, 0.0), math.radians(60.0), math.radians(60.0))
        views, _ = self.planner.scanViews(lastSeen, 0, 3)
        self.assertTrue(all(view.projection is ViewProjection.SPHERICAL for view in views))
        self.assertEqual(self.planner.scanViews(lastSeen, 0, 0), ((), 0))


class _StatefulSession:
    """A session stand-in whose template carries a per-sequence memory."""

    supportsOnlineTemplates = False
    trajectoryLength = 7

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def encodeTemplate(self, rgb, bbox):
        return ARTrackTemplate(rgb, bbox)

    def infer(self, rgb, templateFeatures):
        raise AssertionError("not used")

    def inferBatch(
        self, rgbs, templateFeatures, *, imageFovs=None, priorBoxes=None, trajectories=None
    ):
        template = templateFeatures[0]
        self.calls.append(
            {"memory": dict(template.memory), "trajectory": trajectories is not None}
        )
        template.memory["appearance"] = template.memory.get("appearance", 0) + 1
        inside, outside = BBoxXYWH(8.0, 9.0, 20.0, 18.0), BBoxXYWH(-50.0, 0.0, 5.0, 5.0)
        box = inside if rgbs[0][0, 0, 0] == 0 else outside
        return tuple(ARTrackPrediction(box, 0.8, 0.8, 0.8) for _ in rgbs)

    def close(self):
        return None


class BackendStateTest(unittest.TestCase):
    def setUp(self) -> None:
        point = makeSphericalPoint(0.0, 0.0)
        self.spec = ViewSpec(0, BFoV(point, 1.0, 1.0), 64, 64, priorBox=BBoxXYWH(24, 24, 16, 16))
        self.view = LocalView(self.spec, np.zeros((64, 64, 3), dtype=np.uint8))
        self.session = _StatefulSession()
        self.backend = TrackerBackendImpl(ARTrackBackend(self.session))
        self.backend.initialize(self.view, BBoxXYWH(20.0, 20.0, 16.0, 16.0))

    def _infer(self, revision: int):
        return self.backend.infer(
            (self.view,), TemplateCommand(TemplateCommandKind.KEEP, revision, None, None, revision)
        )

    def testAFrameCanBeUndoneAndTheTrackerCanStartOver(self) -> None:
        self._infer(1)
        saved = self.backend.saveState()
        self._infer(2)
        self.backend.restoreState(saved)
        self._infer(3)
        self.backend.resetState()
        self._infer(4)

        self.assertEqual(
            [call["memory"] for call in self.session.calls],
            [{}, {"appearance": 1}, {"appearance": 1}, {}],
        )

    def testDetachedInferenceSeesOnlyTheTemplateAndLeavesTheTrackerAlone(self) -> None:
        self._infer(1)
        outside = LocalView(
            replace(self.spec, viewId=2), np.full((64, 64, 3), 255, dtype=np.uint8)
        )
        scan = LocalView(replace(self.spec, viewId=1), self.view.rgb)

        found = self.backend.inferDetached((scan, outside, scan))
        self._infer(2)

        # The view whose box left the image yields no observation.
        self.assertEqual([item.viewId for item in found], [1, 1])
        detached = self.session.calls[1:4]
        self.assertTrue(all(call["memory"] == {} for call in detached))
        self.assertFalse(any(call["trajectory"] for call in detached))
        self.assertEqual(self.session.calls[4]["memory"], {"appearance": 1})


class AppearanceCropTest(unittest.TestCase):
    def testCropIsSquareKeepsAspectAndPadsOutsideTheImage(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[40:60, 80:120] = 255
        crop = targetCrop(image, BBoxXYWH(80.0, 40.0, 40.0, 20.0), size=44, margin=1.1)

        self.assertEqual(crop.shape, (44, 44, 3))
        # The 40 x 20 box fills the width and half the height of the 44 px crop.
        self.assertGreater(crop[22, 4:40].mean(), 250)
        self.assertLess(crop[4, :].mean(), 5)
        corner = targetCrop(image, BBoxXYWH(-30.0, -30.0, 40.0, 40.0), size=32)
        self.assertEqual(corner.shape, (32, 32, 3))
        self.assertEqual(int(corner.max()), 0)

    def testHistogramSimilarityIsOneForTheSameImageAndLowerForAnother(self) -> None:
        red = np.zeros((32, 32, 3), dtype=np.uint8)
        red[..., 0] = 200
        blue = np.zeros((32, 32, 3), dtype=np.uint8)
        blue[..., 2] = 200
        self.assertAlmostEqual(histogramSimilarity(red, red), 1.0, places=5)
        self.assertLess(histogramSimilarity(red, blue), 0.2)


if __name__ == "__main__":
    unittest.main()
