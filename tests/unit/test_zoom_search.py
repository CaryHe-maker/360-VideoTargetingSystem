import math
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from track360.controller import TrackControllerImpl
from track360.controller.state_machine import RELATIVE_WARM_FRAMES, TrackStateMachine
from track360.controller.state_model import TrackMode, TransitionReason
from track360.controller.view_planner import REFINE_VIEW_ID_BASE, SCAN_VIEW_ID_BASE
from track360.core.config import loadConfig
from track360.core.errors import ConfigError
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    FramePacket,
    ProjectedObservation,
    SequenceId,
)
from track360.geometry import SphericalGeometryImpl, makeSphericalPoint

ROOT = Path(__file__).resolve().parents[2]
INITIAL_BOX = BBoxXYWH(170.0, 80.0, 20.0, 20.0)
# A doubt when the mean relative deviation is below -0.3, released after two frames
# back within -0.15; lost after three doubted frames.
RELATIVE = {
    "lossHandling": True,
    "relativeEnterDeviation": -0.3,
    "relativeGate": 0.05,
    "releaseFrames": 2,
    "lostAfterFrames": 3,
}


def _tuning(**overrides: object):
    config = loadConfig(ROOT / "configs" / "default.yaml")
    return replace(config.backendTuning, **{**RELATIVE, **overrides})


def _frame(index: int) -> FramePacket:
    return FramePacket(
        SequenceId("s"), FrameIndex(index), index * 1_000_000_000, np.zeros((180, 360, 3), np.uint8)
    )


def _observation(score: float, similarity: float, *, yaw: float = 0.0, viewId: int = 0):
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


class RelativeRuleTest(unittest.TestCase):
    def _run(self, machine: TrackStateMachine, frames):
        mode, decisions = TrackMode.TRACKING, []
        for backend, appearance in frames:
            decision = machine.transition(
                mode,
                0.5,
                measurementAccepted=True,
                backendScore=backend,
                appearanceScore=appearance,
            )
            mode = decision.nextMode
            decisions.append(decision)
        return decisions

    def _machine(self, level: tuple[float, float], **overrides: object) -> TrackStateMachine:
        machine = TrackStateMachine(_tuning(**overrides))
        machine.initialize()
        self._run(machine, [level] * RELATIVE_WARM_FRAMES)
        return machine

    def testTheSameDropIsJudgedAgainstTheSequencesOwnLevel(self) -> None:
        # Two sequences with very different normal scores: the same absolute values
        # are a loss for one and normal for the other.
        high = self._machine((0.9, 0.8))
        low = self._machine((0.5, 0.3))
        self.assertEqual(self._run(high, [(0.5, 0.3)])[0].nextMode, TrackMode.UNCERTAIN)
        self.assertAlmostEqual(high.relativeSignal, 0.5 * (0.5 / 0.9 - 1 + 0.3 / 0.8 - 1))
        self.assertEqual(self._run(low, [(0.5, 0.3)])[0].nextMode, TrackMode.TRACKING)
        self.assertAlmostEqual(low.relativeSignal, 0.0)

    def testADoubtIsLatchedAndBecomesLost(self) -> None:
        machine = self._machine((0.8, 0.6))
        decisions = self._run(
            machine,
            [
                (0.3, 0.2),  # far below: doubt
                (0.7, 0.5),  # above the threshold again, but not back within half of it
                (0.3, 0.2),
                (0.8, 0.6),  # back: first calm frame
                (0.8, 0.6),  # second calm frame: released
            ],
        )
        self.assertEqual(
            [decision.nextMode for decision in decisions],
            [
                TrackMode.UNCERTAIN,
                TrackMode.UNCERTAIN,
                TrackMode.LOST,
                TrackMode.LOST,
                TrackMode.TRACKING,
            ],
        )
        self.assertEqual(decisions[4].reason, TransitionReason.RELEASED)

    def testLowFramesDoNotPullTheBaselineDownAndItSurvivesAJump(self) -> None:
        machine = self._machine((0.8, 0.6))
        self._run(machine, [(0.3, 0.2)] * 30)
        machine.reset()
        # After many low frames and a reset the old level still is the reference.
        self._run(machine, [(0.3, 0.2)])
        self.assertAlmostEqual(machine.relativeSignal, 0.5 * (0.3 / 0.8 - 1 + 0.2 / 0.6 - 1))

    def testAFrameWithoutABoxIsDoubted(self) -> None:
        machine = self._machine((0.8, 0.6))
        decision = machine.transition(TrackMode.TRACKING, 0.0, measurementAccepted=False)
        self.assertEqual(decision.nextMode, TrackMode.UNCERTAIN)
        self.assertEqual(decision.reason, TransitionReason.HARD_MISS)

    def testSettingsAreValidated(self) -> None:
        with self.assertRaises(ConfigError):
            _tuning(relativeEnterDeviation=0.2)
        with self.assertRaises(ConfigError):
            _tuning(lossActions="other")
        with self.assertRaises(ConfigError):
            _tuning(zoomFirstScale=0.5)

    def testWithoutLossHandlingTheStateFollowsTheFusedScore(self) -> None:
        machine = TrackStateMachine(_tuning(lossHandling=False, uncertainScore=0.42))
        machine.initialize()
        # Scores that the relative rule would call a deep drop are not looked at.
        self._run(machine, [(0.9, 0.8)] * RELATIVE_WARM_FRAMES)
        decision = machine.transition(
            TrackMode.TRACKING, 0.5, measurementAccepted=True, backendScore=0.1,
            appearanceScore=0.1,
        )
        self.assertEqual(decision.nextMode, TrackMode.TRACKING)
        low = machine.transition(TrackMode.TRACKING, 0.3, measurementAccepted=True)
        self.assertEqual(low.nextMode, TrackMode.UNCERTAIN)
        # No latch: one frame above the threshold is trusted again.
        back = machine.transition(TrackMode.UNCERTAIN, 0.5, measurementAccepted=True)
        self.assertEqual(back.nextMode, TrackMode.TRACKING)


class ZoomSearchTest(unittest.TestCase):
    def _controller(self, **overrides: object) -> TrackControllerImpl:
        config = loadConfig(ROOT / "configs" / "default.yaml")
        settings = {
            **RELATIVE,
            "zoomFirstScale": 2.0,
            "zoomLastScale": 4.0,
            "zoomLastAfterFrames": 3,
            "scanBudgetPerFrame": 0.5,
            "scanBudgetBurst": 5.0,
            **overrides,
        }
        config = replace(config, backendTuning=replace(config.backendTuning, **settings))
        controller = TrackControllerImpl(SphericalGeometryImpl(), config)
        controller.commitInitialization(controller.buildInitialization(_frame(0), INITIAL_BOX))
        return controller

    def _lose(self, controller: TrackControllerImpl) -> int:
        """Track well, then badly until the state is LOST; the next frame index."""
        index = 1
        for _ in range(RELATIVE_WARM_FRAMES):
            plan = controller.beginFrame(_frame(index))
            controller.consume(plan, _observation(0.9, 0.8))
            index += 1
        for _ in range(3):
            plan = controller.beginFrame(_frame(index))
            self.assertEqual(plan.scanViews, ())
            controller.consume(plan, _observation(0.2, 0.1, yaw=0.3))
            index += 1
        return index

    def testTheFirstLostFrameLooksAgainInPlaceThenTheViewGrows(self) -> None:
        controller = self._controller()
        index = self._lose(controller)
        plans = []
        for _ in range(8):
            plan = controller.beginFrame(_frame(index))
            plans.append(plan)
            controller.consume(plan, _observation(0.2, 0.1, yaw=0.3))
            index += 1

        first = plans[0]
        # In place: the tracker's own view, without its trajectory, and no second view.
        self.assertEqual(len(first.scanViews), 1)
        self.assertFalse(first.scanRefine)
        self.assertEqual(first.scanViews[0].bfov, first.view.bfov)
        self.assertEqual(first.scanViews[0].trajectory, ())
        self.assertEqual(first.scanViews[0].viewId, SCAN_VIEW_ID_BASE)
        self.assertEqual(controller.lossStatistics["scanFrames"] > 0, True)
        # Then an enlarged view around where the target was last trusted (yaw 0), to
        # be refined; it is twice the normal view at first and four times later.
        zoomed = [plan for plan in plans[1:] if plan.scanViews]
        self.assertTrue(all(plan.scanRefine for plan in zoomed))
        self.assertAlmostEqual(zoomed[0].scanViews[0].bfov.center.yawRad, 0.0, places=6)
        normal = first.view.bfov.horizontalFovRad
        self.assertGreater(zoomed[0].scanViews[0].bfov.horizontalFovRad, 1.5 * normal)
        self.assertGreater(
            zoomed[-1].scanViews[0].bfov.horizontalFovRad,
            zoomed[0].scanViews[0].bfov.horizontalFovRad,
        )
        # The budget: one pass in place and two per zoom, earned at half a pass a frame.
        spent = sum(2 if plan.scanRefine else len(plan.scanViews) for plan in plans)
        self.assertLessEqual(spent, 5 + 0.5 * (index - 1))
        self.assertEqual([bool(plan.scanViews) for plan in plans[:4]], [True, True, True, False])

    def testTheRefinementViewIsOfTheNormalSizeAndACandidateWithAHighScoreIsTaken(self) -> None:
        controller = self._controller()
        index = self._lose(controller)
        plan = controller.beginFrame(_frame(index))
        fine = controller.refinementView(makeSphericalPoint(1.0, 0.0))
        self.assertEqual(fine.viewId, REFINE_VIEW_ID_BASE)
        self.assertAlmostEqual(fine.bfov.center.yawRad, 1.0, places=6)
        normal = plan.view.bfov.horizontalFovRad
        self.assertAlmostEqual(fine.bfov.horizontalFovRad, normal, places=6)
        # Only the stateless score decides: a low similarity does not stop the jump.
        weak = _observation(0.5, 0.9, yaw=1.0, viewId=REFINE_VIEW_ID_BASE)
        controller.consume(plan, _observation(0.2, 0.1, yaw=0.3), (weak,))
        self.assertFalse(controller.lastFrameReacquired)
        plan = controller.beginFrame(_frame(index + 1))
        strong = _observation(0.8, 0.05, yaw=1.0, viewId=REFINE_VIEW_ID_BASE)
        result = controller.consume(plan, _observation(0.2, 0.1, yaw=0.3), (strong,))
        self.assertTrue(controller.lastFrameReacquired)
        self.assertAlmostEqual(result.bfov.center.yawRad, 1.0)

    def testACandidateOnTheTrackedBoxIsTakenToo(self) -> None:
        controller = self._controller()
        index = self._lose(controller)
        plan = controller.beginFrame(_frame(index))
        # The look in place found the box the tracker already has, with a high score:
        # the jump moves nothing but restarts the tracker's memory.
        same = _observation(0.9, 0.5, yaw=0.3, viewId=SCAN_VIEW_ID_BASE)
        result = controller.consume(plan, _observation(0.2, 0.1, yaw=0.3), (same,))
        self.assertTrue(controller.lastFrameReacquired)
        self.assertEqual(controller.lastFrameTrace["candidates"][0]["verdict"], "accepted")
        self.assertEqual(controller.lastFrameTrace["action"], "jump")
        self.assertEqual(result.status.name, "TRACKING")
        self.assertAlmostEqual(result.bfov.center.yawRad, 0.3)

    def testWithoutActionsTheStateIsJudgedButNothingIsSearched(self) -> None:
        controller = self._controller(lossActions="none")
        index = self._lose(controller)
        plan = controller.beginFrame(_frame(index))
        self.assertEqual(plan.scanViews, ())
        controller.consume(plan, _observation(0.2, 0.1, yaw=0.3))
        self.assertEqual(controller.lastFrameTrace["modeAfter"], "LOST")
        # The tracker goes on learning: the run stays what it is without loss handling.
        self.assertFalse(controller.lastFrameSuspect)


if __name__ == "__main__":
    unittest.main()
