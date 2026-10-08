import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from track360.controller import TrackControllerImpl
from track360.controller.state_machine import TrackStateMachine
from track360.controller.state_model import TrackMode, TransitionReason
from track360.core.config import loadConfig
from track360.core.errors import ConfigError
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    FramePacket,
    ProjectedObservation,
    SequenceId,
    TrackStatus,
)
from track360.geometry import SphericalGeometryImpl, makeSphericalPoint
from track360.runtime.benchmark import TRACE_COLUMNS, writeStateTrace

ROOT = Path(__file__).resolve().parents[2]
INITIAL_BOX = BBoxXYWH(170.0, 80.0, 20.0, 20.0)
# Enter a doubt below 0.5 (backend), 0.1 (motion) or 0.3 (appearance, mean of five
# frames); leave it at 0.7 / 0.5 for two frames; lost after three doubted frames whose
# mean appearance is below 0.4; a jump is on probation for three frames.
SPLIT = {
    "lossHandling": True,
    "stateRule": "split",
    "backendEnterScore": 0.5,
    "motionEnterScore": 0.1,
    "appearanceEnterScore": 0.3,
    "backendReleaseScore": 0.7,
    "appearanceReleaseScore": 0.5,
    "appearanceLostScore": 0.4,
    "releaseFrames": 2,
    "lostAfterFrames": 3,
    "probationFrames": 3,
}


def _tuning(**overrides: object):
    config = loadConfig(ROOT / "configs" / "default.yaml")
    return replace(config.backendTuning, **{**SPLIT, **overrides})


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


class SplitStateMachineTest(unittest.TestCase):
    def _run(self, frames, **overrides: object):
        machine = TrackStateMachine(_tuning(**overrides))
        machine.initialize()
        mode, decisions = TrackMode.TRACKING, []
        for backend, appearance, motion in frames:
            decision = machine.transition(
                mode,
                0.5,
                measurementAccepted=True,
                backendScore=backend,
                appearanceScore=appearance,
                motionScore=motion,
            )
            mode = decision.nextMode
            decisions.append(decision)
        return decisions

    def testEachScoreRaisesADoubtForItsOwnReason(self) -> None:
        good = (0.9, 0.8, 1.0)
        self.assertEqual(self._run([good])[0].nextMode, TrackMode.TRACKING)
        self.assertEqual(self._run([(0.4, 0.8, 1.0)])[0].reason, TransitionReason.BACKEND_LOW)
        self.assertEqual(self._run([(0.9, 0.8, 0.05)])[0].reason, TransitionReason.MOTION_LOW)
        # One unlike frame is not enough: the mean of five has to be low.
        calm = (0.9, 0.7, 1.0)
        slow = self._run([calm, calm, (0.9, 0.0, 1.0), (0.9, 0.0, 1.0), (0.9, 0.0, 1.0)])
        self.assertEqual(
            [decision.nextMode for decision in slow[:4]], [TrackMode.TRACKING] * 4
        )
        self.assertEqual(slow[4].reason, TransitionReason.APPEARANCE_LOW)
        self.assertEqual(slow[4].nextMode, TrackMode.UNCERTAIN)

    def testADoubtHoldsUntilBothScoresAreBackForAWhile(self) -> None:
        decisions = self._run(
            [
                (0.4, 0.8, 1.0),  # doubt
                (0.9, 0.8, 1.0),  # back, first calm frame
                (0.6, 0.8, 1.0),  # above the entry score but below the release score
                (0.9, 0.8, 1.0),
                (0.9, 0.8, 1.0),  # second calm frame in a row
                (0.9, 0.8, 1.0),
            ]
        )
        self.assertEqual(
            [decision.nextMode for decision in decisions],
            [TrackMode.UNCERTAIN] * 4 + [TrackMode.TRACKING] * 2,
        )
        self.assertEqual(decisions[2].reason, TransitionReason.DOUBT_HELD)
        self.assertEqual(decisions[4].reason, TransitionReason.RELEASED)

    def testOnlyAnUnlikeBoxTurnsADoubtIntoLost(self) -> None:
        # The backend score stays low but the box looks like the target: not lost.
        alike = self._run([(0.4, 0.8, 1.0)] * 6)
        self.assertEqual({decision.nextMode for decision in alike}, {TrackMode.UNCERTAIN})
        unlike = self._run([(0.4, 0.8, 1.0), (0.4, 0.2, 1.0), (0.4, 0.1, 1.0), (0.4, 0.1, 1.0)])
        self.assertEqual(
            [decision.nextMode for decision in unlike],
            [TrackMode.UNCERTAIN, TrackMode.UNCERTAIN, TrackMode.LOST, TrackMode.LOST],
        )
        self.assertEqual(unlike[2].reason, TransitionReason.APPEARANCE_CONFIRMED_LOSS)

    def testProbationPassesOrFailsOnTheMedianOfItsFrames(self) -> None:
        def probation(frames):
            machine = TrackStateMachine(_tuning())
            machine.initialize()
            machine.startProbation()
            mode, decisions = TrackMode.PROBATION, []
            for backend, appearance in frames:
                decision = machine.transition(
                    mode,
                    0.5,
                    measurementAccepted=True,
                    backendScore=backend,
                    appearanceScore=appearance,
                    motionScore=1.0,
                )
                mode = decision.nextMode
                decisions.append(decision)
            return decisions

        passed = probation([(0.9, 0.8), (0.9, 0.1), (0.9, 0.8)])
        self.assertEqual(passed[1].nextMode, TrackMode.PROBATION)
        self.assertEqual(passed[2].reason, TransitionReason.PROBATION_PASSED)
        self.assertEqual(passed[2].nextMode, TrackMode.TRACKING)
        failed = probation([(0.9, 0.2), (0.9, 0.8), (0.9, 0.2)])
        self.assertEqual(failed[2].reason, TransitionReason.PROBATION_FAILED)
        self.assertEqual(failed[2].nextMode, TrackMode.LOST)

    def testTheFusedRuleWithALatchHoldsADoubt(self) -> None:
        tuning = _tuning(
            stateRule="fused",
            stateLatch=True,
            uncertainScore=0.5,
            latchReleaseMargin=0.2,
            releaseFrames=2,
            lostAfterFrames=3,
        )
        machine = TrackStateMachine(tuning)
        machine.initialize()
        mode, modes = TrackMode.TRACKING, []
        for score in (0.8, 0.4, 0.6, 0.75, 0.6, 0.75, 0.75, 0.55):
            mode = machine.transition(mode, score, measurementAccepted=True).nextMode
            modes.append(mode)
        # 0.6 is above the threshold but not above the release level of 0.7.
        self.assertEqual(
            modes,
            [
                TrackMode.TRACKING,
                TrackMode.UNCERTAIN,
                TrackMode.UNCERTAIN,
                TrackMode.LOST,
                TrackMode.LOST,
                TrackMode.LOST,
                TrackMode.TRACKING,
                TrackMode.TRACKING,
            ],
        )

    def testSettingsAreValidated(self) -> None:
        with self.assertRaises(ConfigError):
            _tuning(stateRule="other")
        with self.assertRaises(ConfigError):
            _tuning(lossHandling=False)
        with self.assertRaises(ConfigError):
            _tuning(stateRule="fused", lossActions="probation")
        with self.assertRaises(ConfigError):
            _tuning(probationFrames=0)


class SplitControllerTest(unittest.TestCase):
    def _controller(self, **overrides: object) -> TrackControllerImpl:
        config = loadConfig(ROOT / "configs" / "default.yaml")
        config = replace(
            config,
            backendTuning=replace(config.backendTuning, **{**SPLIT, **overrides}),
        )
        controller = TrackControllerImpl(SphericalGeometryImpl(), config)
        controller.commitInitialization(controller.buildInitialization(_frame(0), INITIAL_BOX))
        return controller

    @staticmethod
    def _step(controller, index: int, observation, candidates=()):
        plan = controller.beginFrame(_frame(index))
        return plan, controller.consume(plan, observation, candidates)

    def _lose(self, controller, last: int) -> None:
        for index in range(1, last + 1):
            self._step(controller, index, _observation(0.3, 0.1))

    def testWithoutActionsTheStateIsOnlyJudgedAndTraced(self) -> None:
        controller = self._controller(lossActions="none")
        self._step(controller, 1, _observation(0.9, 0.8))
        plans = []
        for index in range(2, 8):
            plan, result = self._step(controller, index, _observation(0.3, 0.1))
            plans.append(plan)
            # The tracker is never told to forget a frame and nothing is scanned.
            self.assertFalse(controller.lastFrameSuspect)
        self.assertEqual({len(plan.scanViews) for plan in plans}, {0})
        self.assertEqual(result.status, TrackStatus.LOST)
        trace = controller.lastFrameTrace
        self.assertEqual(trace["modeBefore"], "LOST")
        self.assertEqual(trace["reason"], "DOUBT_HELD")
        self.assertAlmostEqual(trace["backend"], 0.3)
        self.assertAlmostEqual(trace["appearance"], 0.1)

    def testAJumpOnProbationIsConfirmedWhenTheNewBoxKeepsLookingRight(self) -> None:
        controller = self._controller(lossActions="probation")
        self._lose(controller, 4)
        candidate = _observation(0.8, 0.8, yaw=2.0, viewId=1)
        plan, jumped = self._step(controller, 5, _observation(0.3, 0.1), (candidate,))
        self.assertEqual(len(plan.scanViews), 4)
        self.assertTrue(controller.lastFrameReacquired)
        self.assertEqual(jumped.status, TrackStatus.UNCERTAIN)
        self.assertEqual(controller.lastFrameTrace["action"], "jump")
        self.assertEqual(controller.lastFrameTrace["modeAfter"], "PROBATION")
        self.assertEqual(controller.lastFrameTrace["candidates"][0]["verdict"], "accepted")
        for index in (6, 7):
            plan, result = self._step(controller, index, _observation(0.9, 0.8, yaw=2.0))
            # On probation the tracker learns normally and nothing is scanned.
            self.assertEqual(plan.scanViews, ())
            self.assertFalse(controller.lastFrameSuspect)
            self.assertEqual(result.status, TrackStatus.UNCERTAIN)
        _, confirmed = self._step(controller, 8, _observation(0.9, 0.8, yaw=2.0))
        self.assertEqual(confirmed.status, TrackStatus.TRACKING)
        self.assertEqual(controller.lastFrameTrace["action"], "confirm")
        self.assertFalse(controller.lastFrameReverted)

    def testAFailedProbationReturnsToBeforeTheJumpAndRemembersTheDistractor(self) -> None:
        controller = self._controller(lossActions="probation")
        self._lose(controller, 4)
        candidate = _observation(0.8, 0.8, yaw=2.0, viewId=1)
        self._step(controller, 5, _observation(0.3, 0.1), (candidate,))
        for index in (6, 7):
            self._step(controller, index, _observation(0.9, 0.1, yaw=2.0))
        _, failed = self._step(controller, 8, _observation(0.9, 0.1, yaw=2.0))
        self.assertTrue(controller.lastFrameReverted)
        self.assertEqual(failed.status, TrackStatus.LOST)
        self.assertEqual(controller.lastFrameTrace["action"], "revert")
        self.assertEqual(controller.lossStatistics["revertedAt"], [8])
        # The next view is back where the track was before the jump, and scanning resumes.
        plan = controller.beginFrame(_frame(9))
        self.assertAlmostEqual(plan.view.bfov.center.yawRad, 0.0, places=3)
        self.assertEqual(len(plan.scanViews), 4)
        # The same place is not jumped to again; another one is.
        result = controller.consume(plan, _observation(0.3, 0.1), (candidate,))
        self.assertFalse(controller.lastFrameReacquired)
        self.assertEqual(controller.lastFrameTrace["candidates"][0]["verdict"], "distractor")
        self.assertEqual(result.status, TrackStatus.LOST)
        elsewhere = _observation(0.8, 0.8, yaw=-2.0, viewId=2)
        self._step(controller, 10, _observation(0.3, 0.1), (candidate, elsewhere))
        self.assertTrue(controller.lastFrameReacquired)

    def testTheStateTraceIsWrittenWithOneRowPerFrame(self) -> None:
        controller = self._controller(lossActions="jump")
        rows = []
        self._lose(controller, 4)
        rows.append({**controller.lastFrameTrace, "forwards": 1, "memory": "frozen"})
        candidate = _observation(0.8, 0.8, yaw=2.0, viewId=1)
        self._step(controller, 5, _observation(0.3, 0.1), (candidate,))
        rows.append({**controller.lastFrameTrace, "forwards": 5, "memory": "reset"})
        with tempfile.TemporaryDirectory() as directory:
            path = writeStateTrace(Path(directory), "ours", "s", rows)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0].split(","), list(TRACE_COLUMNS))
        self.assertEqual(len(lines), 3)
        self.assertIn("LOST,TRACKING,JUMP,jump,reset", lines[2])
        cell = lines[2].split(",", len(TRACE_COLUMNS) - 1)[-1]
        candidates = json.loads(cell.strip('"').replace('""', '"'))
        self.assertEqual(candidates[0]["verdict"], "accepted")


if __name__ == "__main__":
    unittest.main()
