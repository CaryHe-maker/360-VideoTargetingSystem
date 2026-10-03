import math
import unittest
from pathlib import Path

import numpy as np

from track360.controller import (
    SphericalMotionEstimator,
    TrackControllerImpl,
    TrackStateMachine,
)
from track360.controller.state_model import TransitionReason
from track360.core.config import loadConfig
from track360.core.errors import ProtocolError
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    FramePacket,
    ProjectedObservation,
    SequenceId,
    TemplateCommandKind,
    TrackStatus,
)
from track360.geometry import SphericalGeometryImpl, makeSphericalPoint

ROOT = Path(__file__).resolve().parents[2]


def _observation(viewId: int, bfov: BFoV, score: float = 0.95) -> ProjectedObservation:
    return ProjectedObservation(
        viewId=viewId,
        bfov=bfov,
        bbox=BBoxXYWH(150.0, 70.0, 60.0, 80.0),
        modelScore=score,
        appearanceScore=score,
        motionScore=score,
        scaleScore=score,
        fusedScore=score,
        localBox=BBoxXYWH(96.0, 88.0, 64.0, 80.0),
    )


class ControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = loadConfig(ROOT / "configs" / "default.yaml")
        self.geometry = SphericalGeometryImpl(
            boundarySamplesPerEdge=self.config.geometry.boundarySamplesPerEdge
        )

    def testMotionEstimatorWrapsSphericalDirection(self) -> None:
        estimator = SphericalMotionEstimator()
        estimator.initialize(makeSphericalPoint(math.pi - 0.05, 0.0), 0)
        updated = estimator.update(
            makeSphericalPoint(-math.pi + 0.05, 0.0),
            1_000_000_000,
            0.9,
        )
        predicted = estimator.predict(2_000_000_000)

        self.assertGreater(predicted.confidence, 0.0)
        self.assertGreater(updated.confidence, 0.0)
        self.assertTrue(-math.pi <= predicted.position[0] <= math.pi)
        with self.assertRaises(ProtocolError):
            estimator.predict(500_000_000)

    def testStateMachineUsesScoreGroupThresholds(self) -> None:
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

    def testStateMachineKeepsHardMissInUncertain(self) -> None:
        state = TrackStateMachine(self.config.tracking)
        state.initialize()
        state.update(0.8, True, True)
        state.update(0.8, True, True)
        state.update(0.8, True, True)

        hardMiss = state.update(0.0, False, False)

        self.assertEqual(hardMiss.status, TrackStatus.UNCERTAIN)
        self.assertEqual(hardMiss.reason, TransitionReason.HARD_MISS)

    def testControllerPlansGuardsAndCommitsOrderedUpdates(self) -> None:
        frame0 = FramePacket(
            sequenceId=SequenceId("s"),
            frameIndex=FrameIndex(0),
            timestampNs=0,
            rgb=np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(self.geometry, self.config)
        initialBox = BBoxXYWH(150.0, 70.0, 60.0, 80.0)
        initPlan = controller.buildInitialization(frame0, initialBox)

        self.assertEqual(initPlan.stateRevision, 0)
        self.assertLess(initPlan.templateBox.widthPx, 256.0)
        self.assertGreaterEqual(initPlan.templateBox.xPx, 0.0)
        controller.commitInitialization(initPlan)

        frame1 = FramePacket(
            sequenceId=SequenceId("s"),
            frameIndex=FrameIndex(1),
            timestampNs=1_000_000_000,
            rgb=frame0.rgb,
        )
        plan = controller.plan(frame1)
        self.assertEqual(plan.stateRevision, 1)
        self.assertEqual(plan.templateCommand.expectedRevision, 1)
        self.assertGreaterEqual(len(plan.views), 3)
        self.assertEqual(plan.templateCommand.kind, TemplateCommandKind.KEEP)

        commonBfov = plan.views[1].bfov
        result = controller.update(
            plan,
            tuple(_observation(viewId, commonBfov) for viewId in range(3)),
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.status, TrackStatus.TRACKING)

        frame2 = FramePacket(
            sequenceId=SequenceId("s"),
            frameIndex=FrameIndex(2),
            timestampNs=2_000_000_000,
            rgb=frame0.rgb,
        )
        nextPlan = controller.plan(frame2)
        self.assertEqual(nextPlan.stateRevision, 2)
        with self.assertRaises(ProtocolError):
            controller.update(nextPlan, (_observation(99, commonBfov),))

    def testControllerReturnsPredictionWhenNoCandidateIsAvailable(self) -> None:
        frame0 = FramePacket(
            SequenceId("s"),
            FrameIndex(0),
            0,
            np.zeros((180, 360, 3), dtype=np.uint8),
        )
        controller = TrackControllerImpl(self.geometry, self.config)
        initPlan = controller.buildInitialization(frame0, BBoxXYWH(150.0, 70.0, 60.0, 80.0))
        controller.commitInitialization(initPlan)
        frame1 = FramePacket(SequenceId("s"), FrameIndex(1), 1_000_000_000, frame0.rgb)
        plan = controller.plan(frame1)
        result = controller.update(plan, ())

        self.assertFalse(result.valid)
        self.assertEqual(result.status, TrackStatus.TRACKING)
        self.assertGreater(result.bbox.widthPx, 0.0)


if __name__ == "__main__":
    unittest.main()
