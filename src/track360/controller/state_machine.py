"""Score-group state reduction for the four-state controller."""

from __future__ import annotations

from track360.controller.state_model import (
    ScoreGroup,
    TrackMode,
    TransitionDecision,
    TransitionReason,
)
from track360.core.config import TrackingConfig
from track360.core.errors import ProtocolError


class TrackStateMachine:
    """State selection and a committed ten-score rolling history."""

    def __init__(self, trackingConfig: TrackingConfig) -> None:
        self._config = trackingConfig
        self._initialized = False
        self._scoreGroup = ScoreGroup()

    @property
    def scoreGroup(self) -> ScoreGroup:
        return self._scoreGroup

    def initialize(self) -> None:
        if self._initialized:
            raise ProtocolError("track state machine is already initialized")
        self._initialized = True
        self._scoreGroup = ScoreGroup()

    def transition(
        self,
        mode: TrackMode,
        stateScore: float,
        *,
        measurementAccepted: bool,
    ) -> TransitionDecision:
        """Choose the next mode before appending the current score."""
        if not 0.0 <= float(stateScore) <= 1.0:
            raise ProtocolError("StateScore must be in [0, 1]")
        if mode is TrackMode.INIT:
            nextMode = TrackMode.TRACKING
            reason = TransitionReason.INITIALIZED
        elif len(self._scoreGroup.values) < 2:
            nextMode = TrackMode.TRACKING
            reason = TransitionReason.RELIABLE_MEASUREMENT
        elif len(self._scoreGroup.values) == 2:
            nextMode = (
                TrackMode.TRACKING
                if stateScore > self._scoreGroup.values[-1]
                else TrackMode.UNCERTAIN
            )
            reason = TransitionReason.RELIABLE_MEASUREMENT
        else:
            thresholds = self._scoreGroup.thresholds()
            assert thresholds is not None
            uncertainThreshold, lostThreshold = thresholds
            if stateScore <= 0.0 and uncertainThreshold == 0.0 and lostThreshold == 0.0:
                # Keep the hard-miss diagnostic, but run the UNCERTAIN path for this experiment.
                nextMode = TrackMode.UNCERTAIN
                reason = TransitionReason.HARD_MISS
            elif stateScore >= uncertainThreshold:
                nextMode = TrackMode.TRACKING
                reason = TransitionReason.RELIABLE_MEASUREMENT
            elif stateScore >= lostThreshold:
                nextMode = TrackMode.UNCERTAIN
                reason = TransitionReason.WEAK_MEASUREMENT
            else:
                nextMode = TrackMode.UNCERTAIN
                reason = TransitionReason.HARD_MISS
        return TransitionDecision("COMMIT", nextMode, reason, measurementAccepted)

    def recordScore(self, stateScore: float) -> None:
        self._scoreGroup.append(stateScore)

__all__ = ["TrackStateMachine"]
