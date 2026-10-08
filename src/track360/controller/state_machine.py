"""Track state from the fused state score of each frame."""

from __future__ import annotations

from track360.controller.state_model import TrackMode, TransitionDecision, TransitionReason
from track360.core.config import BackendTuningConfig
from track360.core.errors import ProtocolError


class TrackStateMachine:
    """TRACKING while the state score holds, UNCERTAIN when it drops, LOST when it stays down.

    A frame whose state score is below ``uncertainScore`` is not trusted.  The first
    such frames are UNCERTAIN; after ``lostAfterFrames`` of them in a row the target
    counts as LOST.  One trusted frame brings the track back to TRACKING.
    """

    def __init__(self, tuning: BackendTuningConfig | None = None) -> None:
        tuning = tuning or BackendTuningConfig()
        self._uncertainScore = tuning.uncertainScore
        self._lostAfterFrames = tuning.lostAfterFrames
        self._initialized = False
        self._untrustedFrames = 0

    @property
    def untrustedFrames(self) -> int:
        """How many frames in a row were below the uncertain score."""
        return self._untrustedFrames

    def initialize(self) -> None:
        if self._initialized:
            raise ProtocolError("track state machine is already initialized")
        self._initialized = True
        self._untrustedFrames = 0

    def reset(self) -> None:
        """Start counting afresh, as after the track jumped to a re-found target."""
        self._untrustedFrames = 0

    def transition(
        self,
        mode: TrackMode,
        stateScore: float,
        *,
        measurementAccepted: bool,
    ) -> TransitionDecision:
        if not 0.0 <= float(stateScore) <= 1.0:
            raise ProtocolError("StateScore must be in [0, 1]")
        if mode is TrackMode.INIT:
            self._untrustedFrames = 0
            return TransitionDecision(
                "COMMIT", TrackMode.TRACKING, TransitionReason.INITIALIZED, measurementAccepted
            )
        if stateScore >= self._uncertainScore:
            self._untrustedFrames = 0
            return TransitionDecision(
                "COMMIT",
                TrackMode.TRACKING,
                TransitionReason.RELIABLE_MEASUREMENT,
                measurementAccepted,
            )
        self._untrustedFrames += 1
        nextMode = (
            TrackMode.LOST
            if self._untrustedFrames >= self._lostAfterFrames
            else TrackMode.UNCERTAIN
        )
        reason = (
            TransitionReason.HARD_MISS if stateScore <= 0.0 else TransitionReason.WEAK_MEASUREMENT
        )
        return TransitionDecision("COMMIT", nextMode, reason, measurementAccepted)


__all__ = ["TrackStateMachine"]
