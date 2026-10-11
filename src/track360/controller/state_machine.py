"""Track state from the scores of each frame.

Without loss handling the state is only a report: one number, the weighted mean of the
backend score and the motion score, against ``uncertainScore``.

With loss handling the state decides when to search, and the relative rule is used:
the backend score and the template similarity are each compared with the median of
the frames of this sequence trusted so far, and the mean of the two relative
deviations is thresholded.  No score has a level or a weight of its own.  A doubt
holds until the mean deviation has been back within half the threshold for
``releaseFrames`` frames.
"""

from __future__ import annotations

from statistics import median

from track360.controller.state_model import TrackMode, TransitionDecision, TransitionReason
from track360.core.config import BackendTuningConfig
from track360.core.errors import ProtocolError

# The relative rule: frames trusted without a check at the start of a sequence, and
# the share of the entry threshold the deviation must be back within for release.
RELATIVE_WARM_FRAMES = 10
RELATIVE_RELEASE_SHARE = 0.5


class TrackStateMachine:
    """TRACKING while the scores hold, UNCERTAIN when they drop, LOST when they stay down.

    After ``lostAfterFrames`` untrusted frames in a row the target counts as LOST.
    Which rule judges a frame is described in the module docstring.
    """

    def __init__(self, tuning: BackendTuningConfig | None = None) -> None:
        self._tuning = tuning or BackendTuningConfig()
        self._initialized = False
        self._untrustedFrames = 0
        self._calmFrames = 0
        self._latched = False
        # Trusted values of each score so far; kept across jumps, since they
        # describe the target and not the track.
        self._trusted: dict[str, list[float]] = {"backend": [], "appearance": []}
        self._relativeSignal: float | None = None

    @property
    def relativeSignal(self) -> float | None:
        """Mean relative deviation of the last frame under the relative rule."""
        return self._relativeSignal

    @property
    def untrustedFrames(self) -> int:
        """How many frames in a row were not trusted."""
        return self._untrustedFrames

    @property
    def calmFrames(self) -> int:
        """How many frames in a row met the release condition while doubted."""
        return self._calmFrames

    def initialize(self) -> None:
        if self._initialized:
            raise ProtocolError("track state machine is already initialized")
        self._initialized = True
        self.reset()

    def reset(self) -> None:
        """Start counting afresh, as after the track jumped to a re-found target."""
        self._untrustedFrames = 0
        self._calmFrames = 0
        self._latched = False

    def transition(
        self,
        mode: TrackMode,
        stateScore: float,
        *,
        measurementAccepted: bool,
        backendScore: float | None = None,
        appearanceScore: float | None = None,
    ) -> TransitionDecision:
        if not 0.0 <= float(stateScore) <= 1.0:
            raise ProtocolError("StateScore must be in [0, 1]")
        if mode is TrackMode.INIT:
            self.reset()
            return TransitionDecision(
                "COMMIT", TrackMode.TRACKING, TransitionReason.INITIALIZED, measurementAccepted
            )
        if self._tuning.lossHandling:
            nextMode, reason = self._relative(
                backendScore, appearanceScore, hasBox=backendScore is not None
            )
        else:
            nextMode, reason = self._fused(float(stateScore))
        return TransitionDecision("COMMIT", nextMode, reason, measurementAccepted)

    def _deviation(self, channel: str, value: float) -> float:
        """``value`` relative to the median of the trusted frames, minus one."""
        history = self._trusted[channel]
        if len(history) < RELATIVE_WARM_FRAMES:
            history.append(value)
            return 0.0
        deviation = value / max(median(history), 1e-6) - 1.0
        if deviation > -self._tuning.relativeGate:
            history.append(value)
        return deviation

    def _relative(
        self, backend: float | None, appearance: float | None, *, hasBox: bool
    ) -> tuple[TrackMode, TransitionReason]:
        tuning = self._tuning
        if hasBox:
            signal = 0.5 * (
                self._deviation("backend", float(backend or 0.0))
                + self._deviation("appearance", float(appearance or 0.0))
            )
        else:
            signal = -1.0
        self._relativeSignal = signal
        threshold = tuning.relativeEnterDeviation
        if self._latched:
            calm = signal > threshold * RELATIVE_RELEASE_SHARE
            self._calmFrames = self._calmFrames + 1 if calm else 0
            trusted = self._calmFrames >= tuning.releaseFrames
        else:
            trusted = signal >= threshold
        return self._judged(
            trusted,
            latch=True,
            miss=TransitionReason.WEAK_MEASUREMENT if hasBox else TransitionReason.HARD_MISS,
        )

    def _fused(self, stateScore: float) -> tuple[TrackMode, TransitionReason]:
        return self._judged(
            stateScore >= self._tuning.uncertainScore,
            latch=False,
            miss=(
                TransitionReason.HARD_MISS
                if stateScore <= 0.0
                else TransitionReason.WEAK_MEASUREMENT
            ),
        )

    def _judged(
        self, trusted: bool, *, latch: bool, miss: TransitionReason
    ) -> tuple[TrackMode, TransitionReason]:
        if trusted:
            released = self._latched
            self._untrustedFrames = 0
            self._calmFrames = 0
            self._latched = False
            return TrackMode.TRACKING, (
                TransitionReason.RELEASED if released else TransitionReason.RELIABLE_MEASUREMENT
            )
        self._latched = latch
        self._untrustedFrames += 1
        nextMode = (
            TrackMode.LOST
            if self._untrustedFrames >= self._tuning.lostAfterFrames
            else TrackMode.UNCERTAIN
        )
        return nextMode, miss


__all__ = ["RELATIVE_RELEASE_SHARE", "RELATIVE_WARM_FRAMES", "TrackStateMachine"]
