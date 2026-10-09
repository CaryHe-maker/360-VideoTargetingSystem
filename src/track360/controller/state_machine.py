"""Track state from the scores of each frame.

Two rules decide the state (``backendTuning.stateRule``):

``fused``
    One number, the weighted mean of the three scores, against ``uncertainScore``.

``relative``
    The backend score and the template similarity are each compared with the
    median of the frames of this sequence trusted so far, and the mean of the two
    relative deviations is thresholded.  No score has a level or a weight of its
    own.  A doubt holds until the mean deviation has been back within half the
    threshold for ``releaseFrames`` frames.

``split``
    Each score answers its own question.  The backend score and the motion score say
    when a frame stops being trusted; the similarity to the frame-0 template says
    whether the target is really gone, whether a doubted track may be trusted again
    and whether a jump to a scan candidate was right.
"""

from __future__ import annotations

from collections import deque
from statistics import fmean, median

from track360.controller.state_model import TrackMode, TransitionDecision, TransitionReason
from track360.core.config import BackendTuningConfig
from track360.core.errors import ProtocolError

# Frames whose mean template similarity starts a doubt under the split rule.
APPEARANCE_ENTER_FRAMES = 5
# The relative rule: frames trusted without a check at the start of a sequence, and
# the share of the entry threshold the deviation must be back within for release.
RELATIVE_WARM_FRAMES = 10
RELATIVE_RELEASE_SHARE = 0.5


class TrackStateMachine:
    """TRACKING while the scores hold, UNCERTAIN when they drop, LOST when they stay down.

    Under the fused rule a frame below ``uncertainScore`` is not trusted; after
    ``lostAfterFrames`` such frames in a row the target counts as LOST and one trusted
    frame brings the track back.  With ``stateLatch`` a doubt, once raised, holds
    until the score has been ``latchReleaseMargin`` above the threshold for
    ``releaseFrames`` frames in a row.

    The split rule is described in the module docstring; PROBATION is the state of
    the frames right after a jump to a scan candidate.
    """

    def __init__(self, tuning: BackendTuningConfig | None = None) -> None:
        self._tuning = tuning or BackendTuningConfig()
        self._initialized = False
        self._untrustedFrames = 0
        self._calmFrames = 0
        self._latched = False
        self._recentAppearance: deque[float] = deque(maxlen=APPEARANCE_ENTER_FRAMES)
        self._doubtAppearance: deque[float] = deque(maxlen=self._tuning.lostAfterFrames)
        self._probationAppearance: list[float] = []
        self._probationBackend: list[float] = []
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

    @property
    def recentAppearance(self) -> float | None:
        """Mean template similarity of the last trusted frames; None before the first."""
        return fmean(self._recentAppearance) if self._recentAppearance else None

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
        self._recentAppearance.clear()
        self._doubtAppearance.clear()
        self._probationAppearance.clear()
        self._probationBackend.clear()

    def startProbation(self) -> None:
        """The track jumped to a scan candidate that still has to prove itself."""
        self.reset()

    def transition(
        self,
        mode: TrackMode,
        stateScore: float,
        *,
        measurementAccepted: bool,
        backendScore: float | None = None,
        appearanceScore: float | None = None,
        motionScore: float | None = None,
    ) -> TransitionDecision:
        if not 0.0 <= float(stateScore) <= 1.0:
            raise ProtocolError("StateScore must be in [0, 1]")
        if mode is TrackMode.INIT:
            self.reset()
            return TransitionDecision(
                "COMMIT", TrackMode.TRACKING, TransitionReason.INITIALIZED, measurementAccepted
            )
        if self._tuning.stateRule == "relative":
            nextMode, reason = self._relative(
                backendScore, appearanceScore, hasBox=backendScore is not None
            )
        elif self._tuning.stateRule == "split":
            nextMode, reason = self._split(
                mode,
                0.0 if backendScore is None else float(backendScore),
                0.0 if appearanceScore is None else float(appearanceScore),
                0.0 if motionScore is None else float(motionScore),
                hasBox=backendScore is not None and stateScore > 0.0,
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
        if trusted:
            released = self._latched
            self._untrustedFrames = 0
            self._calmFrames = 0
            self._latched = False
            return TrackMode.TRACKING, (
                TransitionReason.RELEASED if released else TransitionReason.RELIABLE_MEASUREMENT
            )
        self._latched = True
        self._untrustedFrames += 1
        nextMode = (
            TrackMode.LOST
            if self._untrustedFrames >= tuning.lostAfterFrames
            else TrackMode.UNCERTAIN
        )
        return nextMode, (
            TransitionReason.WEAK_MEASUREMENT if hasBox else TransitionReason.HARD_MISS
        )

    def _fused(self, stateScore: float) -> tuple[TrackMode, TransitionReason]:
        tuning = self._tuning
        if self._latched:
            release = tuning.uncertainScore + tuning.latchReleaseMargin
            self._calmFrames = self._calmFrames + 1 if stateScore >= release else 0
            trusted = self._calmFrames >= tuning.releaseFrames
        else:
            trusted = stateScore >= tuning.uncertainScore
        if trusted:
            released = self._latched
            self._untrustedFrames = 0
            self._calmFrames = 0
            self._latched = False
            return TrackMode.TRACKING, (
                TransitionReason.RELEASED if released else TransitionReason.RELIABLE_MEASUREMENT
            )
        self._latched = tuning.stateLatch
        self._untrustedFrames += 1
        nextMode = (
            TrackMode.LOST
            if self._untrustedFrames >= tuning.lostAfterFrames
            else TrackMode.UNCERTAIN
        )
        return nextMode, (
            TransitionReason.HARD_MISS if stateScore <= 0.0 else TransitionReason.WEAK_MEASUREMENT
        )

    def _split(
        self, mode: TrackMode, backend: float, appearance: float, motion: float, *, hasBox: bool
    ) -> tuple[TrackMode, TransitionReason]:
        tuning = self._tuning
        if mode is TrackMode.PROBATION:
            self._probationAppearance.append(appearance)
            self._probationBackend.append(backend)
            if len(self._probationAppearance) < tuning.probationFrames:
                return TrackMode.PROBATION, TransitionReason.ON_PROBATION
            passed = (
                median(self._probationAppearance) >= tuning.appearanceReleaseScore
                and median(self._probationBackend) >= tuning.backendReleaseScore
            )
            self.reset()
            if passed:
                return TrackMode.TRACKING, TransitionReason.PROBATION_PASSED
            # Back to where the track was before the jump: lost.
            self._untrustedFrames = tuning.lostAfterFrames
            return TrackMode.LOST, TransitionReason.PROBATION_FAILED
        if mode is TrackMode.TRACKING:
            self._recentAppearance.append(appearance)
            if not hasBox:
                reason = TransitionReason.HARD_MISS
            elif backend < tuning.backendEnterScore:
                reason = TransitionReason.BACKEND_LOW
            elif motion < tuning.motionEnterScore:
                reason = TransitionReason.MOTION_LOW
            elif (
                len(self._recentAppearance) == APPEARANCE_ENTER_FRAMES
                and fmean(self._recentAppearance) < tuning.appearanceEnterScore
            ):
                reason = TransitionReason.APPEARANCE_LOW
            else:
                return TrackMode.TRACKING, TransitionReason.RELIABLE_MEASUREMENT
            self._untrustedFrames = 1
            self._calmFrames = 0
            self._doubtAppearance.clear()
            self._doubtAppearance.append(appearance)
            return self._doubted(TrackMode.UNCERTAIN, reason)
        # UNCERTAIN or LOST: the doubt holds until both scores are back for a while.
        self._untrustedFrames += 1
        self._doubtAppearance.append(appearance)
        calm = (
            hasBox
            and backend >= tuning.backendReleaseScore
            and appearance >= tuning.appearanceReleaseScore
        )
        self._calmFrames = self._calmFrames + 1 if calm else 0
        if self._calmFrames >= tuning.releaseFrames:
            self.reset()
            self._recentAppearance.append(appearance)
            return TrackMode.TRACKING, TransitionReason.RELEASED
        return self._doubted(mode, TransitionReason.DOUBT_HELD)

    def _doubted(
        self, mode: TrackMode, reason: TransitionReason
    ) -> tuple[TrackMode, TransitionReason]:
        """LOST once the doubt lasted long enough and the box does not look like the target."""
        tuning = self._tuning
        if mode is TrackMode.LOST:
            return TrackMode.LOST, reason
        if (
            self._untrustedFrames >= tuning.lostAfterFrames
            and fmean(self._doubtAppearance) < tuning.appearanceLostScore
        ):
            return TrackMode.LOST, TransitionReason.APPEARANCE_CONFIRMED_LOSS
        return TrackMode.UNCERTAIN, reason


__all__ = ["APPEARANCE_ENTER_FRAMES", "TrackStateMachine"]
