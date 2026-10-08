"""Template update policy owned by the DTC control thread.

Online templates are deliberately conservative: the immutable anchor remains in
the cache while only sufficiently confident observations are allowed to refresh
the appearance stream.  This is the same long/short-term memory split used by
modern online trackers and prevents a single bad frame from poisoning recovery.
"""

from __future__ import annotations

from dataclasses import dataclass

from track360.controller.state_model import StateObservation
from track360.core.config import BackendTuningConfig, TrackingConfig
from track360.core.types import BBoxXYWH, TemplateCommandKind, TrackStatus


@dataclass(frozen=True, slots=True)
class TemplateDecision:
    kind: TemplateCommandKind
    viewId: int | None = None
    localBox: BBoxXYWH | None = None


class TemplatePolicy:
    """Maintain a safe short/long-term online template pair."""

    def __init__(
        self,
        trackingConfig: TrackingConfig,
        backendTuning: BackendTuningConfig | None = None,
    ) -> None:
        self._tracking = trackingConfig
        self._tuning = backendTuning or BackendTuningConfig()
        # The sequence-level model rewrites its own appearance feature every frame and
        # takes no template updates from outside.
        self._enabled = self._tuning.onlineTemplate and not self._tuning.sequenceModel

    def decide(
        self,
        status: TrackStatus,
        stableFrames: int,
        observation: StateObservation | None,
    ) -> TemplateDecision:
        if (
            not self._enabled
            or status not in {TrackStatus.TRACKING, TrackStatus.UNCERTAIN}
            or observation is None
            or observation.viewId is None
            or observation.localBox is None
        ):
            return TemplateDecision(TemplateCommandKind.KEEP)

        # Do not encode weak boxes.  ARTrack's sigmoid quality score is not a
        # calibrated probability (typical valid values are around 0.49), so the
        # threshold is its own setting rather than ``candidateMinScore``.
        if observation.stateScore < self._tuning.templateMinConfidence:
            return TemplateDecision(TemplateCommandKind.KEEP)
        # The stable slot is refreshed less frequently and only after a sustained
        # confirmed streak; recent is intentionally refreshed sooner so
        # scale/appearance changes are tracked.
        stablePeriod = max(1, self._tracking.stableFramesBeforeUpdate)
        if stableFrames >= stablePeriod and stableFrames % stablePeriod == 0:
            return TemplateDecision(
                TemplateCommandKind.UPDATE_STABLE,
                viewId=observation.viewId,
                localBox=observation.localBox,
            )
        # Refresh the short-term stream at a modest cadence.  Encoding every
        # frame is unnecessary and can overfit transient blur/occlusion.
        # ARTrack quality scores are not calibrated probabilities and the
        # state machine can legitimately spend many frames in UNCERTAIN while
        # the box remains geometrically correct. Refresh every accepted call
        # in that case so appearance/scale changes do not accumulate drift.
        if stableFrames > 0 and stableFrames % 2 != 0:
            return TemplateDecision(TemplateCommandKind.KEEP)
        return TemplateDecision(
            TemplateCommandKind.UPDATE_RECENT,
            viewId=observation.viewId,
            localBox=observation.localBox,
        )


__all__ = ["TemplateDecision", "TemplatePolicy"]
