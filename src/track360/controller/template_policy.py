"""Template update policy owned by the DTC control thread.

Online templates are deliberately conservative: the immutable anchor remains in
the cache while only high-confidence, geometrically supported observations are
allowed to refresh the appearance stream.  This is the same long/short-term
memory split used by modern online trackers and prevents a single bad frame from
poisoning recovery.
"""

from __future__ import annotations

from dataclasses import dataclass

from track360.controller.decision_gate import FrameAggregate
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
        self._enabled = self._tuning.onlineTemplate

    def decide(
        self,
        status: TrackStatus,
        stableFrames: int,
        aggregate: FrameAggregate | None,
    ) -> TemplateDecision:
        if (
            not self._enabled
            or status not in {TrackStatus.TRACKING, TrackStatus.UNCERTAIN}
            or aggregate is None
            or aggregate.localBox is None
        ):
            return TemplateDecision(TemplateCommandKind.KEEP)

        # Do not encode weak or disagreeing boxes.  The stable slot is refreshed
        # less frequently and only after a sustained confirmed streak; recent is
        # intentionally refreshed sooner so scale/appearance changes are tracked.
        confidence = max(float(aggregate.confidence), float(aggregate.decisionScore))
        # ARTrack's sigmoid quality score is not the calibrated HiViT score
        # used by the legacy config (typical valid values are around 0.49).
        # Keep the threshold configurable so short-window tuning can lower it
        # without changing the immutable anchor policy.
        artrackMin = self._tuning.templateMinConfidence
        configuredMin = self._tracking.candidateMinScore if not self._enabled else 0.0
        if confidence < max(configuredMin, artrackMin):
            return TemplateDecision(TemplateCommandKind.KEEP)
        # The production fast path intentionally uses one centered view for
        # small targets. Permit that route to refresh templates only at a
        # stronger confidence threshold; multi-view support remains preferred.
        allowSingle = self._tuning.allowSingleViewTemplate
        if not aggregate.supported and not allowSingle and confidence < 0.84:
            return TemplateDecision(TemplateCommandKind.KEEP)
        if aggregate.supported and aggregate.agreementScore < 0.40:
            return TemplateDecision(TemplateCommandKind.KEEP)
        stablePeriod = max(1, self._tracking.stableFramesBeforeUpdate)
        if stableFrames >= stablePeriod and stableFrames % stablePeriod == 0:
            return TemplateDecision(
                TemplateCommandKind.UPDATE_STABLE,
                viewId=aggregate.representativeViewId,
                localBox=aggregate.localBox,
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
            viewId=aggregate.representativeViewId,
            localBox=aggregate.localBox,
        )


__all__ = ["TemplateDecision", "TemplatePolicy"]
