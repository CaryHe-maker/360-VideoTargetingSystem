"""Template slots and command execution for a stateful tracker backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from track360.backends.artrack_model import ARTrackBackend
from track360.core.errors import ModelError, ProtocolError
from track360.core.types import (
    BBoxXYWH,
    FrameIndex,
    LocalView,
    TemplateCommand,
    TemplateCommandKind,
)


@dataclass(frozen=True, slots=True)
class TemplateSample:
    """An immutable description of one encoded local template."""

    viewId: int
    frameIndex: FrameIndex
    bbox: BBoxXYWH
    features: object


@dataclass(frozen=True, slots=True)
class TemplateSnapshot:
    """The fixed ordering presented to the ARTrackV2 session."""

    anchor: TemplateSample
    recent: TemplateSample | None
    stable: TemplateSample | None
    revision: int

    @property
    def features(self) -> tuple[object, ...]:
        # ARTrackV2 has exactly two template streams. Keep the immutable anchor
        # and select the newest online sample. Once a stable sample exists, a
        # later recent update must still be visible to the model; otherwise the
        # supposedly adaptive path silently freezes after the first promotion.
        candidates = tuple(item for item in (self.recent, self.stable) if item is not None)
        secondary = max(candidates, key=lambda item: int(item.frameIndex)) if candidates else None
        return (
            (self.anchor.features,)
            if secondary is None
            else (
                self.anchor.features,
                secondary.features,
            )
        )


class TemplateCache:
    """Own anchor/recent/stable slots and apply commands atomically."""

    def __init__(self) -> None:
        self._anchor: TemplateSample | None = None
        self._recent: TemplateSample | None = None
        self._stable: TemplateSample | None = None
        self._revision = 0

    @property
    def initialized(self) -> bool:
        return self._anchor is not None

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def activeTemplateFrameIndex(self) -> int:
        snapshot = self.snapshot()
        if snapshot.stable is not None:
            return int(snapshot.stable.frameIndex)
        if snapshot.recent is not None:
            return int(snapshot.recent.frameIndex)
        return int(snapshot.anchor.frameIndex)

    def initialize(
        self, backend: ARTrackBackend, template: LocalView, templateBox: BBoxXYWH
    ) -> None:
        if self.initialized:
            raise ProtocolError("template cache is already initialized")
        _validateBox(templateBox, template)
        features = backend.encodeTemplateView(template, templateBox)
        self._anchor = TemplateSample(
            viewId=template.spec.viewId,
            frameIndex=FrameIndex(0),
            bbox=templateBox,
            features=features,
        )
        self._recent = None
        self._stable = None
        self._revision = 0

    def apply(
        self,
        backend: ARTrackBackend,
        command: TemplateCommand,
        previousViews: Mapping[int, LocalView],
    ) -> None:
        """Apply one command, committing revision only after successful encoding."""
        if not self.initialized or self._anchor is None:
            raise ProtocolError("template cache is not initialized")
        if command.expectedRevision != self._revision + 1:
            raise ProtocolError(
                "template command revision mismatch: "
                f"expected={self._revision + 1}, actual={command.expectedRevision}"
            )
        if command.kind is TemplateCommandKind.KEEP:
            self._revision = command.expectedRevision
            return
        if not backend.supportsOnlineTemplates:
            raise ProtocolError("ARTrackV2 session does not support online template commands")
        if command.kind is TemplateCommandKind.RESET_TO_ANCHOR:
            self._recent = None
            self._stable = None
            self._revision = command.expectedRevision
            return
        if command.viewId is None or command.localBox is None:
            raise ProtocolError("template update command has no selected view")
        view = previousViews.get(command.viewId)
        if view is None:
            raise ProtocolError(f"template update viewId is not available: {command.viewId}")
        _validateBox(command.localBox, view)
        features = backend.encodeTemplateView(view, command.localBox)
        sample = TemplateSample(
            viewId=command.viewId,
            frameIndex=command.frameIndex,
            bbox=command.localBox,
            features=features,
        )
        if command.kind is TemplateCommandKind.UPDATE_RECENT:
            self._recent = sample
        elif command.kind is TemplateCommandKind.UPDATE_STABLE:
            self._stable = sample
        else:
            raise ProtocolError(f"unsupported template command: {command.kind}")
        self._revision = command.expectedRevision

    def snapshot(self) -> TemplateSnapshot:
        if self._anchor is None:
            raise ProtocolError("template cache is not initialized")
        return TemplateSnapshot(self._anchor, self._recent, self._stable, self._revision)

    def clear(self) -> None:
        self._anchor = None
        self._recent = None
        self._stable = None
        self._revision = 0


def _validateBox(box: BBoxXYWH, view: LocalView) -> None:
    if (
        box.xPx < 0.0
        or box.yPx < 0.0
        or box.xPx + box.widthPx > view.spec.outputWidthPx
        or box.yPx + box.heightPx > view.spec.outputHeightPx
    ):
        raise ModelError(f"template box is outside view {view.spec.viewId}: {box}")


__all__ = ["TemplateCache", "TemplateSample", "TemplateSnapshot"]
