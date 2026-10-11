"""Temporary pseudo-ground-truth helpers for AirSim360 masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from track360.core.errors import DecodeError, ProtocolError
from track360.core.protocols import PseudoTrackBuilder as PseudoTrackBuilderProtocol
from track360.core.types import BBoxXYWH, FramePacket
from track360.geometry.seam import minimalCircularInterval


@dataclass(slots=True)
class MaskPseudoTrackBuilder(PseudoTrackBuilderProtocol):
    """Build an initial box and a pseudo-ground-truth box from one instance mask."""

    missingVisibleBox: BBoxXYWH = BBoxXYWH(0.0, 0.0, 1.0, 1.0)

    def buildInitialBox(self, frame: FramePacket, targetInstanceId: int) -> BBoxXYWH:
        bbox, visible = self.buildPseudoGroundTruth(frame, targetInstanceId)
        if not visible:
            raise DecodeError(f"target instance {targetInstanceId} is not visible")
        return bbox

    def buildPseudoGroundTruth(
        self,
        frame: FramePacket,
        targetInstanceId: int,
    ) -> tuple[BBoxXYWH, bool]:
        instance = _requireInstanceMask(frame)
        targetMask = instance == int(targetInstanceId)
        if not np.any(targetMask):
            return self.missingVisibleBox, False
        ys, xs = np.where(targetMask)
        xPx, widthPx = minimalCircularInterval(xs.astype(np.float64), frame.rgb.shape[1])
        y0 = float(np.min(ys))
        y1 = float(np.max(ys) + 1.0)
        return (
            BBoxXYWH(xPx=xPx, yPx=y0, widthPx=max(widthPx, 1.0), heightPx=max(y1 - y0, 1.0)),
            True,
        )


def _requireInstanceMask(frame: FramePacket) -> np.ndarray:
    if frame.segmentation is None or frame.segmentation.instance is None:
        raise ProtocolError("AirSim360 frame does not contain an instance mask")
    return frame.segmentation.instance


PseudoTrackBuilder = MaskPseudoTrackBuilder

__all__ = ["MaskPseudoTrackBuilder", "PseudoTrackBuilder"]
