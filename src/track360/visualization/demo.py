"""A side-by-side video of a tracking run: the panorama with the result, and what the
tracker was shown.

``DemoRecorder`` plugs into ``runTracking`` as both its ``recorder`` and its
``resultRecorder``.  Left: the ERP frame with the committed box (drawn in two parts
when it crosses the seam).  Right: the local view of the frame with the backend's box.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from track360.core.errors import OutputError
from track360.core.types import (
    BBoxXYWH,
    FramePacket,
    LocalObservation,
    LocalView,
    ProjectedObservation,
    TrackResult,
    TrackStatus,
)

_STATUS_COLOURS = {
    TrackStatus.TRACKING: (40, 220, 60),
    TrackStatus.UNCERTAIN: (255, 190, 30),
    TrackStatus.LOST: (240, 60, 60),
}


class DemoRecorder:
    """Write an MP4, and optionally a GIF of every ``gifEvery``-th frame."""

    def __init__(
        self,
        videoPath: str | Path | None,
        *,
        gifPath: str | Path | None = None,
        heightPx: int = 480,
        framesPerSecond: float = 30.0,
        gifEvery: int = 3,
        gifWidthPx: int = 720,
        gifMaxFrames: int = 150,
    ) -> None:
        if videoPath is None and gifPath is None:
            raise OutputError("a demo needs a video path or a GIF path")
        if heightPx < 120 or heightPx % 2:
            raise OutputError("demo height must be an even number of at least 120 pixels")
        self._videoPath = None if videoPath is None else Path(videoPath)
        self._gifPath = None if gifPath is None else Path(gifPath)
        self._height = heightPx
        self._fps = framesPerSecond
        self._gifEvery = max(1, gifEvery)
        self._gifWidth = gifWidthPx
        self._gifMaxFrames = gifMaxFrames
        self._writer = None
        self._view: LocalView | None = None
        self._localBox: BBoxXYWH | None = None
        self._gifFrames: list[np.ndarray] = []
        self._count = 0

    # ---- the recorder interface of the tracking loop
    def recordLocalRgb(self, frame: FramePacket, views: Sequence[LocalView]) -> None:
        del frame
        self._view = views[0] if views else None
        self._localBox = None

    def recordBackendBoxes(
        self,
        frame: FramePacket,
        views: Sequence[LocalView],
        observations: Sequence[LocalObservation],
    ) -> None:
        del frame, views
        self._localBox = observations[0].bbox if observations else None

    def recordGeometryBoxes(
        self, frame: FramePacket, observations: Sequence[ProjectedObservation]
    ) -> None:
        del frame, observations

    def record(
        self, frame: FramePacket, result: TrackResult, *, stateScore: float | None = None
    ) -> None:
        del stateScore
        canvas = self._compose(frame, result)
        self._write(canvas)
        self._view = None
        self._localBox = None

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._gifPath is not None and self._gifFrames:
            from PIL import Image

            self._gifPath.parent.mkdir(parents=True, exist_ok=True)
            images = [Image.fromarray(image) for image in self._gifFrames]
            images[0].save(
                self._gifPath,
                save_all=True,
                append_images=images[1:],
                duration=int(1000 * self._gifEvery / self._fps),
                loop=0,
                optimize=True,
            )
        self._gifFrames = []

    # ---- drawing
    def _compose(self, frame: FramePacket, result: TrackResult) -> np.ndarray:
        import cv2

        height = self._height
        frameHeight, frameWidth = frame.rgb.shape[:2]
        panoramaWidth = 2 * (int(round(height * frameWidth / frameHeight)) // 2)
        panorama = cv2.resize(frame.rgb, (panoramaWidth, height), interpolation=cv2.INTER_AREA)
        colour = _STATUS_COLOURS.get(result.status, (255, 255, 255))
        scale = panoramaWidth / frameWidth
        box = result.bbox
        left = (box.xPx % frameWidth) * scale
        top, bottom = (
            box.yPx * height / frameHeight,
            (box.yPx + box.heightPx) * height / frameHeight,
        )
        width = min(box.widthPx * scale, panoramaWidth)
        thickness = max(2, height // 200)
        # A box that runs past the right edge continues from the left one.
        for start in (left, left - panoramaWidth):
            cv2.rectangle(
                panorama,
                (int(round(start)), int(round(top))),
                (int(round(start + width)), int(round(bottom))),
                colour,
                thickness,
            )
        _label(
            panorama,
            f"frame {int(frame.frameIndex)}  {result.status.name}  score {result.confidence:.2f}",
            colour,
        )
        side = np.zeros((height, height, 3), dtype=np.uint8)
        if self._view is not None:
            view = self._view.rgb
            side = cv2.resize(view, (height, height), interpolation=cv2.INTER_LINEAR)
            if self._localBox is not None:
                fx, fy = height / view.shape[1], height / view.shape[0]
                local = self._localBox
                cv2.rectangle(
                    side,
                    (int(round(local.xPx * fx)), int(round(local.yPx * fy))),
                    (
                        int(round((local.xPx + local.widthPx) * fx)),
                        int(round((local.yPx + local.heightPx) * fy)),
                    ),
                    colour,
                    thickness,
                )
            _label(side, "tracker view", (255, 255, 255))
        return np.concatenate((panorama, side), axis=1)

    def _write(self, canvas: np.ndarray) -> None:
        import cv2

        if self._videoPath is not None:
            if self._writer is None:
                self._videoPath.parent.mkdir(parents=True, exist_ok=True)
                self._writer = cv2.VideoWriter(
                    str(self._videoPath),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self._fps,
                    (canvas.shape[1], canvas.shape[0]),
                )
                if not self._writer.isOpened():
                    self._writer = None
                    raise OutputError(f"cannot open the demo video for writing: {self._videoPath}")
            self._writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        if (
            self._gifPath is not None
            and self._count % self._gifEvery == 0
            and len(self._gifFrames) < self._gifMaxFrames
        ):
            gifHeight = 2 * (int(round(canvas.shape[0] * self._gifWidth / canvas.shape[1])) // 2)
            self._gifFrames.append(
                cv2.resize(canvas, (self._gifWidth, gifHeight), interpolation=cv2.INTER_AREA)
            )
        self._count += 1


def _label(image: np.ndarray, text: str, colour: tuple[int, int, int]) -> None:
    import cv2

    scale = max(0.45, image.shape[0] / 800.0)
    thickness = max(1, int(round(scale * 2)))
    origin = (8, int(22 * scale / 0.45 * 0.6) + 8)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness)


__all__ = ["DemoRecorder"]
