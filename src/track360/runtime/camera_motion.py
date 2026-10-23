"""The camera's rotation while a track is in doubt (experimental, E053).

A turning 360-degree camera moves every point of the image the same way on the sphere,
so a place where the target was last trusted is no longer where it appears after the
camera turned.  The rotation is needed only from the last trusted frame on, so nothing
is computed while the track is trusted: the latest trusted frame is only kept.  When
a frame is not trusted (``mark``), a worker thread starts from that kept frame and
from then on matches every new frame's ORB features with the previous frame's,
accumulating the rotation (``track360.geometry.camera_rotation``) alongside the
tracker's own work.  ``rotation(a, b)`` takes a direction seen in the last trusted
frame ``a`` to frame ``b``, waiting for frame ``b``.  A step with too few inliers
counts as no rotation.  All state changes go through the worker's queue in order, so
results do not depend on thread timing.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np

from track360.geometry import camera_rotation as cr

WORK_WIDTH = 384
ITERATIONS = 100


class CameraMotion:
    def __init__(
        self,
        workWidth: int = WORK_WIDTH,
        seed: int = 0,
        fast: bool = True,
        iterations: int = ITERATIONS,
        lazy: bool = True,
    ) -> None:
        self._width = int(workWidth)
        self._fast = bool(fast)
        self._iterations = int(iterations)
        self._lazy = bool(lazy)
        self._seed = int(seed)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="track360-camera")
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._jobs: dict[int, Future[np.ndarray]] = {}
        # Main-thread view: the latest frame read, and the frame the chain starts from.
        self._latest: tuple[int, np.ndarray] | None = None
        self._kept: tuple[int, np.ndarray] | None = None
        self._anchor: int | None = None
        self._active = not self._lazy
        # Worker-thread state.
        self._rng = np.random.default_rng(seed)
        self._previous: tuple | None = None
        self._orientation = np.eye(3)
        self.unreliableSteps = 0
        self.measuredFrames = 0

    # ---------------------------------------------------------------- main thread
    def submit(self, frameIndex: int, rgb: np.ndarray) -> None:
        """Hand over a frame as soon as it is read; frames come in order."""
        index = int(frameIndex)
        self._latest = (index, rgb)
        if self._active:
            self._jobs[index] = self._executor.submit(self._step, rgb)

    def mark(self, frameIndex: int, trusted: bool) -> None:
        """After the frame's decision: is it now the last trusted frame?"""
        if not self._lazy:
            return
        index = int(frameIndex)
        if self._latest is None or self._latest[0] != index:
            return
        if trusted:
            self._kept = self._latest
            if self._active:
                self._active = False
                self._jobs = {}
            return
        if self._active or self._kept is None:
            return
        # The first frame in doubt: the chain starts at the last trusted frame.
        anchor, anchorRgb = self._kept
        self._anchor = anchor
        self._active = True
        self._jobs = {anchor: self._executor.submit(self._start, anchor, anchorRgb)}
        self._jobs[index] = self._executor.submit(self._step, self._latest[1])

    def rotation(self, start: int, end: int) -> np.ndarray:
        """The rotation taking directions of frame ``start`` to frame ``end``."""
        start, end = int(start), int(end)
        if start not in self._jobs or end not in self._jobs:
            return np.eye(3)  # no chain from that frame: leave the place where it is
        return self._jobs[end].result() @ self._jobs[start].result().T

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    # ---------------------------------------------------------------- worker thread
    def _start(self, anchor: int, rgb: np.ndarray) -> np.ndarray:
        self._rng = np.random.default_rng(self._seed + anchor)
        self._orientation = np.eye(3)
        self._previous = self._describe(rgb)
        self.measuredFrames += 1
        return self._orientation.copy()

    def _step(self, rgb: np.ndarray) -> np.ndarray:
        current = self._describe(rgb)
        self.measuredFrames += 1
        if self._previous is not None:
            pairs = cr.matchedDirections(self._previous, current, self._matcher)
            step = None if pairs is None else cr.fitRotation(*pairs, self._rng, self._iterations)
            if step is None or not step.reliable:
                self.unreliableSteps += 1
            else:
                self._orientation = step.matrix @ self._orientation
        self._previous = current
        return self._orientation.copy()

    def _describe(self, rgb: np.ndarray) -> tuple:
        gray = cr.downsample(rgb, self._width, self._fast)
        keys, desc = cr._features(gray)
        return (keys, desc, gray.shape)
