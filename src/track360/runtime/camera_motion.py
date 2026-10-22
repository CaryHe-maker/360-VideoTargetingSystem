"""The camera's rotation over a sequence, estimated while tracking (experimental, E053).

A turning 360-degree camera moves every point of the image the same way on the sphere,
so a place where the target was last trusted is no longer where it appears after the
camera turned.  Each frame is handed to ``submit`` as soon as it is read; a worker
thread matches its ORB features with the previous frame's and accumulates the
orientation (``track360.geometry.camera_rotation``), alongside the tracker's own work.
``rotation(a, b)`` waits for frame ``b`` and takes a direction seen in frame ``a`` to
frame ``b``.  A step with too few inliers counts as no rotation.  Results do not depend
on thread timing: every query waits for the frames it needs.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np

from track360.geometry import camera_rotation as cr

WORK_WIDTH = 512


class CameraMotion:
    def __init__(self, workWidth: int = WORK_WIDTH, seed: int = 0, stride: int = 1) -> None:
        self._width = int(workWidth)
        # Only every ``stride``-th frame is measured; the others take the last measured.
        self._stride = max(1, int(stride))
        self._measured: dict[int, int] = {}
        self._last: int | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="track360-camera")
        self._jobs: dict[int, Future[np.ndarray]] = {}
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._rng = np.random.default_rng(seed)
        self._previous: tuple | None = None
        self._orientation = np.eye(3)
        self.unreliableSteps = 0

    def submit(self, frameIndex: int, rgb: np.ndarray) -> None:
        """Queue a frame; frames must come in order, starting anywhere."""
        index = int(frameIndex)
        if self._last is None or index - self._last >= self._stride:
            self._jobs[index] = self._executor.submit(self._step, rgb)
            self._last = index
        self._measured[index] = self._last

    def orientation(self, frameIndex: int) -> np.ndarray:
        return self._jobs[self._measured[int(frameIndex)]].result()

    def rotation(self, start: int, end: int) -> np.ndarray:
        """The rotation taking directions of frame ``start`` to frame ``end``."""
        return self.orientation(end) @ self.orientation(start).T

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def _step(self, rgb: np.ndarray) -> np.ndarray:
        height = self._width // 2
        small = cv2.resize(rgb, (self._width, height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        keys, desc = cr._features(gray)
        current = (keys, desc, gray.shape)
        if self._previous is not None:
            step = self._estimate(self._previous, current)
            if step is None or not step.reliable:
                self.unreliableSteps += 1
            else:
                self._orientation = step.matrix @ self._orientation
        self._previous = current
        return self._orientation.copy()

    def _estimate(self, previous: tuple, current: tuple) -> cr.RotationEstimate | None:
        (k1, d1, shape), (k2, d2, _) = previous, current
        if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
            return None
        pairs = self._matcher.knnMatch(d1, d2, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if len(good) < 10:
            return None
        height, width = shape
        p1 = np.asarray([k1[m.queryIdx].pt for m in good])
        p2 = np.asarray([k2[m.trainIdx].pt for m in good])
        a = cr.pixelsToUnit(p1[:, 0] - 0.5, p1[:, 1] - 0.5, width, height)
        b = cr.pixelsToUnit(p2[:, 0] - 0.5, p2[:, 1] - 0.5, width, height)
        return cr.fitRotation(a, b, self._rng)
