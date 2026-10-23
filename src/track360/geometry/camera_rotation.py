"""Camera rotation between two equirectangular frames (experimental, V2.1, E053).

A 360-degree camera that turns moves every point of the image the same way on the
sphere.  The rotation is estimated from ORB features matched between downsampled
frames: each matched pair gives two unit vectors, and the rotation is fitted to the
pairs by RANSAC on two-pair samples and refined by the Kabsch method on the inliers.
Moving objects and parallax of near objects are outliers.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_WIDTH = 1024
INLIER_DEG = 0.6
ITERATIONS = 300
# Offline studies (tools/camera_track.py) set these; the tracker passes its own.
FAST_DOWNSAMPLE = False


@dataclass(frozen=True, slots=True)
class RotationEstimate:
    """``matrix`` takes a unit vector of the first frame to the second frame."""

    matrix: np.ndarray
    inliers: int
    matches: int

    @property
    def angleDeg(self) -> float:
        cosine = (np.trace(self.matrix) - 1.0) / 2.0
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    @property
    def reliable(self) -> bool:
        return self.inliers >= 30 and self.inliers >= 0.25 * self.matches


def pixelsToUnit(x: np.ndarray, y: np.ndarray, width: int, height: int) -> np.ndarray:
    """Unit vectors of ERP pixel centres, in the project's yaw/pitch convention."""
    yaw = (x + 0.5) / width * 2.0 * np.pi - np.pi
    pitch = np.pi / 2.0 - (y + 0.5) / height * np.pi
    return np.stack(
        [np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)], axis=-1
    )


def downsample(rgb: np.ndarray, width: int, fast: bool = False) -> np.ndarray:
    """A grey image ``width`` pixels wide.

    ``fast`` takes every n-th pixel of the green channel down to about twice the width,
    then averages 2 x 2: a fraction of a millisecond instead of about four for a
    3840-pixel frame, at the cost of some aliasing.
    """
    height = width // 2
    if fast:
        step = max(1, rgb.shape[1] // (2 * width))
        green = np.ascontiguousarray(rgb[::step, ::step, 1])
        return cv2.resize(green, (width, height), interpolation=cv2.INTER_AREA)
    small = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


def _gray(rgb: np.ndarray) -> np.ndarray:
    return downsample(rgb, WORK_WIDTH, FAST_DOWNSAMPLE)


_ORB = None


def _features(gray: np.ndarray):
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(nfeatures=3000, fastThreshold=10)
    orb = _ORB
    # Leave out the polar caps, where the projection stretches the image most.
    mask = np.zeros_like(gray)
    mask[gray.shape[0] // 8 : gray.shape[0] * 7 // 8] = 255
    return orb.detectAndCompute(gray, mask)


def _kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The rotation R that best maps rows of ``a`` onto rows of ``b`` (b ~ a R^T)."""
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def fitRotation(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, iterations: int | None = None
) -> RotationEstimate | None:
    """RANSAC on two-pair samples, all hypotheses at once, then Kabsch on the inliers.

    Vectorised so that the work stays in NumPy and other threads keep running.
    """
    if len(a) < 3:
        return None
    count = ITERATIONS if iterations is None else int(iterations)
    limit = np.cos(np.radians(INLIER_DEG))
    first = rng.integers(0, len(a), count)
    second = (first + rng.integers(1, len(a), count)) % len(a)  # never the same pair
    picks = np.stack([first, second], axis=1)  # hypotheses x 2
    h = np.einsum("kpi,kpj->kij", a[picks], b[picks])
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.transpose(vt, (0, 2, 1)) @ np.transpose(u, (0, 2, 1))))
    fix = np.ones((count, 3))
    fix[:, 2] = d
    rotations = np.transpose(vt, (0, 2, 1)) @ (fix[:, :, None] * np.transpose(u, (0, 2, 1)))
    scores = np.einsum("kij,nj,ni->kn", rotations, a, b) > limit
    best = scores[int(np.argmax(scores.sum(1)))]
    if best.sum() < 3:
        return None
    rotation = _kabsch(a[best], b[best])
    inliers = np.einsum("ij,ij->i", a @ rotation.T, b) > limit
    return RotationEstimate(rotation, int(inliers.sum()), len(a))


def matchedDirections(previous, current, matcher) -> tuple[np.ndarray, np.ndarray] | None:
    """Unit vectors of the features matched between two frames' ``(keys, descriptors, shape)``."""
    (k1, d1, shape), (k2, d2, _) = previous, current
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None
    pairs = matcher.knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < 10:
        return None
    height, width = shape
    q = np.fromiter((m.queryIdx for m in good), dtype=np.int64, count=len(good))
    r = np.fromiter((m.trainIdx for m in good), dtype=np.int64, count=len(good))
    p1 = cv2.KeyPoint_convert(k1)[q]
    p2 = cv2.KeyPoint_convert(k2)[r]
    a = pixelsToUnit(p1[:, 0] - 0.5, p1[:, 1] - 0.5, width, height)
    b = pixelsToUnit(p2[:, 0] - 0.5, p2[:, 1] - 0.5, width, height)
    return a, b


def estimateRotation(
    first: np.ndarray, second: np.ndarray, *, seed: int = 0
) -> RotationEstimate | None:
    """The camera rotation from ``first`` to ``second`` (RGB ERP frames)."""
    g1, g2 = _gray(first), _gray(second)
    k1, d1 = _features(g1)
    k2, d2 = _features(g2)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < 10:
        return None
    width, height = g1.shape[1], g1.shape[0]
    p1 = np.asarray([k1[m.queryIdx].pt for m in good])
    p2 = np.asarray([k2[m.trainIdx].pt for m in good])
    a = pixelsToUnit(p1[:, 0] - 0.5, p1[:, 1] - 0.5, width, height)
    b = pixelsToUnit(p2[:, 0] - 0.5, p2[:, 1] - 0.5, width, height)
    limit = np.cos(np.radians(INLIER_DEG))
    rng = np.random.default_rng(seed)
    best = np.zeros(len(a), dtype=bool)
    for _ in range(ITERATIONS):
        pick = rng.choice(len(a), 2, replace=False)
        rotation = _kabsch(a[pick], b[pick])
        inliers = np.einsum("ij,ij->i", a @ rotation.T, b) > limit
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 3:
        return None
    rotation = _kabsch(a[best], b[best])
    inliers = np.einsum("ij,ij->i", a @ rotation.T, b) > limit
    return RotationEstimate(rotation, int(inliers.sum()), len(a))


def rotateErp(rgb: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """The frame a camera turned by ``rotation`` would see (for checks and tests)."""
    height, width = rgb.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    # Pixel of the new frame shows the direction R^T v of the old frame.
    v = pixelsToUnit(x, y, width, height) @ rotation
    yaw = np.arctan2(v[..., 0], v[..., 2])
    pitch = np.arcsin(np.clip(v[..., 1], -1.0, 1.0))
    mapX = ((yaw + np.pi) / (2.0 * np.pi) * width - 0.5).astype(np.float32)
    mapY = ((np.pi / 2.0 - pitch) / np.pi * height - 0.5).astype(np.float32)
    return cv2.remap(rgb, mapX, mapY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
