"""Derive 360VOT-style labels from 360VOS segmentation masks.

360VOS training sequences ship masks instead of boxes.  This turns the mask of the
first object into the ``bbox`` and ``bfov`` entries of a 360VOT ``label.json``, so a
training sequence can be tracked and scored exactly like a 360VOT one.

The boxes are fitted here, not by the authors' annotation pipeline, so they are not
bit-identical to official 360VOT labels; ``tools/prepare_tune_set.py --validate``
measures the difference on sequences that have both.  Rotated labels are not derived:
``rbbox`` and ``rbfov`` repeat the unrotated ones.
"""

from __future__ import annotations

import json
from math import degrees, pi
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from track360.core.errors import DecodeError
from track360.datasets.vot360 import LABEL_FILE, Vot360Sequence
from track360.geometry.seam import minimalCircularInterval
from track360.geometry.spherical_geometry import fitBfovToVectors

MASK_DIRECTORY = "mask"
# Masks use the DAVIS palette; the first object is RGB (128, 0, 0).
FIRST_OBJECT_RGB = (128, 0, 0)
MAX_BOUNDARY_POINTS = 4000
_ABSENT_BOX = {"cx": 0, "cy": 0, "w": 0, "h": 0, "rotation": 0}
_ABSENT_BFOV = {"clon": 0, "clat": 0, "fov_h": 0, "fov_v": 0, "rotation": 0}


def readTargetMask(sequence: Vot360Sequence, frameIndex: int) -> NDArray[np.bool_]:
    """Return the first object's mask of one frame."""
    import cv2

    stem = Path(sequence.frameNames[frameIndex]).stem
    member = f"{MASK_DIRECTORY}/{stem}.png"
    buffer = np.frombuffer(sequence.readMember(member), dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise DecodeError(f"cannot decode mask {sequence.name}/{member}")
    red, green, blue = FIRST_OBJECT_RGB
    return (bgr[..., 2] == red) & (bgr[..., 1] == green) & (bgr[..., 0] == blue)


def masksMatchFrames(sequence: Vot360Sequence) -> bool:
    """Whether every frame has a mask of the same name and no mask is left over."""
    frames = {Path(name).stem for name in sequence.frameNames}
    masks = {Path(name).stem for name in sequence.listMembers(MASK_DIRECTORY)}
    return frames == masks


def labelFromMask(mask: NDArray[np.bool_]) -> dict[str, dict[str, float]]:
    """Fit the label of one frame; an empty mask gives the absent-target label."""
    import cv2

    height, width = mask.shape
    if not mask.any():
        return {
            "bfov": dict(_ABSENT_BFOV),
            "rbfov": dict(_ABSENT_BFOV),
            "bbox": dict(_ABSENT_BOX),
            "rbbox": dict(_ABSENT_BOX),
        }
    columns = np.flatnonzero(mask.any(axis=0)).astype(np.float64)
    rows = np.flatnonzero(mask.any(axis=1))
    # The shortest circular run of occupied columns: a target on the seam stays whole.
    left, span = minimalCircularInterval(columns, width)
    boxWidth = span + 1.0
    top = float(rows[0])
    boxHeight = float(rows[-1] + 1 - rows[0])
    box = {
        # Pixel-index center; it runs past the right border for a seam-crossing target.
        "cx": left + boxWidth / 2.0 - 0.5,
        "cy": top + boxHeight / 2.0 - 0.5,
        "w": boxWidth,
        "h": boxHeight,
        "rotation": 0,
    }

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    points = np.concatenate([contour.reshape(-1, 2) for contour in contours])
    if len(points) > MAX_BOUNDARY_POINTS:
        points = points[:: len(points) // MAX_BOUNDARY_POINTS + 1]
    yaw = 2.0 * pi * (points[:, 0] + 0.5) / width - pi
    pitch = pi / 2.0 - pi * (points[:, 1] + 0.5) / height
    vectors = np.stack(
        (np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)), axis=1
    )
    fitted = fitBfovToVectors(vectors)
    bfov = {
        "clon": degrees(fitted.center.yawRad),
        "clat": degrees(fitted.center.pitchRad),
        "fov_h": degrees(fitted.horizontalFovRad),
        "fov_v": degrees(fitted.verticalFovRad),
        "rotation": 0,
    }
    return {"bfov": bfov, "rbfov": dict(bfov), "bbox": box, "rbbox": dict(box)}


def buildSequenceLabels(sequence: Vot360Sequence) -> dict[str, Any]:
    """Build the full ``label.json`` content of a mask-annotated sequence."""
    return {
        name: labelFromMask(readTargetMask(sequence, index))
        for index, name in enumerate(sequence.frameNames)
    }


def writeSequenceLabels(sequence: Vot360Sequence, labelRoot: str | Path) -> Path:
    """Write ``<labelRoot>/<sequence>/label.json`` and return its path."""
    if not masksMatchFrames(sequence):
        raise DecodeError(f"sequence {sequence.name} has frames and masks that do not pair up")
    labels = buildSequenceLabels(sequence)
    if not next(iter(labels.values()))["bbox"]["w"]:
        raise DecodeError(f"sequence {sequence.name} has no target in frame 0")
    path = Path(labelRoot) / sequence.name / LABEL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(labels), encoding="utf-8")
    partial.replace(path)
    return path


__all__ = [
    "FIRST_OBJECT_RGB",
    "buildSequenceLabels",
    "labelFromMask",
    "masksMatchFrames",
    "readTargetMask",
    "writeSequenceLabels",
]
