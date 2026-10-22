"""The camera's orientation over a sequence, from frame-to-frame rotations (E053).

    python tools/camera_track.py --dataset-root <dir> --label-root <dir> \\
        --sequence-file configs/splits/360vos_dev_train.txt --out outputs/E053/camera --shard 0/3

For every frame, ORB features are matched with the previous frame and the rotation
between them estimated (``track360.geometry.camera_rotation``).  ``orientation[t]``
takes a direction seen in frame 0 to frame t, so the rotation from frame a to frame b
is ``orientation[b] @ orientation[a].T``.  A step with too few inliers is taken as no
rotation and flagged.  One archive per sequence.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.geometry import camera_rotation as cr  # noqa: E402


def track(sequence, count: int, stride: int = 1) -> dict[str, np.ndarray]:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    orientation = np.zeros((count, 3, 3))
    orientation[0] = np.eye(3)
    inliers = np.zeros(count, dtype=np.int32)
    matches = np.zeros(count, dtype=np.int32)
    reliable = np.ones(count, dtype=bool)
    previous = None
    rng = np.random.default_rng(0)
    for index in range(count):
        if index % stride:
            orientation[index] = orientation[index - 1]
            continue
        gray = cr._gray(sequence.readRgb(index))
        keys, desc = cr._features(gray)
        current = (keys, desc, gray.shape)
        if previous is not None:
            step = _step(previous, current, matcher, rng)
            if step is None or not step.reliable:
                orientation[index] = orientation[index - stride]
                reliable[index] = False
                if step is not None:
                    inliers[index], matches[index] = step.inliers, step.matches
            else:
                orientation[index] = step.matrix @ orientation[index - stride]
                inliers[index], matches[index] = step.inliers, step.matches
        previous = current
    return {
        "orientation": orientation,
        "inliers": inliers,
        "matches": matches,
        "reliable": reliable,
    }


def _step(previous, current, matcher, rng):
    (k1, d1, shape), (k2, d2, _) = previous, current
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None
    pairs = matcher.knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < 10:
        return None
    height, width = shape
    p1 = np.asarray([k1[m.queryIdx].pt for m in good])
    p2 = np.asarray([k2[m.trainIdx].pt for m in good])
    a = cr.pixelsToUnit(p1[:, 0] - 0.5, p1[:, 1] - 0.5, width, height)
    b = cr.pixelsToUnit(p2[:, 0] - 0.5, p2[:, 1] - 0.5, width, height)
    return cr.fitRotation(a, b, rng)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--width", type=int, default=cr.WORK_WIDTH, help="working width")
    parser.add_argument("--stride", type=int, default=1, help="measure every n-th frame")
    args = parser.parse_args(argv)
    cr.WORK_WIDTH = args.width
    index, count = (int(x) for x in args.shard.split("/"))
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    names = readSequenceFile(args.sequence_file)[index::count]
    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = args.out / f"{name}.npz"
        if path.exists():
            continue
        started = time.perf_counter()
        sequence = dataset.sequence(name)
        result = track(sequence, sequence.frameCount, args.stride)
        sequence.close()
        np.savez_compressed(path, **result)
        turned = np.degrees(
            np.arccos(np.clip((np.trace(result["orientation"][-1]) - 1) / 2, -1, 1))
        )
        print(
            f"{name}: {len(result['reliable'])} frames, unreliable steps"
            f" {int((~result['reliable']).sum())}, net turn {turned:.1f} deg,"
            f" {time.perf_counter() - started:.0f} s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
