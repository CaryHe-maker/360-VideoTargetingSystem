import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from track360.geometry.camera_rotation import rotateErp
from track360.runtime.camera_motion import CameraMotion


def _scene() -> np.ndarray:
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (192, 384, 3), dtype=np.uint8)
    texture = cv2.resize(noise, (1536, 768), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(texture, (3, 3), 0)


def _angle(rotation: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(rotation).magnitude()))


class CameraMotionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = _scene()
        step = Rotation.from_euler("y", 2.0, degrees=True).as_matrix()
        self.frames = [rotateErp(self.scene, np.linalg.matrix_power(step, k)) for k in range(6)]

    def testNothingIsMeasuredWhileTrustedAndTheChainStartsAtTheLastTrustedFrame(self) -> None:
        camera = CameraMotion(workWidth=384)
        try:
            for index in range(3):
                camera.submit(index, self.frames[index])
                camera.mark(index, True)
            self.assertEqual(camera.measuredFrames, 0)
            # Frames 3 and 4 are in doubt: the chain starts at frame 2.
            for index in (3, 4):
                camera.submit(index, self.frames[index])
                camera.mark(index, False)
            turned = camera.rotation(2, 4)
            self.assertAlmostEqual(_angle(turned), 4.0, delta=0.3)
            self.assertEqual(camera.measuredFrames, 3)
            # Unknown start: no rotation.
            np.testing.assert_allclose(camera.rotation(1, 4), np.eye(3))
            # Trusted again: the chain is dropped.
            camera.submit(5, self.frames[5])
            camera.mark(5, True)
            np.testing.assert_allclose(camera.rotation(2, 4), np.eye(3))
        finally:
            camera.close()

    def testEagerModeMeasuresEveryFrame(self) -> None:
        camera = CameraMotion(workWidth=384, lazy=False, fast=False, iterations=300)
        try:
            for index in range(4):
                camera.submit(index, self.frames[index])
                camera.mark(index, True)
            self.assertAlmostEqual(_angle(camera.rotation(0, 3)), 6.0, delta=0.3)
            self.assertEqual(camera.measuredFrames, 4)
        finally:
            camera.close()


if __name__ == "__main__":
    unittest.main()
