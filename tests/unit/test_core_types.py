import unittest

import numpy as np

from track360.core.errors import ProtocolError
from track360.core.types import (
    BBoxXYWH,
    FrameIndex,
    FramePacket,
    SequenceId,
    SphericalPoint,
)


class CoreTypesTest(unittest.TestCase):
    def testFramePacketAcceptsRgb(self) -> None:
        rgb = np.zeros((4, 8, 3), dtype=np.uint8)

        packet = FramePacket(
            sequenceId=SequenceId("sequence"),
            frameIndex=FrameIndex(0),
            timestampNs=0,
            rgb=rgb,
        )

        self.assertIs(packet.rgb, rgb)

    def testSphericalPointRejectsNonUnitVector(self) -> None:
        with self.assertRaises(ProtocolError):
            SphericalPoint(x=2.0, y=0.0, z=0.0, yawRad=0.0, pitchRad=0.0)

    def testBoundingBoxRejectsNonPositiveDimensions(self) -> None:
        with self.assertRaises(ProtocolError):
            BBoxXYWH(xPx=0.0, yPx=0.0, widthPx=0.0, heightPx=1.0)


if __name__ == "__main__":
    unittest.main()
