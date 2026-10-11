import unittest

import numpy as np

from track360.core.errors import DecodeError
from track360.core.types import FrameIndex, FramePacket, SequenceId
from track360.runtime.driver import _PrefetchReader


def _frame(index: int) -> FramePacket:
    return FramePacket(
        SequenceId("pipeline"),
        FrameIndex(index),
        index,
        np.full((4, 8, 3), index, dtype=np.uint8),
    )


class PipelineIntegrationTest(unittest.TestCase):
    def testPrefetchPropagatesWorkerFailureAfterReadyFrames(self) -> None:
        class _Source:
            def __init__(self) -> None:
                self.index = 0

            def read(self):
                if self.index == 0:
                    self.index += 1
                    return _frame(0)
                raise DecodeError("injected decode failure")

        reader = _PrefetchReader(_Source())
        reader.start()
        try:
            self.assertEqual(reader.read().frameIndex, FrameIndex(0))
            with self.assertRaisesRegex(DecodeError, "injected decode failure"):
                reader.read()
        finally:
            reader.close()

    def testPrefetchCannotStartTwiceAndCloseIsIdempotent(self) -> None:
        class _EmptySource:
            def read(self):
                return None

        reader = _PrefetchReader(_EmptySource())
        reader.start()
        with self.assertRaisesRegex(RuntimeError, "already started"):
            reader.start()
        self.assertIsNone(reader.read())
        reader.close()
        reader.close()
        self.assertIsNone(reader._thread)

    def testPrefetchProfilesEveryFrameWithoutChangingOrder(self) -> None:
        frames = [_frame(index) for index in range(5)]

        class _Source:
            def read(self):
                return frames.pop(0) if frames else None

        reader = _PrefetchReader(_Source())
        reader.start()
        returned = []
        profiles = []
        try:
            while True:
                frame = reader.read()
                if frame is None:
                    break
                returned.append(int(frame.frameIndex))
                profiles.append(reader.lastProfile)
        finally:
            reader.close()

        self.assertEqual(returned, [0, 1, 2, 3, 4])
        self.assertEqual(
            [int(profile["pipelineFrameIndex"]) for profile in profiles],
            returned,
        )
        self.assertTrue(all(profile["pipelinePrefetchEnabled"] for profile in profiles))


if __name__ == "__main__":
    unittest.main()
