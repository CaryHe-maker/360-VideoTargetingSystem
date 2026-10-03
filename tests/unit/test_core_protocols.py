import unittest

from track360.core.protocols import FrameSource, ResultSink


class _FrameSourceImplementation:
    def open(self, uri: str) -> None:
        self.uri = uri

    def read(self):
        return None

    def close(self) -> None:
        pass


class _ResultSinkImplementation:
    def open(self, destination: str) -> None:
        self.destination = destination

    def write(self, result) -> None:
        pass

    def finalize(self, expectedFrameCount: int) -> None:
        pass


class CoreProtocolsTest(unittest.TestCase):
    def testRuntimeCheckableIoProtocolsAcceptStructuralImplementations(self) -> None:
        self.assertIsInstance(_FrameSourceImplementation(), FrameSource)
        self.assertIsInstance(_ResultSinkImplementation(), ResultSink)


if __name__ == "__main__":
    unittest.main()
