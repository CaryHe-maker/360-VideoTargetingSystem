import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from track360.core.errors import ProtocolError
from track360.runtime.gpu_lock import exclusiveGpuRun, isDeviceFailure

HOLD_LOCK = """
import sys, time
from track360.runtime.gpu_lock import exclusiveGpuRun
with exclusiveGpuRun(sys.argv[1]):
    print("held", flush=True)
    time.sleep(30)
"""


class GpuLockTest(unittest.TestCase):
    def testASecondRunIsRefusedWhileTheFirstHoldsTheLock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.lock"
            holder = subprocess.Popen(
                [sys.executable, "-c", HOLD_LOCK, str(path)],
                stdout=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
            )
            try:
                self.assertEqual(holder.stdout.readline().strip(), "held")
                with self.assertRaisesRegex(ProtocolError, "GPU lock"):
                    with exclusiveGpuRun(path):
                        pass
            finally:
                holder.kill()
                holder.wait()
                holder.stdout.close()
            # A lock left by a process that ended is free again (Windows may take a
            # moment to release it after the process is gone).
            for _ in range(50):
                try:
                    with exclusiveGpuRun(path):
                        break
                except ProtocolError:
                    time.sleep(0.1)
            else:
                self.fail("the lock of an ended process was not released")

    def testTheLockIsReleasedAfterTheBlock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.lock"
            with exclusiveGpuRun(path):
                pass
            with exclusiveGpuRun(path):
                pass


class DeviceFailureTest(unittest.TestCase):
    def testGpuErrorsAreRecognisedThroughTheCauseChain(self) -> None:
        try:
            try:
                raise RuntimeError("CUDA error: out of memory")
            except RuntimeError as inner:
                raise ValueError("ARTrackV2 batch inference failed") from inner
        except ValueError as outer:
            self.assertTrue(isDeviceFailure(outer))
        self.assertTrue(isDeviceFailure(RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED")))

    def testOtherErrorsAreNot(self) -> None:
        self.assertFalse(isDeviceFailure(ValueError("box left the view")))
        self.assertFalse(isDeviceFailure(ProtocolError("frame order")))


if __name__ == "__main__":
    unittest.main()
