"""One Track360 run on the GPU at a time.

Two runs started side by side on the development machine (8 GB GPU, 15 GB RAM) have
failed while the second one was loading its model: the first then reported a CUDA
allocation failure on every remaining frame, and finished as if nothing had happened
(evaluation log, E043 and E047).  Each run holds about 5 GB of committed host memory,
so the limit is the host, not the GPU.  A run takes this lock for its whole duration;
a second run started meanwhile stops at once with a clear message instead.

The lock is an operating-system file lock, so it is released when the process ends,
however it ends.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from track360.core.errors import ProtocolError

LOCK_NAME = "track360-gpu.lock"


@contextmanager
def exclusiveGpuRun(lockPath: str | Path | None = None) -> Iterator[Path]:
    """Hold the GPU lock while the block runs; raise ``ProtocolError`` if it is taken."""
    path = Path(lockPath) if lockPath is not None else Path(tempfile.gettempdir()) / LOCK_NAME
    stream = path.open("a+b")
    try:
        _lock(stream)
    except OSError:
        stream.close()
        raise ProtocolError(
            f"another Track360 run holds the GPU lock ({path}); "
            "runs share the machine's memory badly, so start this one when it has finished"
        ) from None
    try:
        yield path
    finally:
        try:
            _unlock(stream)
        finally:
            stream.close()


def _lock(stream) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(stream) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def isDeviceFailure(error: BaseException) -> bool:
    """True for an error of the GPU itself (out of memory, illegal access, cuBLAS).

    After such an error every later frame fails the same way, so a run must stop
    rather than emit placeholder results.  The cause chain is searched because the
    backend wraps the original error in its own.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__
        text = str(current)
        if name in {"OutOfMemoryError", "AcceleratorError"} or any(
            marker in text for marker in ("CUDA error", "CUDA out of memory", "CUBLAS_STATUS")
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


__all__ = ["LOCK_NAME", "exclusiveGpuRun", "isDeviceFailure"]
