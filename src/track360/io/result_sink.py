"""Atomic text sink for frame-ordered tracking results."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from track360.core.errors import OutputError, ProtocolError
from track360.core.protocols import ResultSink as ResultSinkProtocol
from track360.core.types import TrackResult
from track360.io.result_writer import TextResultWriter


@dataclass(slots=True)
class FileResultSink(ResultSinkProtocol):
    """Write one line per frame and publish the file atomically."""

    writer: TextResultWriter = field(default_factory=TextResultWriter)
    _destination: Path | None = field(init=False, default=None, repr=False)
    _partialPath: Path | None = field(init=False, default=None, repr=False)
    _stream: object | None = field(init=False, default=None, repr=False)
    _lastFrameIndex: int = field(init=False, default=-1, repr=False)
    _sequenceId: str | None = field(init=False, default=None, repr=False)
    _count: int = field(init=False, default=0, repr=False)

    def open(self, destination: str) -> None:
        if self._stream is not None:
            raise ProtocolError("result sink is already open")
        outputPath = Path(destination).expanduser().resolve()
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        partialPath = outputPath.with_name(outputPath.name + ".partial")
        self._destination = outputPath
        self._partialPath = partialPath
        self._stream = partialPath.open("w", encoding="utf-8", newline="\n")
        self._lastFrameIndex = -1
        self._sequenceId = None
        self._count = 0

    def write(self, result: TrackResult) -> None:
        if self._stream is None or self._partialPath is None:
            raise ProtocolError("result sink is not open")
        frameIndex = int(result.frameIndex)
        if self._count == 0 and frameIndex != 0:
            raise OutputError("result stream must start at frame 0")
        if frameIndex != self._lastFrameIndex + 1:
            raise OutputError(
                f"result frame order mismatch: expected={self._lastFrameIndex + 1}, "
                f"actual={frameIndex}"
            )
        if self._sequenceId is None:
            self._sequenceId = str(result.sequenceId)
        elif self._sequenceId != str(result.sequenceId):
            raise OutputError("result sink cannot mix sequenceIds")
        self._stream.write(self.writer.formatResult(result))
        self._stream.write("\n")
        self._lastFrameIndex = frameIndex
        self._count += 1

    def finalize(self, expectedFrameCount: int) -> None:
        if self._stream is None or self._partialPath is None or self._destination is None:
            raise ProtocolError("result sink is not open")
        actualCount = self._count
        if expectedFrameCount != self._count:
            self.close()
            raise OutputError(
                f"result count mismatch: expected={expectedFrameCount}, actual={actualCount}"
            )
        try:
            self._stream.flush()
            os.fsync(self._stream.fileno())
        finally:
            self._closeStream()
        os.replace(self._partialPath, self._destination)
        self._partialPath = None
        self._destination = None

    def close(self) -> None:
        self._closeStream()
        self._partialPath = None
        self._destination = None
        self._lastFrameIndex = -1
        self._sequenceId = None
        self._count = 0

    def _closeStream(self) -> None:
        if self._stream is not None:
            self._stream.close()
        self._stream = None


ResultSink = FileResultSink

__all__ = ["FileResultSink", "ResultSink"]
