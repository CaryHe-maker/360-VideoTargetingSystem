"""Generic frame source for directories, single images, and simple videos."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from shutil import which

import numpy as np

from track360.core.errors import DecodeError, ProtocolError
from track360.core.protocols import FrameSource as FrameSourceProtocol
from track360.core.types import FrameIndex, FramePacket, SequenceId
from track360.io.image_reader import RGB_IMAGE_EXTENSIONS, readRgbImage

SUPPORTED_IMAGE_EXTENSIONS = RGB_IMAGE_EXTENSIONS
SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm"})


@dataclass(slots=True)
class VideoFrameSource(FrameSourceProtocol):
    """Read a strictly ordered frame stream from a directory or file."""

    recursive: bool = False
    sequenceId: str | None = None
    supportedExtensions: frozenset[str] = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS
    _root: Path | None = field(init=False, default=None, repr=False)
    _paths: tuple[Path, ...] = field(init=False, default=(), repr=False)
    _frames: tuple[np.ndarray, ...] = field(init=False, default=(), repr=False)
    _cursor: int = field(init=False, default=0, repr=False)
    _resolvedSequenceId: str = field(init=False, default="sequence", repr=False)
    _videoProcess: subprocess.Popen[bytes] | None = field(init=False, default=None, repr=False)
    _videoWidthPx: int = field(init=False, default=0, repr=False)
    _videoHeightPx: int = field(init=False, default=0, repr=False)
    _frameIntervalNs: int = field(init=False, default=33_333_333, repr=False)

    @property
    def frameCount(self) -> int:
        if self._root is None:
            return 0
        if self._frames:
            return len(self._frames)
        if self._paths:
            return len(self._paths)
        return -1

    def open(self, uri: str) -> None:
        root = Path(uri).expanduser().resolve()
        if not root.exists():
            raise DecodeError(f"input path does not exist: {root}")

        self.close()
        if root.is_dir():
            self._openDirectory(root)
        elif root.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            self._openVideo(root)
        else:
            self._openImageSequence(root)
        self._root = root
        self._resolvedSequenceId = self.sequenceId or root.stem or root.name or "sequence"

    def read(self) -> FramePacket | None:
        if self._root is None:
            raise ProtocolError("frame source is not open")
        if self._frames:
            return self._readCachedFrame()
        if self._videoProcess is not None:
            return self._readVideoFrame()
        return self._readDirectoryFrame()

    def close(self) -> None:
        if self._videoProcess is not None:
            try:
                if self._videoProcess.stdout is not None:
                    self._videoProcess.stdout.close()
                if self._videoProcess.stderr is not None:
                    self._videoProcess.stderr.close()
            finally:
                try:
                    self._videoProcess.terminate()
                    self._videoProcess.wait(timeout=1)
                except Exception:
                    try:
                        self._videoProcess.kill()
                        self._videoProcess.wait(timeout=1)
                    except Exception:
                        pass
        self._root = None
        self._paths = ()
        self._frames = ()
        self._cursor = 0
        self._resolvedSequenceId = "sequence"
        self._videoProcess = None
        self._videoWidthPx = 0
        self._videoHeightPx = 0
        self._frameIntervalNs = 33_333_333

    def _openDirectory(self, root: Path) -> None:
        iterator = root.rglob("*") if self.recursive else root.iterdir()
        paths = tuple(
            path
            for path in sorted(iterator, key=_sortKey(root))
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if not paths:
            raise DecodeError(f"no readable images found in {root}")
        self._paths = paths
        self._cursor = 0

    def _openImageSequence(self, path: Path) -> None:
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise DecodeError(f"unsupported frame source: {path.suffix}")
        self._frames = (readRgbImage(path),)
        if not self._frames:
            raise DecodeError(f"no readable frames found in {path}")
        self._cursor = 0

    def _openVideo(self, path: Path) -> None:
        if which("ffmpeg") is None or which("ffprobe") is None:
            raise DecodeError(f"video decoding requires ffmpeg/ffprobe: {path}")
        try:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,r_frame_rate",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                check=True,
                text=True,
            )
            payload = json.loads(probe.stdout)
            stream = payload["streams"][0]
            self._videoWidthPx = int(stream["width"])
            self._videoHeightPx = int(stream["height"])
            fpsText = str(stream.get("r_frame_rate", "30/1"))
            numerator, denominator = (int(part) for part in fpsText.split("/", 1))
            fps = numerator / max(denominator, 1)
            self._frameIntervalNs = int(round(1_000_000_000 / max(fps, 1e-6)))
            self._videoProcess = subprocess.Popen(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-an",
                    "-sn",
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (
            OSError,
            subprocess.CalledProcessError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ) as error:
            raise DecodeError(f"cannot open video source {path}: {error}") from error

    def _readDirectoryFrame(self) -> FramePacket | None:
        if self._cursor >= len(self._paths):
            return None
        path = self._paths[self._cursor]
        rgb = readRgbImage(path)
        frame = self._makeFrame(rgb)
        self._cursor += 1
        return frame

    def _readCachedFrame(self) -> FramePacket | None:
        if self._cursor >= len(self._frames):
            return None
        rgb = self._frames[self._cursor]
        frame = self._makeFrame(rgb)
        self._cursor += 1
        return frame

    def _readVideoFrame(self) -> FramePacket | None:
        assert self._videoProcess is not None
        assert self._videoProcess.stdout is not None
        frameSize = self._videoWidthPx * self._videoHeightPx * 3
        payload = self._videoProcess.stdout.read(frameSize)
        if len(payload) != frameSize:
            self.close()
            return None
        rgb = np.frombuffer(payload, dtype=np.uint8).reshape(
            self._videoHeightPx,
            self._videoWidthPx,
            3,
        ).copy()
        frame = self._makeFrame(rgb)
        self._cursor += 1
        return frame

    def _makeFrame(self, rgb: np.ndarray) -> FramePacket:
        return FramePacket(
            sequenceId=SequenceId(self._resolvedSequenceId),
            frameIndex=FrameIndex(self._cursor),
            timestampNs=self._cursor * self._frameIntervalNs,
            rgb=rgb,
        )


def _sortKey(root: Path):
    def sortPath(path: Path) -> str:
        return path.relative_to(root).as_posix().lower()

    return sortPath


FrameSource = VideoFrameSource

__all__ = [
    "FrameSource",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "VideoFrameSource",
]
