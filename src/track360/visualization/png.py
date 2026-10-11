"""Minimal lossless RGB PNG output without an image-library dependency."""

from __future__ import annotations

import binascii
import os
import struct
import tempfile
import zlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from track360.core.errors import OutputError, ProtocolError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def writeRgbPng(path: str | Path, rgb: NDArray[np.uint8]) -> Path:
    """Atomically write an RGB uint8 HWC array as a lossless PNG."""
    _requireRgb(rgb)
    outputPath = Path(path)
    try:
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        contiguousRgb = np.ascontiguousarray(rgb)
        scanlines = b"".join(b"\x00" + row.tobytes() for row in contiguousRgb)
        header = struct.pack(">IIBBBBB", rgb.shape[1], rgb.shape[0], 8, 2, 0, 0, 0)
        payload = (
            PNG_SIGNATURE
            + _pngChunk(b"IHDR", header)
            + _pngChunk(b"IDAT", zlib.compress(scanlines))
            + _pngChunk(b"IEND", b"")
        )
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=outputPath.parent,
            prefix=f".{outputPath.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporaryPath = Path(stream.name)
            stream.write(payload)
        os.replace(temporaryPath, outputPath)
    except OSError as error:
        if "temporaryPath" in locals():
            temporaryPath.unlink(missing_ok=True)
        raise OutputError(f"cannot write visualization PNG {outputPath}: {error}") from error
    return outputPath


def _pngChunk(kind: bytes, data: bytes) -> bytes:
    checksum = binascii.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _requireRgb(rgb: NDArray[np.uint8]) -> None:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ProtocolError("visualization RGB must be a uint8 NumPy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0:
        raise ProtocolError(
            f"visualization RGB must have non-empty shape [H, W, 3], actual={rgb.shape}"
        )


__all__ = ["writeRgbPng"]
