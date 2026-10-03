"""Image decoding helpers.

RGB frames are decoded with OpenCV.  ``readImageArray`` keeps the exact PNG decoder
used for label planes such as AirSim360 segmentation, where channel layout and bit
depth must be preserved.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from track360.core.errors import DecodeError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
RGB_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})


def readImageArray(path: str | Path) -> np.ndarray:
    """Read a small image as a NumPy array."""
    imagePath = Path(path)
    if imagePath.suffix.lower() == ".npy":
        try:
            return np.load(imagePath)
        except OSError as error:
            raise DecodeError(f"cannot read NumPy image {imagePath}: {error}") from error
    if imagePath.suffix.lower() == ".png":
        return _readPng(imagePath)
    raise DecodeError(f"unsupported image format: {imagePath.suffix}")


def readRgbImage(path: str | Path) -> np.ndarray:
    """Read one RGB image as a ``uint8`` HWC array."""
    imagePath = Path(path)
    if imagePath.suffix.lower() in RGB_IMAGE_EXTENSIONS:
        try:
            payload = imagePath.read_bytes()
        except OSError as error:
            raise DecodeError(f"cannot read image {imagePath}: {error}") from error
        return decodeRgbImage(payload, str(imagePath))
    array = readImageArray(path)
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=2).astype(np.uint8, copy=False)
    if array.ndim != 3:
        raise DecodeError(f"unsupported image rank: {array.shape}")
    if array.shape[2] == 3:
        return array.astype(np.uint8, copy=False)
    if array.shape[2] == 4:
        return array[..., :3].astype(np.uint8, copy=False)
    if array.shape[2] == 1:
        return np.repeat(array, 3, axis=2).astype(np.uint8, copy=False)
    raise DecodeError(f"unsupported image channel count: {array.shape}")


def decodeRgbImage(payload: bytes, name: str = "<bytes>") -> np.ndarray:
    """Decode an encoded PNG or JPEG payload to a ``uint8`` RGB HWC array."""
    import cv2

    buffer = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None
    if bgr is None:
        raise DecodeError(f"cannot decode image: {name}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _readPng(path: Path) -> np.ndarray:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise DecodeError(f"cannot read PNG image {path}: {error}") from error
    if not payload.startswith(PNG_SIGNATURE):
        raise DecodeError(f"unsupported PNG signature: {path}")
    width = height = bitDepth = colorType = None
    compressed = bytearray()
    offset = len(PNG_SIGNATURE)
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", data)
            width, height, bitDepth, colorType, compression, filterMethod, interlace = header
            if compression != 0 or filterMethod != 0 or interlace != 0:
                raise DecodeError(f"unsupported PNG compression/filter/interlace: {path}")
            if bitDepth != 8 or colorType not in {0, 2, 6}:
                raise DecodeError(
                    f"unsupported PNG format: bitDepth={bitDepth}, colorType={colorType}"
                )
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"IEND":
            break
    if width is None or height is None or bitDepth is None or colorType is None:
        raise DecodeError(f"invalid PNG file: {path}")
    channels = {0: 1, 2: 3, 6: 4}[colorType]
    rowBytes = width * channels
    raw = zlib.decompress(bytes(compressed))
    expected = height * (1 + rowBytes)
    if len(raw) != expected:
        raise DecodeError(f"unexpected PNG payload length: {path}")
    rows = np.zeros((height, rowBytes), dtype=np.uint8)
    previous = np.zeros(rowBytes, dtype=np.uint8)
    cursor = 0
    for rowIndex in range(height):
        filterType = raw[cursor]
        cursor += 1
        current = np.frombuffer(raw[cursor : cursor + rowBytes], dtype=np.uint8).copy()
        cursor += rowBytes
        rows[rowIndex] = _unfilter(filterType, current, previous, channels)
        previous = rows[rowIndex]
    if channels == 1:
        return rows.reshape(height, width)
    return rows.reshape(height, width, channels)


def _unfilter(
    filterType: int,
    current: np.ndarray,
    previous: np.ndarray,
    channels: int,
) -> np.ndarray:
    if filterType == 0:
        return current
    result = current.astype(np.uint16, copy=True)
    if filterType == 1:
        for index in range(channels, len(result)):
            result[index] = (result[index] + result[index - channels]) & 0xFF
    elif filterType == 2:
        result = (result + previous) & 0xFF
    elif filterType == 3:
        for index in range(len(result)):
            left = result[index - channels] if index >= channels else 0
            up = previous[index]
            result[index] = (result[index] + ((left + up) >> 1)) & 0xFF
    elif filterType == 4:
        for index in range(len(result)):
            left = result[index - channels] if index >= channels else 0
            up = previous[index]
            upLeft = previous[index - channels] if index >= channels else 0
            result[index] = (result[index] + _paeth(left, up, upLeft)) & 0xFF
    else:
        raise DecodeError(f"unsupported PNG filter type: {filterType}")
    return result.astype(np.uint8, copy=False)


def _paeth(left: int, up: int, upLeft: int) -> int:
    # PNG row buffers are uint8; widen before arithmetic to avoid wraparound
    # in the Paeth predictor on real RGBA AirSim360 masks.
    left, up, upLeft = int(left), int(up), int(upLeft)
    estimate = left + up - upLeft
    leftDistance = abs(estimate - left)
    upDistance = abs(estimate - up)
    upLeftDistance = abs(estimate - upLeft)
    if leftDistance <= upDistance and leftDistance <= upLeftDistance:
        return left
    if upDistance <= upLeftDistance:
        return up
    return upLeft


__all__ = ["readImageArray", "readRgbImage"]
