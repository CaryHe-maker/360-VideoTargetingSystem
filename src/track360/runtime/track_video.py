"""``track360 track``: track one target in a video or image sequence."""

from __future__ import annotations

import argparse
import sys
from math import radians

from track360.core.config import loadConfig
from track360.core.errors import (
    ConfigError,
    DecodeError,
    GeometryError,
    ModelError,
    OutputError,
    ProtocolError,
    Track360Error,
)
from track360.core.types import BBoxXYWH, BFoV
from track360.datasets.frame_source import FrameSource
from track360.geometry.projection_math import makeSphericalPoint
from track360.runtime.driver import (
    buildRuntime,
    closeRuntime,
    finalizeSink,
    openSink,
    runTracking,
)
from track360.runtime.reproducibility import writeRunMetadata

EXIT_CONFIG = 2
EXIT_DECODE = 3
EXIT_MODEL = 4
EXIT_OUTPUT = 5
EXIT_INVARIANT = 10


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="track360 track")
    parser.add_argument("--input", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--init-box", help="frame-0 ERP pixel box: x,y,width,height")
    target.add_argument(
        "--init-bfov",
        help="frame-0 spherical box in degrees: clon,clat,fov_h,fov_v (360VOT convention)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sequence-id", default=None)
    parser.add_argument("--recursive", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = buildParser().parse_args(argv)
    runtime = None
    source = None
    try:
        config = loadConfig(args.config)
        runtime = buildRuntime(
            config, allowUncalibratedScoring=config.scoring.calibrationArtifact is None
        )
        source = FrameSource(recursive=args.recursive, sequenceId=args.sequence_id)
        source.open(args.input)
        openSink(runtime.sink, args.output)
        initialBox = _parseBox(args.init_box) if args.init_box is not None else None
        initialBfov = _parseBfov(args.init_bfov) if args.init_bfov is not None else None
        resultCount = runTracking(
            source=source,
            initialBox=initialBox,
            initialBfov=initialBfov,
            geometry=runtime.geometry,
            controller=runtime.controller,
            backend=runtime.backend,
            sink=runtime.sink,
            recorder=runtime.recorder,
            scoreCalibration=runtime.scoreCalibration,
            useMotionScore=runtime.useMotionScore,
        )
        expectedCount = resultCount if getattr(source, "frameCount", 0) <= 0 else source.frameCount
        finalizeSink(runtime.sink, expectedCount)
        writeRunMetadata(args.output, config)
        return 0
    except ConfigError as error:
        _report(error)
        return EXIT_CONFIG
    except DecodeError as error:
        _report(error)
        return EXIT_DECODE
    except ModelError as error:
        _report(error)
        return EXIT_MODEL
    except OutputError as error:
        _report(error)
        return EXIT_OUTPUT
    except (GeometryError, ProtocolError, Track360Error) as error:
        _report(error)
        return EXIT_INVARIANT
    except Exception as error:
        _report(error)
        return EXIT_INVARIANT
    finally:
        if runtime is not None:
            try:
                closeRuntime(runtime)
            except Exception:
                pass
        if source is not None:
            try:
                source.close()
            except Exception:
                pass


def _parseBox(text: str) -> BBoxXYWH:
    try:
        xPx, yPx, widthPx, heightPx = (float(part.strip()) for part in text.split(","))
    except ValueError as error:
        raise ConfigError("--init-box must contain four comma-separated numbers") from error
    return BBoxXYWH(xPx=xPx, yPx=yPx, widthPx=widthPx, heightPx=heightPx)


def _parseBfov(text: str) -> BFoV:
    try:
        clon, clat, fovH, fovV = (float(part.strip()) for part in text.split(","))
    except ValueError as error:
        raise ConfigError("--init-bfov must contain four comma-separated numbers") from error
    if not (-180.0 <= clon <= 180.0 and -90.0 <= clat <= 90.0):
        raise ConfigError("--init-bfov center must satisfy -180<=clon<=180 and -90<=clat<=90")
    if not (0.0 < fovH < 180.0 and 0.0 < fovV < 180.0):
        raise ConfigError("--init-bfov field of view must be in (0, 180) degrees")
    return BFoV(
        center=makeSphericalPoint(radians(clon), radians(clat)),
        horizontalFovRad=radians(fovH),
        verticalFovRad=radians(fovV),
    )


def _report(error: Exception) -> None:
    print(f"{type(error).__name__}: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
