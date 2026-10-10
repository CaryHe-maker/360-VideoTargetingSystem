"""``track360 track``: track one target in a video or image sequence."""

from __future__ import annotations

import argparse
import sys
from time import perf_counter

from track360.api import PRESETS, Track360Tracker, bfovFromDegrees, boxFromPixels, resolveConfig
from track360.core.config import MODEL_PRECISIONS
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
from track360.runtime.reproducibility import writeRunMetadata

EXIT_CONFIG = 2
EXIT_DECODE = 3
EXIT_MODEL = 4
EXIT_OUTPUT = 5
EXIT_INVARIANT = 10


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="track360 track",
        description="Track one target through a 360-degree video or a directory of frames.",
    )
    parser.add_argument("--input", required=True, help="video file or directory of frames")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--init-box", help="frame-0 ERP pixel box: x,y,width,height")
    target.add_argument(
        "--init-bfov",
        help="frame-0 spherical box in degrees: clon,clat,fov_h,fov_v (360VOT convention)",
    )
    parser.add_argument("--output", default=None, help="result text file, one line per frame")
    parser.add_argument(
        "--preset",
        default="default",
        choices=sorted(PRESETS),
        help="default: one forward pass per frame; loss_handling: also looks for a lost "
        "target again (experimental)",
    )
    parser.add_argument(
        "--precision",
        default=None,
        choices=MODEL_PRECISIONS,
        help="tf32: faster on RTX 30 series and later, last digits differ",
    )
    parser.add_argument("--config", default=None, help="a configuration file instead of a preset")
    parser.add_argument("--weights", default=None, help="checkpoint file, if not models/")
    parser.add_argument("--demo", default=None, help="write a side-by-side MP4 of the run")
    parser.add_argument("--gif", default=None, help="write a GIF of the run")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sequence-id", default=None)
    parser.add_argument("--recursive", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = buildParser().parse_args(argv)
    tracker = None
    source = None
    try:
        if args.output is None and args.demo is None and args.gif is None:
            raise ConfigError("nothing to write: give --output, --demo or --gif")
        config = resolveConfig(
            args.preset, config=args.config, precision=args.precision, weights=args.weights
        )
        initialBox = _parseBox(args.init_box) if args.init_box is not None else None
        initialBfov = _parseBfov(args.init_bfov) if args.init_bfov is not None else None
        tracker = Track360Tracker(config)
        source = FrameSource(recursive=args.recursive, sequenceId=args.sequence_id)
        source.open(args.input)
        started = perf_counter()
        results = tracker.track(
            source,
            initBox=initialBox,
            initBfov=initialBfov,
            output=args.output,
            demo=args.demo,
            gif=args.gif,
            maxFrames=args.max_frames,
        )
        seconds = perf_counter() - started
        if args.output is not None:
            writeRunMetadata(args.output, config)
        print(
            f"{len(results)} frames in {seconds:.1f} s ({len(results) / max(seconds, 1e-9):.1f} "
            "frames per second, model loading included)",
            file=sys.stderr,
        )
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
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass
        if source is not None:
            try:
                source.close()
            except Exception:
                pass


def _parseBox(text: str) -> BBoxXYWH:
    try:
        return boxFromPixels([part.strip() for part in text.split(",")])
    except ConfigError as error:
        raise ConfigError("--init-box must contain four comma-separated numbers") from error


def _parseBfov(text: str) -> BFoV:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise ConfigError("--init-bfov must contain four comma-separated numbers")
    try:
        return bfovFromDegrees(parts)
    except ConfigError as error:
        raise ConfigError(f"--init-bfov: {error}") from error


def _report(error: Exception) -> None:
    print(f"{type(error).__name__}: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
