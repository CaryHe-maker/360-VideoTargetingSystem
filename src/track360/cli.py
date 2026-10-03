"""Unified command-line interface: ``track360 <command> [options]``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from track360.core.errors import Track360Error

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "default.yaml"

COMMANDS: dict[str, str] = {
    "track": "Track one target in a video or image sequence from an initial ERP box.",
    "airsim360": "Track one instance of an AirSim360 sequence with optional diagnostics.",
    "list-instances": "List the instance IDs visible in the first frame of an AirSim360 sequence.",
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _printUsage()
        return 0 if args else 2
    command, rest = args[0], args[1:]
    handlers: dict[str, Callable[[list[str]], int]] = {
        "track": _track,
        "airsim360": _airsim360,
        "list-instances": listInstancesMain,
    }
    handler = handlers.get(command)
    if handler is None:
        print(f"unknown command: {command}", file=sys.stderr)
        _printUsage(sys.stderr)
        return 2
    return handler(rest)


def listInstancesMain(argv: list[str]) -> int:
    from track360.datasets.airsim360_source import AirSim360DataSource
    from track360.datasets.instance_ids import (
        collectInstanceIdGroups,
        formatInstanceIdDocument,
        writeInstanceIdDocument,
    )

    parser = argparse.ArgumentParser(
        prog="track360 list-instances",
        description=COMMANDS["list-instances"],
    )
    parser.add_argument("dataset_root", help="Sequence or dataset parent directory.")
    parser.add_argument("--sequence", default=None, help="Sequence subdirectory, if any.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output text file.")
    args = parser.parse_args(argv)

    source = AirSim360DataSource(maxFrames=1)
    try:
        source.open(str(Path(args.dataset_root).expanduser()), args.sequence)
        frame = source.read()
        if frame is None:
            raise ValueError("the sequence contains no frames")
        groups = collectInstanceIdGroups(frame)
        if args.output is None:
            print(formatInstanceIdDocument(groups), end="")
        else:
            print(writeInstanceIdDocument(args.output, groups))
        return 0
    except (OSError, ValueError, Track360Error) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        source.close()


def _track(argv: list[str]) -> int:
    from track360.runtime.track_video import main as trackMain

    return trackMain(_withDefaultConfig(argv))


def _airsim360(argv: list[str]) -> int:
    from track360.runtime.track_airsim360 import main as trackAirSim360Main

    return trackAirSim360Main(_withDefaultConfig(argv))


def _withDefaultConfig(argv: list[str]) -> list[str]:
    if any(arg in {"-h", "--help"} for arg in argv):
        return argv
    if any(arg == "--config" or arg.startswith("--config=") for arg in argv):
        return argv
    return [*argv, "--config", str(DEFAULT_CONFIG)]


def _printUsage(stream=None) -> None:
    stream = stream or sys.stdout
    print("usage: track360 <command> [options]\n\ncommands:", file=stream)
    for name, summary in COMMANDS.items():
        print(f"  {name:<16}{summary}", file=stream)
    print("\nRun 'track360 <command> --help' for command options.", file=stream)


__all__ = ["DEFAULT_CONFIG", "listInstancesMain", "main"]
