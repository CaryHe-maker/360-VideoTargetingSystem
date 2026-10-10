"""Fetch the model weights and check that they are the ones the project was built on.

The ARTrackV2 checkpoint is not ours to redistribute: it is downloaded from the place
its authors publish it (see ``models/README.md``).  Whatever the source, the file is
only accepted when its SHA-256 is the one recorded here, unless the caller says
otherwise.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from track360.core.errors import ModelError

_BLOCK = 1 << 20
_USER_AGENT = "track360-download"


@dataclass(frozen=True, slots=True)
class WeightsEntry:
    """One published checkpoint: where it comes from and what it must hash to."""

    name: str
    fileName: str
    sha256: str
    sizeBytes: int
    url: str
    page: str
    note: str


def _googleDrive(fileId: str) -> str:
    # The address Google Drive serves large files from without an interstitial page.
    return f"https://drive.usercontent.google.com/download?id={fileId}&export=download&confirm=t"


WEIGHTS: dict[str, WeightsEntry] = {
    "artrackv2_b_256": WeightsEntry(
        name="artrackv2_b_256",
        fileName="artrackv2_b_256.pth.tar",
        sha256="a99b7f8086e4827ecfe32ec8a9d32ad41c1ca9ff3cac551b62ec95576ca01d05",
        sizeBytes=1_614_788_967,
        url=_googleDrive("1tGaY5jQxZOTzJDWXgOgoHtBwc5l4NLQ2"),
        page="https://github.com/MIV-XJTU/ARTrack",
        note=(
            "ARTrackV2-B-256, the sequence-level checkpoint published by the ARTrack "
            "authors.  Their repository says the project is not for commercial use."
        ),
    ),
}


def sha256Of(path: str | Path, progress: Callable[[int], None] | None = None) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(_BLOCK), b""):
            digest.update(block)
            if progress is not None:
                progress(len(block))
    return digest.hexdigest()


def checkWeights(path: str | Path, name: str = "artrackv2_b_256") -> bool:
    """Whether ``path`` is the checkpoint ``name``, by its SHA-256."""
    target = Path(path)
    return target.is_file() and sha256Of(target) == WEIGHTS[name].sha256


def downloadWeights(
    target: str | Path,
    name: str = "artrackv2_b_256",
    *,
    url: str | None = None,
    force: bool = False,
    verify: bool = True,
    report: Callable[[str], None] | None = None,
) -> Path:
    """Download the checkpoint ``name`` to ``target`` and verify it.

    A file already at ``target`` with the right hash is kept.  The download goes to
    ``<target>.part`` and continues one left by an interrupted run; it only becomes
    ``target`` once its hash is right, so a broken download never looks like weights.
    ``url`` replaces the recorded source, for a mirror or a local copy
    (``file:///...``).  ``verify=False`` accepts a file with another hash.
    """
    if name not in WEIGHTS:
        raise ModelError(f"unknown weights '{name}'; known: {', '.join(sorted(WEIGHTS))}")
    entry = WEIGHTS[name]
    say = report or (lambda message: print(message, file=sys.stderr))
    destination = Path(target).expanduser()
    if destination.is_dir():
        destination = destination / entry.fileName
    if destination.is_file() and not force:
        if not verify or sha256Of(destination) == entry.sha256:
            say(f"{destination} is already there" + (" and verified" if verify else ""))
            return destination
        raise ModelError(
            f"{destination} exists but is not {name} (SHA-256 differs); "
            "remove it or pass force=True to download again"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if force and partial.exists():
        partial.unlink()
    source = url or entry.url
    say(f"downloading {name} from {source}")
    say(entry.note)
    _fetch(source, partial, entry.sizeBytes, say)
    actual = sha256Of(partial)
    if verify and actual != entry.sha256:
        raise ModelError(
            f"the downloaded file is not {name}: SHA-256 {actual}, expected {entry.sha256}.  "
            f"It is kept as {partial}.  If the source page ({entry.page}) changed its file, "
            "download it by hand and check models/README.md."
        )
    partial.replace(destination)
    say(f"saved {destination} ({destination.stat().st_size} bytes, SHA-256 {actual})")
    return destination


def _fetch(url: str, partial: Path, expectedBytes: int, say: Callable[[str], None]) -> None:
    have = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": _USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)  # noqa: S310 - caller's URL
    except urllib.error.HTTPError as error:
        if error.code == 416 and have:
            # Nothing left to fetch: the part file is already complete.
            return
        raise ModelError(f"cannot download {url}: HTTP {error.code} {error.reason}") from error
    except (urllib.error.URLError, OSError) as error:
        raise ModelError(f"cannot download {url}: {error}") from error
    with response:
        kind = response.headers.get("Content-Type", "")
        if kind.startswith("text/html"):
            raise ModelError(
                f"{url} answered with a web page instead of the file (a sign-in or a "
                "confirmation page).  Download the file in a browser and put it in place, "
                "or pass another address with url=."
            )
        resumed = have and getattr(response, "status", 200) == 206
        if have and not resumed:
            have = 0  # the server ignored the range: start over
        total = have + int(response.headers.get("Content-Length") or 0)
        shown = -1
        with partial.open("ab" if resumed else "wb") as stream:
            while True:
                block = response.read(_BLOCK)
                if not block:
                    break
                stream.write(block)
                have += len(block)
                percent = int(100 * have / max(total or expectedBytes, 1))
                if percent // 5 != shown:
                    shown = percent // 5
                    say(f"  {have / 1e6:.0f} MB ({min(percent, 100)}%)")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="track360 download",
        description="Download the model weights and verify their SHA-256.",
    )
    parser.add_argument("--name", default="artrackv2_b_256", choices=sorted(WEIGHTS))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="file or directory; default: models/ of this checkout",
    )
    parser.add_argument("--url", default=None, help="another source (mirror, file:///...)")
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    parser.add_argument("--no-verify", action="store_true", help="accept any SHA-256")
    parser.add_argument("--check", action="store_true", help="only verify the file that is there")
    args = parser.parse_args(argv)
    entry = WEIGHTS[args.name]
    target = args.output or Path(__file__).resolve().parents[2] / "models" / entry.fileName
    if target.is_dir():
        target = target / entry.fileName
    try:
        if args.check:
            if not target.is_file():
                print(f"{target} does not exist", file=sys.stderr)
                return 1
            ok = checkWeights(target, args.name)
            print(f"{target}: {'verified' if ok else 'SHA-256 differs from the recorded one'}")
            return 0 if ok else 1
        downloadWeights(
            target, args.name, url=args.url, force=args.force, verify=not args.no_verify
        )
        return 0
    except ModelError as error:
        print(f"ModelError: {error}", file=sys.stderr)
        return 4


__all__ = ["WEIGHTS", "WeightsEntry", "checkWeights", "downloadWeights", "main", "sha256Of"]
