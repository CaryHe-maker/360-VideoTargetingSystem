"""The Python entry point: load the tracker once, track as many videos as needed.

    from track360.api import Track360Tracker

    with Track360Tracker.fromPretrained() as tracker:
        results = tracker.track("panorama.mp4", initBfov=(12.0, -3.0, 20.0, 35.0))
        for result in results:
            print(int(result.frameIndex), result.bbox, result.status.name)

``preset`` picks one of the configurations shipped in ``configs/``:

    default         every frame one forward pass; the reference results
    loss_handling   as default, plus looking for the target again after losing it
                    (experimental: more forward passes, gain not established)

``precision="tf32"`` runs either of them faster on NVIDIA GPUs with TensorFloat-32
(RTX 30 series and later), with results that differ in the last digits.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import radians
from pathlib import Path

from track360.core.config import MODEL_PRECISIONS, AppConfig, loadConfig
from track360.core.errors import ConfigError
from track360.core.types import BBoxXYWH, BFoV, FramePacket, TrackResult
from track360.geometry.projection_math import makeSphericalPoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRESETS: dict[str, str] = {
    "default": "default.yaml",
    "loss_handling": "loss_handling.yaml",
}


def presetPath(preset: str) -> Path:
    """The configuration file of a preset."""
    if preset not in PRESETS:
        raise ConfigError(f"unknown preset '{preset}'; known: {', '.join(sorted(PRESETS))}")
    return REPOSITORY_ROOT / "configs" / PRESETS[preset]


def resolveConfig(
    preset: str = "default",
    *,
    config: str | Path | AppConfig | None = None,
    precision: str | None = None,
    weights: str | Path | None = None,
) -> AppConfig:
    """The configuration of a preset or a file, with the overrides applied."""
    loaded = config if isinstance(config, AppConfig) else loadConfig(config or presetPath(preset))
    model = loaded.model
    if precision is not None:
        if precision not in MODEL_PRECISIONS:
            raise ConfigError(
                f"unknown precision '{precision}'; known: {', '.join(MODEL_PRECISIONS)}"
            )
        model = replace(model, precision=precision)
    if weights is not None:
        model = replace(model, weights=Path(weights).expanduser().resolve())
    return replace(loaded, model=model)


def bfovFromDegrees(values: Sequence[float]) -> BFoV:
    """A BFoV from ``(clon, clat, fov_h, fov_v)`` in degrees, the 360VOT convention."""
    try:
        clon, clat, fovH, fovV = (float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ConfigError("a BFoV needs four numbers: clon, clat, fov_h, fov_v") from error
    if not (-180.0 <= clon <= 180.0 and -90.0 <= clat <= 90.0):
        raise ConfigError("BFoV center must satisfy -180<=clon<=180 and -90<=clat<=90")
    if not (0.0 < fovH < 180.0 and 0.0 < fovV < 180.0):
        raise ConfigError("BFoV field of view must be in (0, 180) degrees")
    return BFoV(
        center=makeSphericalPoint(radians(clon), radians(clat)),
        horizontalFovRad=radians(fovH),
        verticalFovRad=radians(fovV),
    )


def boxFromPixels(values: Sequence[float]) -> BBoxXYWH:
    """An ERP box from ``(x, y, width, height)`` in pixels of the first frame."""
    try:
        xPx, yPx, widthPx, heightPx = (float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ConfigError("a box needs four numbers: x, y, width, height") from error
    return BBoxXYWH(xPx=xPx, yPx=yPx, widthPx=widthPx, heightPx=heightPx)


class _Limited:
    """A frame source cut off after a number of frames."""

    def __init__(self, source, maxFrames: int) -> None:
        self._source = source
        self._left = maxFrames

    def read(self) -> FramePacket | None:
        if self._left <= 0:
            return None
        self._left -= 1
        return self._source.read()


class Track360Tracker:
    """ARTrackV2 on 360-degree video.  The weights are loaded on first use and kept."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._session = None

    @classmethod
    def fromPretrained(
        cls,
        preset: str = "default",
        *,
        precision: str | None = None,
        weights: str | Path | None = None,
        config: str | Path | AppConfig | None = None,
    ) -> Track360Tracker:
        """The tracker of a preset (or of a configuration file), not loaded yet."""
        return cls(resolveConfig(preset, config=config, precision=precision, weights=weights))

    @property
    def config(self) -> AppConfig:
        return self._config

    def __enter__(self) -> Track360Tracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def track(
        self,
        source,
        *,
        initBox: Sequence[float] | BBoxXYWH | None = None,
        initBfov: Sequence[float] | BFoV | None = None,
        output: str | Path | None = None,
        demo: str | Path | None = None,
        gif: str | Path | None = None,
        maxFrames: int | None = None,
    ) -> list[TrackResult]:
        """Track the target given in the first frame through ``source``.

        ``source`` is a video file, a directory of frames, or an object with a
        ``read()`` returning frames in order.  The target is given as exactly one of
        ``initBox`` (ERP pixels: x, y, width, height) and ``initBfov`` (degrees: clon,
        clat, fov_h, fov_v).  ``output`` also writes the results as a text file,
        ``demo`` / ``gif`` a side-by-side video of the run.  One result per frame is
        returned, the first being the given target.
        """
        from track360.backends import createArtrackSession
        from track360.datasets.frame_source import FrameSource
        from track360.io.vot360_results import ResultCollector
        from track360.runtime.benchmark import _SharedSession
        from track360.runtime.driver import (
            buildRuntime,
            closeRuntime,
            finalizeSink,
            openSink,
            runTracking,
        )

        if (initBox is None) == (initBfov is None):
            raise ConfigError("give exactly one of initBox and initBfov")
        box = (
            None
            if initBox is None
            else initBox
            if isinstance(initBox, BBoxXYWH)
            else boxFromPixels(initBox)
        )
        bfov = (
            None
            if initBfov is None
            else initBfov
            if isinstance(initBfov, BFoV)
            else bfovFromDegrees(initBfov)
        )
        if maxFrames is not None and maxFrames <= 0:
            raise ConfigError("maxFrames must be positive")
        if self._session is None:
            self._session = createArtrackSession(self._config)
        session = self._session
        opened = None
        if isinstance(source, (str, Path)):
            opened = FrameSource()
            opened.open(str(source))
            frames = opened
        else:
            frames = source
        if maxFrames is not None:
            frames = _Limited(frames, maxFrames)
        recorder = None
        if demo is not None or gif is not None:
            from track360.visualization.demo import DemoRecorder

            recorder = DemoRecorder(demo, gifPath=gif)
        runtime = buildRuntime(
            self._config, artrackSessionFactory=lambda _: _SharedSession(session)
        )
        collector = ResultCollector()
        try:
            runTracking(
                source=frames,
                initialBox=box,
                initialBfov=bfov,
                geometry=runtime.geometry,
                controller=runtime.controller,
                backend=runtime.backend,
                sink=collector,
                recorder=recorder,
                resultRecorder=recorder,
                verifier=runtime.verifier,
            )
            if output is not None:
                openSink(runtime.sink, str(output))
                for result in collector.results:
                    runtime.sink.write(result)
                finalizeSink(runtime.sink, len(collector.results))
        finally:
            if recorder is not None:
                recorder.close()
            closeRuntime(runtime)
            if opened is not None:
                opened.close()
        return list(collector.results)


__all__ = [
    "PRESETS",
    "Track360Tracker",
    "bfovFromDegrees",
    "boxFromPixels",
    "presetPath",
    "resolveConfig",
]
