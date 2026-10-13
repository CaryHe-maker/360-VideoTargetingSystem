"""Record what the backend knows beyond its box and score, for learned judges.

    python tools/backend_signals.py freerun --dataset-root <dir> --label-root <dir> \\
        --sequence-file configs/splits/360vos_dev_train.txt --out outputs/E045
    python tools/backend_signals.py candidates --dataset-root <dir> --label-root <dir> \\
        --sequence-file configs/splits/360vos_dev_train.txt --run outputs/E015_train \\
        --out outputs/E045

For every pass the session's diagnostics are kept: the entropy and peak probability of
the four coordinate distributions, the gap between their arg-max and expectation, and
the 768-value token the score head reads.

``freerun`` tracks every sequence with the default configuration (loss handling off)
and writes one row per frame to ``<out>/freerun/<sequence>.npz``; the result files go to
``<out>/run`` and must equal those of baseline 1.  It answers: can the box on a frame be
told wrong from inside the network?

``candidates`` makes stateless passes only, on every ``--step``-th frame with the target
in the picture, with views of the normal size laid

    near      around the true target, shifted by up to 0.7 of the view's half-width
    far       1.3 to 4 half-widths away from it, so the target is outside the view
    tracked   around the box a finished run had on that frame, when that run was lost

and writes them to ``<out>/candidates.npz`` with the IoU of each returned box.  It
answers: can a search candidate be told from the target better than by its score?
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Sequence
from math import pi, sqrt
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from view_scale_study import shifted  # noqa: E402

from track360.backends import createArtrackSession  # noqa: E402
from track360.controller.view_planner import ViewPlanner  # noqa: E402
from track360.core.config import loadConfig  # noqa: E402
from track360.core.types import BBoxXYWH, FrameIndex, FramePacket, SequenceId  # noqa: E402
from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.loss_rate import dualIou, lostFrameMask  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402
from track360.geometry.projection_math import makeSphericalPoint  # noqa: E402
from track360.runtime.benchmark import _SharedSession, runBenchmark  # noqa: E402
from track360.runtime.driver import buildRuntime, closeRuntime  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
SCALARS = ("coordEntropy", "coordPeak", "hardSoftGap")
NEAR_SHIFT, FAR_SHIFT = (0.0, 0.7), (1.3, 4.0)


class _Recorder:
    """A session that keeps the diagnostics of every pass, split by sequence."""

    def __init__(self, session: Any) -> None:
        self._session = session
        session.enableDiagnostics(True)
        self.sequences: list[list[dict[str, Any]]] = []

    def encodeTemplate(self, rgb: Any, bbox: BBoxXYWH) -> Any:
        self.sequences.append([])
        return self._session.encodeTemplate(rgb, bbox)

    def infer(self, rgb: Any, templateFeatures: Sequence[object]) -> Any:
        return self.inferBatch((rgb,), templateFeatures)[0]

    def inferBatch(self, rgbs: Sequence[Any], *args: Any, **kwargs: Any) -> Any:
        predictions = self._session.inferBatch(rgbs, *args, **kwargs)
        self.sequences[-1].append(
            {**self._session.lastDiagnostics, "score": float(predictions[-1].modelScore)}
        )
        return predictions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def close(self) -> None:
        self._session.close()


def _stack(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        **{key: np.asarray([row[key] for row in rows], dtype=np.float32) for key in SCALARS},
        "score": np.asarray([row["score"] for row in rows], dtype=np.float32),
        "scoreFeature": np.asarray([row["scoreFeature"] for row in rows], dtype=np.float16),
    }


def freerun(args: argparse.Namespace) -> int:
    config = loadConfig(args.config)
    recorders: list[_Recorder] = []

    def factory(_: Any) -> _Recorder:
        recorders.append(_Recorder(createArtrackSession(config)))
        return recorders[-1]

    summary = runBenchmark(
        datasetRoot=args.dataset_root,
        outputRoot=args.out / "run",
        labelRoot=args.label_root,
        method="ours",
        config=config,
        sequences=readSequenceFile(args.sequence_file),
        resume=False,
        sessionFactory=factory,
    )
    target = args.out / "freerun"
    target.mkdir(parents=True, exist_ok=True)
    done = [report for report in summary.reports if report.status == "done"]
    if len(done) != len(recorders[0].sequences):
        print("a sequence failed: the recorded passes cannot be matched to sequences")
        return 1
    for report, rows in zip(done, recorders[0].sequences, strict=True):
        if len(rows) != report.frames - 1:
            print(f"{report.sequence}: {len(rows)} passes for {report.frames} frames")
            return 1
        # Pass i belongs to frame i + 1: frame 0 only initialises the tracker.
        np.savez_compressed(target / f"{report.sequence}.npz", **_stack(rows))
    print(f"{len(done)} sequences written to {target}")
    return 0


def candidates(args: argparse.Namespace) -> int:
    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    boxes = loadTrackerResults(args.run / "bbox" / args.method)
    spheres = loadTrackerResults(args.run / "bfov" / args.method)
    session = createArtrackSession(config)
    session.enableDiagnostics(True)
    planner = ViewPlanner(config.geometry, config.tracking, config.backendTuning)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    meta: list[tuple[str, int, str, float, float]] = []
    names = readSequenceFile(args.sequence_file)
    try:
        for number, name in enumerate(names, 1):
            sequence = dataset.sequence(name)
            truthBox, present = sequence.groundTruth("bbox")
            truthBfov, _ = sequence.groundTruth("bfov")
            present = np.asarray(present, dtype=bool)
            width = sequence.frameSize[0]
            iou = np.zeros(len(present))
            iou[present] = dualIou(truthBox[present], boxes[name][present], width)
            lost = lostFrameMask(iou, present)
            runtime = buildRuntime(config, artrackSessionFactory=lambda _: _SharedSession(session))
            try:
                first = FramePacket(SequenceId(name), FrameIndex(0), 0, sequence.readRgb(0))
                plan = runtime.controller.buildInitialization(
                    first, initialBfov=sequence.initialBfov()
                )
                template = runtime.geometry.cropViews(first, [plan.templateView])[0]
                runtime.backend.initialize(template, plan.templateBox)
                for index in range(args.start, sequence.frameCount, args.step):
                    if not present[index]:
                        continue
                    frame = FramePacket(
                        SequenceId(name), FrameIndex(index), index, sequence.readRgb(index)
                    )
                    height = frame.rgb.shape[0]

                    def extent(row: np.ndarray) -> tuple[float, float]:
                        return (
                            min(max(np.radians(row[2]), 1e-4), 2 * pi - 1e-3),
                            min(max(np.radians(row[3]), 1e-4), pi - 1e-3),
                        )

                    truth = makeSphericalPoint(
                        np.radians(truthBfov[index, 0]), np.radians(truthBfov[index, 1])
                    )
                    wide, tall = extent(truthBfov[index])
                    base = planner._searchView(truth, wide, tall)
                    half = 0.5 * min(base.bfov.horizontalFovRad, base.bfov.verticalFovRad)
                    views = []
                    for kind, (low, high) in (("near", NEAR_SHIFT), ("far", FAR_SHIFT)):
                        centre = shifted(
                            truth, rng.uniform(low, high) * half, rng.uniform(0.0, 2 * pi)
                        )
                        views.append((kind, planner._searchView(centre, wide, tall)))
                    if lost[index]:
                        row = spheres[name][index]
                        views.append(
                            (
                                "tracked",
                                planner._searchView(
                                    makeSphericalPoint(np.radians(row[0]), np.radians(row[1])),
                                    *extent(row),
                                ),
                            )
                        )
                    for kind, spec in views:
                        found = runtime.backend.inferDetached(
                            (runtime.geometry.cropViews(frame, [spec])[0],)
                        )
                        if not found:
                            continue
                        try:
                            projected = runtime.geometry.projectLocalBoxBoundary(
                                found[0].bbox, spec, width, height
                            )
                        except Exception:
                            continue
                        box = projected.bbox
                        overlap = float(
                            dualIou(
                                truthBox[index : index + 1],
                                np.asarray([[box.xPx, box.yPx, box.widthPx, box.heightPx]]),
                                width,
                            )[0]
                        )
                        rows.append(
                            {**session.lastDiagnostics, "score": float(found[0].fusedScore)}
                        )
                        size = sqrt(max(truthBfov[index, 2] * truthBfov[index, 3], 1e-6))
                        meta.append((name, index, kind, overlap, float(size)))
                    releaseFrame = getattr(runtime.geometry, "releaseFrame", None)
                    if callable(releaseFrame):
                        releaseFrame()
            finally:
                closeRuntime(runtime)
                sequence.close()
            print(f"[{number}/{len(names)}] {name}: {len(rows)} passes so far", flush=True)
    finally:
        session.close()
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "candidates.npz",
        **_stack(rows),
        sequence=np.asarray([item[0] for item in meta]),
        frame=np.asarray([item[1] for item in meta]),
        kind=np.asarray([item[2] for item in meta]),
        iou=np.asarray([item[3] for item in meta], dtype=np.float32),
        targetSizeDeg=np.asarray([item[4] for item in meta], dtype=np.float32),
    )
    print(f"{len(rows)} passes written to {args.out / 'candidates.npz'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("freerun", "candidates"))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=None, help="candidates: a finished run")
    parser.add_argument("--method", default="ours")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--start", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    if args.command == "freerun":
        return freerun(args)
    if args.run is None:
        print("candidates needs --run", file=sys.stderr)
        return 2
    return candidates(args)


if __name__ == "__main__":
    raise SystemExit(main())
