"""Does a search pass judge better when it carries the target's recent appearance?

    python tools/memory_search_study.py --dataset-root <dir> --label-root <dir> \\
        --sequence-file configs/splits/360vos_dev_train.txt --run outputs/E015_train \\
        --out outputs/E046

Only forward passes: nothing is tracked.  A search pass today is stateless: it sees the
frame-0 template and nothing of what the target looked like later.  Here the same pass
is also made with the tracker's appearance memory as it was some frames earlier, when
the target was last seen.

On every ``--step``-th frame with the target in the picture:

- a reference pass, centred on the true target and starting from the bare template,
  gives the appearance memory "as of this frame";
- candidate views of the normal size are laid ``near`` the true target (shifted by up
  to 0.7 of the view's half-width), ``far`` from it (1.3 to 4 half-widths, target
  outside), and, when the finished run was lost on this frame, around the box it had
  (``tracked``);
- every candidate view is passed once without memory and once with the memory of the
  reference pass made each of ``--lags`` frames earlier.

The memories come from passes centred on the ground truth, so this is what recent
appearance can give at best, not what a running system would have.  One row per pass
goes to ``<out>/passes.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from math import pi
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verifier_study import auroc  # noqa: E402
from view_scale_study import shifted  # noqa: E402

from track360.backends import createArtrackSession  # noqa: E402
from track360.backends.observation import buildRgbObservation  # noqa: E402
from track360.controller.view_planner import ViewPlanner  # noqa: E402
from track360.core.config import loadConfig  # noqa: E402
from track360.core.errors import ModelError  # noqa: E402
from track360.core.types import FrameIndex, FramePacket, SequenceId  # noqa: E402
from track360.datasets.tune_split import readSequenceFile  # noqa: E402
from track360.datasets.vot360 import Vot360Dataset  # noqa: E402
from track360.evaluation.loss_rate import dualIou, lostFrameMask  # noqa: E402
from track360.evaluation.vot360_metrics import loadTrackerResults  # noqa: E402
from track360.geometry.projection_math import makeSphericalPoint  # noqa: E402
from track360.runtime.benchmark import _SharedSession  # noqa: E402
from track360.runtime.driver import buildRuntime, closeRuntime  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
APPEARANCE_KEY = "appearance"
NEAR_SHIFT, FAR_SHIFT = (0.0, 0.7), (1.3, 4.0)
THRESHOLDS = (0.5, 0.6, 0.7, 0.8)


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    boxes = loadTrackerResults(args.run / "bbox" / args.method)
    spheres = loadTrackerResults(args.run / "bfov" / args.method)
    lags = [int(item) for item in args.lags.split(",")]
    session = createArtrackSession(config)
    planner = ViewPlanner(config.geometry, config.tracking, config.backendTuning)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
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
                view0 = runtime.geometry.cropViews(first, [plan.templateView])[0]
                runtime.backend.initialize(view0, plan.templateBox)
                backend = runtime.backend
                template = backend._artrackBackend.encodeTemplateView(*backend._detachedSource)

                def passWith(view: Any, memory: Any, template=template, backend=backend):
                    """One pass that starts from ``memory`` (``None``: the bare template)."""
                    template.memory.clear()
                    if memory is not None:
                        template.memory[APPEARANCE_KEY] = memory.clone()
                    prediction = backend._artrackBackend.inferBatch(
                        (view.rgb,),
                        (template,),
                        ((view.spec.bfov.horizontalFovRad, view.spec.bfov.verticalFovRad),),
                        priorBoxes=(view.spec.priorBox,),
                    )[0]
                    after = template.memory.get(APPEARANCE_KEY)
                    try:
                        return buildRgbObservation(view, prediction, 0), after
                    except ModelError:
                        return None, after

                def extent(row: np.ndarray) -> tuple[float, float]:
                    return (
                        min(max(np.radians(row[2]), 1e-4), 2 * pi - 1e-3),
                        min(max(np.radians(row[3]), 1e-4), pi - 1e-3),
                    )

                memories: dict[int, Any] = {}
                for index in range(args.start, sequence.frameCount, args.step):
                    if not present[index]:
                        continue
                    frame = FramePacket(
                        SequenceId(name), FrameIndex(index), index, sequence.readRgb(index)
                    )
                    height = frame.rgb.shape[0]
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
                    options: list[tuple[str, Any]] = [("none", None)]
                    for lag in lags:
                        # The newest reference at least ``lag`` frames back.
                        source = max((key for key in memories if key <= index - lag), default=None)
                        if source is not None:
                            options.append((str(lag), memories[source]))
                    for kind, spec in views:
                        view = runtime.geometry.cropViews(frame, [spec])[0]
                        for label, memory in options:
                            observation, _ = passWith(view, memory)
                            item: dict[str, object] = {
                                "sequence": name,
                                "frame": index,
                                "kind": kind,
                                "memory": label,
                                "lostRun": int(lost[index]),
                                "found": 0,
                                "score": "",
                                "iou": "",
                            }
                            if observation is not None:
                                try:
                                    box = runtime.geometry.projectLocalBoxBoundary(
                                        observation.bbox, spec, width, height
                                    ).bbox
                                except Exception:
                                    box = None
                                if box is not None:
                                    item.update(
                                        found=1,
                                        score=round(float(observation.fusedScore), 4),
                                        iou=round(
                                            float(
                                                dualIou(
                                                    truthBox[index : index + 1],
                                                    np.asarray(
                                                        [
                                                            [
                                                                box.xPx,
                                                                box.yPx,
                                                                box.widthPx,
                                                                box.heightPx,
                                                            ]
                                                        ]
                                                    ),
                                                    width,
                                                )[0]
                                            ),
                                            4,
                                        ),
                                    )
                            rows.append(item)
                    # The reference of this frame, for the frames that follow.
                    _, after = passWith(runtime.geometry.cropViews(frame, [base])[0], None)
                    if after is not None:
                        memories[index] = after.detach().clone()
                    for key in [key for key in memories if key < index - max(lags) - args.step]:
                        del memories[key]
                    releaseFrame = getattr(runtime.geometry, "releaseFrame", None)
                    if callable(releaseFrame):
                        releaseFrame()
            finally:
                closeRuntime(runtime)
                sequence.close()
            print(f"[{number}/{len(names)}] {name}: {len(rows)} passes so far", flush=True)
    finally:
        session.close()
    return rows


def report(rows: list[dict[str, object]]) -> dict[str, object]:
    """Per memory: how well the score separates boxes on the target from the others."""
    result: dict[str, object] = {}
    labels = sorted({str(row["memory"]) for row in rows}, key=lambda v: (v != "none", v.zfill(4)))
    # Compare on the same passes: frames and views that have every memory option.
    keyed: dict[tuple, dict[str, dict[str, object]]] = {}
    for row in rows:
        keyed.setdefault((row["sequence"], row["frame"], row["kind"]), {})[str(row["memory"])] = row
    complete = [group for group in keyed.values() if all(label in group for label in labels)]
    for title, subset in (
        ("all sampled frames", complete),
        (
            "frames on which the finished run was lost",
            [group for group in complete if int(group["none"]["lostRun"])],
        ),
    ):
        print(f"\n{title}: {len(subset)} views with every memory")
        print(
            f"  {'memory':<8}{'on target':>10}{'elsewhere':>10}{'AUROC':>8}"
            + "".join(f"{f'T>={t:g}':>9}{f'F>={t:g}':>9}" for t in THRESHOLDS)
            + f"{'near: on target':>17}"
        )
        entry = {}
        for label in labels:
            scores, positive = [], []
            nearHit = nearAll = 0
            for group in subset:
                row = group[label]
                if row["kind"] == "near":
                    nearAll += 1
                    nearHit += int(int(row["found"]) and float(row["iou"]) >= 0.5)
                if not int(row["found"]):
                    continue
                value = float(row["iou"])
                if 0.1 <= value < 0.5:
                    continue
                scores.append(float(row["score"]))
                positive.append(value >= 0.5)
            scoreArray, positiveArray = np.asarray(scores), np.asarray(positive, dtype=bool)
            cells = []
            for threshold in THRESHOLDS:
                cells.append(float((scoreArray[positiveArray] >= threshold).mean()))
                cells.append(float((scoreArray[~positiveArray] >= threshold).mean()))
            entry[label] = {
                "onTarget": int(positiveArray.sum()),
                "elsewhere": int((~positiveArray).sum()),
                "auroc": auroc(scoreArray, positiveArray),
                "rates": cells,
                "nearOnTarget": nearHit / max(1, nearAll),
            }
            print(
                f"  {label:<8}{int(positiveArray.sum()):>10}{int((~positiveArray).sum()):>10}"
                f"{entry[label]['auroc']:>8.3f}"
                + "".join(f"{value:>9.1%}" for value in cells)
                + f"{nearHit / max(1, nearAll):>17.1%}"
            )
        result[title] = entry
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--method", default="ours")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--start", type=int, default=10)
    parser.add_argument("--lags", default="10,50,150")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    path = args.out / "passes.csv"
    if args.report_only:
        with path.open(encoding="utf-8") as stream:
            rows: list[dict[str, object]] = list(csv.DictReader(stream))
    else:
        if args.run is None:
            print("--run is required unless --report-only", file=sys.stderr)
            return 2
        rows = run(args)
        args.out.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = report(rows)
    (args.out / "passes.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
