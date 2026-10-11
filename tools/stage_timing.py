"""Where does the time of a frame go?  The speed baseline for later optimisation.

    python tools/stage_timing.py --dataset-root <dir> --label-root <dir> \\
        --sequence-file configs/splits/360vos_dev_train.txt --out outputs/E038 \\
        --repeat-sequences 006,011,068,078,159 --repeats 3

Run it with nothing else on the machine.  Four measurements, all with the given config
(the default one: loss handling off):

    pipeline   the tracking loop as ``tools/benchmark.py run`` runs it, with the stage
               profiler on: per frame, the wall time between two results and the time
               in each stage of the main thread, plus what the decode thread spent
    repeats    the same on a few sequences several times, and once without the profiler,
               for the run-to-run spread and the cost of profiling
    in-memory  one sequence with its frames decoded beforehand: the loop without any
               decoding, the speed the rest would reach if frames were free
    micro      single steps in isolation with the GPU synchronised around each: reading
               a frame (archive read, JPEG decode, colour order), and the steps of a
               forward pass (search crop, to device, network, decoding the output)

The main-thread stages of ``pipeline`` are wall times while the decode thread is
running, so they include whatever that thread takes from them; ``micro`` gives the same
steps undisturbed.  Result files go to ``<out>/results`` so they can be compared with a
reference run: timing must not change a single box.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np

from track360.backends import createArtrackSession
from track360.core.config import loadConfig
from track360.core.types import FrameIndex, FramePacket, SequenceId
from track360.datasets.vot360 import Vot360Dataset, Vot360DataSource
from track360.evaluation.profiler import RuntimeProfiler
from track360.runtime.benchmark import _SharedSession, _TimedCollector, writeSequenceResults
from track360.runtime.driver import buildRuntime, closeRuntime, runTracking

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
STAGES = ("decode", "controller", "crop", "backend", "calibration", "projection", "total")
COLUMNS = (
    "pass",
    "sequence",
    "frame",
    "intervalMs",
    "waitMs",
    "controllerMs",
    "cropMs",
    "backendMs",
    "calibrationMs",
    "projectionMs",
    "totalMs",
    "workerDecodeMs",
    "queueWaitMs",
    "readyLeadMs",
)


class _MemorySource:
    """Frames decoded beforehand, handed out without any work."""

    def __init__(self, frames: list[FramePacket]) -> None:
        self._frames = frames
        self._next = 0

    def read(self) -> FramePacket | None:
        if self._next >= len(self._frames):
            return None
        frame = self._frames[self._next]
        self._next += 1
        return frame


def _track(config, session, source, initialBfov, profiler):
    runtime = buildRuntime(config, artrackSessionFactory=lambda _: _SharedSession(session))
    collector = _TimedCollector()
    try:
        runTracking(
            source=source,
            initialBfov=initialBfov,
            geometry=runtime.geometry,
            controller=runtime.controller,
            backend=runtime.backend,
            sink=collector,
            verifier=runtime.verifier,
            profiler=profiler,
        )
    finally:
        closeRuntime(runtime)
    return collector


def _rows(label: str, name: str, collector, profiler) -> list[dict[str, object]]:
    times = np.asarray(collector.writeTimes)
    intervals = np.diff(times) * 1000.0
    rows = []
    frames = profiler.frameRows[1:] if profiler is not None else [None] * len(intervals)
    for index, interval in enumerate(intervals):
        row: dict[str, object] = {
            "pass": label,
            "sequence": name,
            "frame": index + 1,
            "intervalMs": round(float(interval), 4),
        }
        frame = frames[index] if index < len(frames) else None
        if frame is not None:
            stages = frame.get("stages", {})
            for stage, column in (
                ("decode", "waitMs"),
                ("controller", "controllerMs"),
                ("crop", "cropMs"),
                ("backend", "backendMs"),
                ("calibration", "calibrationMs"),
                ("projection", "projectionMs"),
                ("total", "totalMs"),
            ):
                row[column] = round(float(stages.get(stage, 0.0)), 4)
            for key, column in (
                ("pipelineDecodeNs", "workerDecodeMs"),
                ("pipelineQueueWaitNs", "queueWaitMs"),
                ("pipelineReadyLeadNs", "readyLeadMs"),
            ):
                # The driver records the reader's figures twice per frame (before and
                # after the frame's work); the profiler adds them up.
                if key in stages:
                    row[column] = round(float(stages[key]) / 2.0, 4)
        rows.append(row)
    return rows


def pipeline(args, config, session, dataset, names, label, profile=True, save=False):
    rows = []
    for number, name in enumerate(names, 1):
        source = Vot360DataSource(labelRoot=dataset.labelRoot)
        source.open(str(dataset.root), name)
        try:
            profiler = RuntimeProfiler(enabled=True) if profile else None
            collector = _track(config, session, source, source.sequence.initialBfov(), profiler)
            part = _rows(label, name, collector, profiler)
            rows.extend(part)
            if save:
                collector.finalize(source.frameCount)
                writeSequenceResults(
                    args.out / "results",
                    "ours",
                    name,
                    collector.results,
                    source.sequence.frameSize[0],
                )
            intervals = [float(r["intervalMs"]) for r in part]
            print(
                f"[{label} {number}/{len(names)}] {name}: {len(part)} frames, "
                f"median {np.median(intervals):.1f} ms, {1000.0 / np.mean(intervals):.1f} FPS",
                flush=True,
            )
        finally:
            source.close()
    return rows


def inMemory(args, config, session, dataset, name, count):
    source = Vot360DataSource(maxFrames=count, labelRoot=dataset.labelRoot)
    source.open(str(dataset.root), name)
    try:
        frames = []
        while (frame := source.read()) is not None:
            frames.append(frame)
        initial = source.sequence.initialBfov()
    finally:
        source.close()
    rows = []
    for repeat in range(args.repeats):
        profiler = RuntimeProfiler(enabled=True)
        collector = _track(config, session, _MemorySource(frames), initial, profiler)
        rows.extend(_rows(f"memory{repeat + 1}", name, collector, profiler))
    return rows


def microDecode(dataset, names, count):
    import cv2

    result = {}
    for name in names:
        sequence = dataset.sequence(name)
        read, decode, colour, size = [], [], [], []
        try:
            for index in range(min(count, sequence.frameCount)):
                member = f"image/{sequence._frameName(index)}"
                started = perf_counter()
                payload = sequence.readMember(member)
                afterRead = perf_counter()
                bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                afterDecode = perf_counter()
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                afterColour = perf_counter()
                read.append(afterRead - started)
                decode.append(afterDecode - afterRead)
                colour.append(afterColour - afterDecode)
                size.append(len(payload))
            result[name] = {
                "frames": len(read),
                "widthPx": int(bgr.shape[1]),
                "heightPx": int(bgr.shape[0]),
                "bytes": float(np.median(size)),
                "readMs": float(np.median(read) * 1000),
                "jpegMs": float(np.median(decode) * 1000),
                "colourMs": float(np.median(colour) * 1000),
            }
        finally:
            sequence.close()
    return result


def microForward(config, session, dataset, name, count):
    """The steps of one tracking forward pass on real search views, GPU synchronised."""
    from track360.backends.artrack_model import _SEARCH_FACTOR, _SEARCH_SIZE, _sampleTarget
    from track360.backends.artrack_seq_session import padTrajectory, trajectoryTokens
    from track360.controller.view_planner import ViewPlanner, localBoxOfBfov

    torch = session._torch
    planner = ViewPlanner(config.geometry, config.tracking, config.backendTuning)
    runtime = buildRuntime(config, artrackSessionFactory=lambda _: _SharedSession(session))
    sequence = dataset.sequence(name)
    steps: dict[str, list[float]] = {
        key: []
        for key in (
            "viewCrop",
            "searchCrop",
            "toDevice",
            "tokens",
            "network",
            "output",
            "projection",
        )
    }
    try:
        first = FramePacket(SequenceId(name), FrameIndex(0), 0, sequence.readRgb(0))
        plan = runtime.controller.buildInitialization(first, initialBfov=sequence.initialBfov())
        template = runtime.geometry.cropViews(first, [plan.templateView])[0]
        runtime.backend.initialize(template, plan.templateBox)
        anchor = runtime.backend._template
        target = sequence.initialBfov()
        frame = first
        height, width = frame.rgb.shape[:2]
        for index in range(count + 20):
            spec = planner.searchView(
                target.center, target.horizontalFovRad, target.verticalFovRad, ()
            )
            torch.cuda.synchronize()
            t0 = perf_counter()
            view = runtime.geometry.cropViews(frame, [spec])[0]
            t1 = perf_counter()
            prior = spec.priorBox
            crop, resizeFactor, _ = _sampleTarget(view.rgb, prior, _SEARCH_FACTOR, _SEARCH_SIZE)
            t2 = perf_counter()
            search = session._preprocess(crop).unsqueeze(0)
            torch.cuda.synchronize()
            t3 = perf_counter()
            boxes = padTrajectory((prior,), session._trajectoryLength)
            tokens = torch.tensor(
                [trajectoryTokens(boxes, prior, resizeFactor, session._bins)],
                dtype=torch.float32,
                device=session._device,
            )
            torch.cuda.synchronize()
            t4 = perf_counter()
            with torch.inference_mode():
                appearance = session._model.backbone.patch_embed(anchor.tensor.unsqueeze(0))
                output = session._model(
                    template=anchor.tensor[None, None],
                    dz_feat=appearance.clone(),
                    search=search,
                    seq_input=tokens,
                )
            torch.cuda.synchronize()
            t5 = perf_counter()
            output["seqs"][:, 0:4].float().tolist()
            float(output["score"].reshape(-1)[0].item())
            t6 = perf_counter()
            runtime.geometry.projectLocalBoxBoundary(
                localBoxOfBfov(spec, target), spec, width, height
            )
            t7 = perf_counter()
            if index >= 20:
                for key, value in zip(
                    steps,
                    (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6),
                    strict=True,
                ):
                    steps[key].append(value * 1000.0)
    finally:
        closeRuntime(runtime)
        sequence.close()
    return {
        key: {
            "medianMs": float(np.median(values)),
            "p25Ms": float(np.percentile(values, 25)),
            "p75Ms": float(np.percentile(values, 75)),
        }
        for key, values in steps.items()
    }


def environment() -> dict[str, object]:
    import torch

    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnnDeterministic": bool(torch.backends.cudnn.deterministic),
        "cudnnBenchmark": bool(torch.backends.cudnn.benchmark),
        "threads": torch.get_num_threads(),
    }
    for key, command in (
        ("powerScheme", ["powercfg", "/getactivescheme"]),
        (
            "nvidia",
            [
                "nvidia-smi",
                "--query-gpu=driver_version,power.draw,temperature.gpu,utilization.gpu,memory.used",
                "--format=csv,noheader",
            ],
        ),
    ):
        try:
            info[key] = subprocess.run(
                command, capture_output=True, text=True, timeout=20, check=False
            ).stdout.strip()
        except Exception as error:  # the baseline must not fail on a missing tool
            info[key] = f"unavailable: {error}"
    return info


def _summary(rows: list[dict[str, object]], label: str) -> dict[str, object]:
    part = [row for row in rows if row["pass"] == label]
    if not part:
        return {}
    item: dict[str, object] = {"frames": len(part)}
    for column in COLUMNS[3:]:
        values = np.asarray([float(row[column]) for row in part if column in row])
        if values.size:
            item[column] = {
                "mean": float(values.mean()),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            }
    intervals = np.asarray([float(row["intervalMs"]) for row in part])
    item["fps"] = float(1000.0 / intervals.mean())
    return item


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-root", default=None)
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeat-sequences", default="006,011,068,078,159")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory-sequence", default="068")
    parser.add_argument("--memory-frames", type=int, default=200)
    parser.add_argument("--warmup", default="159")
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    config = loadConfig(args.config)
    dataset = Vot360Dataset(args.dataset_root, args.label_root)
    names = [
        line.strip()
        for line in args.sequence_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    subset = [name for name in args.repeat_sequences.split(",") if name]
    args.out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"environmentBefore": environment(), "config": str(args.config)}
    session = createArtrackSession(config)
    rows: list[dict[str, object]] = []
    try:
        pipeline(args, config, session, dataset, [args.warmup], "warmup")
        rows += pipeline(args, config, session, dataset, names, "pipeline", save=True)
        for repeat in range(args.repeats):
            rows += pipeline(args, config, session, dataset, subset, f"repeat{repeat + 1}")
        rows += pipeline(args, config, session, dataset, subset, "noprofile", profile=False)
        rows += inMemory(args, config, session, dataset, args.memory_sequence, args.memory_frames)
        payload["microDecode"] = microDecode(dataset, names, 40)
        payload["microForward"] = {
            name: microForward(config, session, dataset, name, 200) for name in subset[:3]
        }
    finally:
        session.close()
    payload["environmentAfter"] = environment()
    with (args.out / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    labels = list(dict.fromkeys(str(row["pass"]) for row in rows))
    payload["passes"] = {label: _summary(rows, label) for label in labels}
    perSequence = {}
    for name in names:
        part = [r for r in rows if r["pass"] == "pipeline" and r["sequence"] == name]
        intervals = np.asarray([float(r["intervalMs"]) for r in part])
        perSequence[name] = {
            "frames": len(part),
            "fps": float(1000.0 / intervals.mean()),
            "p50": float(np.percentile(intervals, 50)),
            "p95": float(np.percentile(intervals, 95)),
            "backendP50": float(np.median([float(r["backendMs"]) for r in part])),
            "cropP50": float(np.median([float(r["cropMs"]) for r in part])),
            "waitP50": float(np.median([float(r["waitMs"]) for r in part])),
            "workerDecodeP50": float(np.median([float(r["workerDecodeMs"]) for r in part])),
        }
    payload["perSequence"] = perSequence
    (args.out / "stage_timing.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for label in labels:
        item = payload["passes"][label]
        cells = "  ".join(
            f"{column[:-2]} {item[column]['p50']:.2f}/{item[column]['mean']:.2f}"
            for column in COLUMNS[3:]
            if column in item
        )
        print(
            f"{label:<10} {item['frames']:>6} frames  {item['fps']:.1f} FPS  p50/mean ms: {cells}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
