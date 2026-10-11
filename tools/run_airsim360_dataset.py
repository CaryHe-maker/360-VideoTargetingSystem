"""Run one explicitly selected AirSim360 instance and publish an evaluation bundle."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from track360.core.types import BBoxXYWH
from track360.datasets.airsim360_source import AirSim360DataSource
from track360.datasets.pseudo_track_builder import PseudoTrackBuilder
from track360.evaluation.otb_metrics import OtbMetrics, circularBBoxIoU, readResultFile
from track360.geometry.seam import minimalCircularInterval
from track360.runtime.track_airsim360 import main as trackMain


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory; otherwise artifacts/<data-relative>/output_N.",
    )
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--target-instance", type=int, default=None)
    parser.add_argument(
        "--list-instances",
        action="store_true",
        help="List first-frame instance IDs and exit; does not run tracking.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = buildParser().parse_args(argv)
    source = AirSim360DataSource(maxFrames=args.max_frames)
    source.open(args.dataset_root, args.sequence)
    first = source.read()
    if first is None:
        raise RuntimeError("dataset is empty")
    if args.list_instances:
        print(json.dumps(_instanceCatalog(first), indent=2))
        source.close()
        return 0
    if args.target_instance is None:
        raise ValueError("--target-instance is required; use --list-instances to inspect IDs")
    runRoot = _resolveOutputDir(args)
    resultRoot = runRoot / "result"
    midVisualRoot = runRoot / "midVisual"
    resultVisualRoot = resultRoot / "visualResult"
    resultRoot.mkdir(parents=True, exist_ok=True)
    targetBox = PseudoTrackBuilder().buildInitialBox(first, args.target_instance)
    manifest = {
        "format": "airsim360",
        "datasetRoot": str(Path(args.dataset_root).resolve()),
        "sequence": str(first.sequenceId),
        "frameCount": source.frameCount,
        "shape": list(first.rgb.shape[:2]),
        "targetInstance": args.target_instance,
        "outputDirectory": str(runRoot.resolve()),
        "initialBox": _boxToDict(targetBox),
        "classNames": first.segmentation.classNames if first.segmentation else {},
    }
    source.close()

    resultPath = resultRoot / "tracking.txt"
    code = trackMain(
        [
            "--dataset-root", str(args.dataset_root),
            "--config", str(args.config),
            "--output", str(resultPath),
            "--target-instance", str(args.target_instance),
            "--mid-visual-root", str(midVisualRoot),
            "--result-visual-root", str(resultVisualRoot),
            *( ["--sequence", args.sequence] if args.sequence else [] ),
            *( ["--max-frames", str(args.max_frames)] if args.max_frames else [] ),
        ]
    )

    if code != 0 or not resultPath.is_file():
        manifest.update(
            {
                "trackingExitCode": code,
                "trackedFrameCount": 0,
                "error": (
                    "tracking failed before a complete tracking.txt was produced"
                    if code != 0
                    else "tracking returned success without producing tracking.txt"
                ),
            }
        )
        (resultRoot / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2))
        return code if code != 0 else 1

    iou = _evaluate(args, resultPath)
    (resultRoot / "iou.json").write_text(json.dumps(iou, indent=2), encoding="utf-8")
    manifest.update(
        {
            "trackingExitCode": code,
            "trackedFrameCount": len(readResultFile(resultPath)) if resultPath.exists() else 0,
            "iouFile": "iou.json",
            "resultVisualDirectory": "visualResult",
        }
    )
    (resultRoot / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "result": str(resultRoot.resolve()),
                "midVisual": str(midVisualRoot.resolve()),
                **manifest,
                "iou": iou["summary"],
            },
            indent=2,
        )
    )
    return code


def _evaluate(args: argparse.Namespace, resultPath: Path) -> dict[str, object]:
    predictions = readResultFile(resultPath)
    source = AirSim360DataSource(maxFrames=args.max_frames)
    source.open(args.dataset_root, args.sequence)
    builder = PseudoTrackBuilder()
    metrics = OtbMetrics()
    frames: list[dict[str, object]] = []
    try:
        for index, prediction in enumerate(predictions):
            frame = source.read()
            if frame is None:
                break
            groundTruth, visible = builder.buildPseudoGroundTruth(frame, args.target_instance)
            value = circularBBoxIoU(prediction, groundTruth, frame.rgb.shape[1]) if visible else 0.0
            # Frame 0 is the supplied initialization box, not a tracker prediction.
            if visible and int(frame.frameIndex) != 0:
                metrics.ious.append(value)
            frames.append(
                {
                    "frameIndex": int(frame.frameIndex),
                    "visible": visible,
                    "prediction": _boxToDict(prediction),
                    "groundTruth": _boxToDict(groundTruth) if visible else None,
                    "iou": value,
                    "includedInSummary": visible and int(frame.frameIndex) != 0,
                }
            )
    finally:
        source.close()
    return {"summary": metrics.summarize(), "frames": frames}


def _boxToDict(box: BBoxXYWH) -> dict[str, float]:
    return asdict(box)


def _resolveOutputDir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    repositoryRoot = Path(__file__).resolve().parents[1]
    dataRoot = (repositoryRoot / "data").resolve()
    datasetPath = Path(args.dataset_root).expanduser().resolve()
    if args.sequence:
        datasetPath = datasetPath / args.sequence
    try:
        relativeDataset = datasetPath.relative_to(dataRoot)
    except ValueError:
        relativeDataset = Path(datasetPath.name)
    outputParent = repositoryRoot / "artifacts" / relativeDataset
    existing = [
        int(path.name.removeprefix("output_"))
        for path in outputParent.glob("output_*")
        if path.is_dir() and path.name.removeprefix("output_").isdigit()
    ]
    nextIndex = max(existing, default=0) + 1
    return outputParent / f"output_{nextIndex}"


def _instanceCatalog(frame) -> list[dict[str, object]]:
    if frame.segmentation is None or frame.segmentation.instance is None:
        return []
    instance = frame.segmentation.instance
    semantic = frame.segmentation.semantic
    height, width = instance.shape
    ids, counts = np.unique(instance, return_counts=True)
    catalog: list[dict[str, object]] = []
    for instanceId, count in zip(ids, counts):
        value = int(instanceId)
        if value == 0:
            continue
        mask = instance == value
        ys, xs = np.where(mask)
        xPx, widthPx = minimalCircularInterval(xs.astype(np.float64), width)
        semanticName = None
        semanticId = None
        if semantic is not None:
            semanticIds, semanticCounts = np.unique(semantic[mask], return_counts=True)
            semanticId = int(semanticIds[int(np.argmax(semanticCounts))])
            semanticName = frame.segmentation.classNames.get(semanticId)
        catalog.append(
            {
                "instanceId": value,
                "pixels": int(count),
                "bbox": {
                    "xPx": xPx,
                    "yPx": float(ys.min()),
                    "widthPx": widthPx,
                    "heightPx": float(ys.max() - ys.min() + 1),
                },
                "semanticId": semanticId,
                "semanticName": semanticName,
                "frameFraction": float(count) / float(height * width),
            }
        )
    return sorted(catalog, key=lambda item: int(item["pixels"]), reverse=True)


if __name__ == "__main__":
    raise SystemExit(main())
