import json
import struct
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

from track360.core.config import VisualizationConfig
from track360.core.types import (
    BBoxXYWH,
    BFoV,
    FrameIndex,
    FramePacket,
    LocalObservation,
    LocalView,
    ProjectedObservation,
    SegmentationPlane,
    SequenceId,
    SphericalPoint,
    ViewSpec,
)
from track360.datasets.instance_ids import (
    collectInstanceIdGroups,
    formatInstanceIdDocument,
    writeInstanceIdDocument,
)
from track360.visualization import (
    FLUORESCENT_GREEN_RGB,
    ResultVisualizationRecorder,
    TimeCounter,
    VisualizationRecorder,
    drawBoxRgb,
)


class VisualizationRecorderTest(unittest.TestCase):
    def testTimeCounterWritesProcessingDuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result" / "time.json"
            counter = TimeCounter()
            counter.start()
            counter.startProcessing()
            time.sleep(0.001)
            counter.stopProcessing()

            written = counter.stop(output)

            self.assertEqual(written, output)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], "track360.time.v1")
            self.assertEqual(payload["scope"], "tracking_processing")
            self.assertGreater(payload["elapsedNanoseconds"], 0)
            self.assertGreater(payload["elapsedSeconds"], 0.0)
            self.assertIn("startedAtUtc", payload)
            self.assertIn("finishedAtUtc", payload)

    def testTimeCounterExcludesSurroundingWork(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "time.json"
            counter = TimeCounter()
            counter.start()
            time.sleep(0.005)
            counter.startProcessing()
            time.sleep(0.001)
            counter.stopProcessing()
            time.sleep(0.005)

            counter.stop(output)

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertLess(payload["elapsedMilliseconds"], 4.0)

    def testTimeCounterReportsZeroWhenNoProcessingIntervalRan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "time.json"
            counter = TimeCounter()
            counter.start()

            counter.stop(output)

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["elapsedNanoseconds"], 0)

    def testDisabledRecorderDoesNotCreateOutputRoot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputRoot = Path(directory) / "visualization"
            recorder = _recorder(outputRoot, enabled=False)

            artifacts = recorder.recordLocalRgb(_frame(), [_view()])

            self.assertEqual(artifacts, ())
            self.assertFalse(outputRoot.exists())

    def testWritesLocalRgbWithoutChangingPixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = _recorder(Path(directory))
            frame = _frame()
            view = _view()
            localPaths = recorder.recordLocalRgb(frame, [view])

            np.testing.assert_array_equal(_readRgbPng(localPaths[0]), view.rgb)
            self.assertIn("frame_000007", localPaths[0].as_posix())
            self.assertIn("local_rgb", localPaths[0].as_posix())

    def testWritesBackendBoxOverLocalRgbInFluorescentGreen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = _recorder(Path(directory), stages={"backend_box"})
            frame = _frame()
            view = _view()
            originalRgb = view.rgb.copy()
            observation = _localObservation(BBoxXYWH(1.0, 1.0, 4.0, 3.0))

            with patch(
                "track360.visualization.recorder.drawBoxRgb", wraps=drawBoxRgb
            ) as draw:
                paths = recorder.recordBackendBoxes(frame, [view], [observation])

            annotated = _readRgbPng(paths[0])
            np.testing.assert_array_equal(annotated[1, 1], FLUORESCENT_GREEN_RGB)
            np.testing.assert_array_equal(annotated[3, 4], FLUORESCENT_GREEN_RGB)
            np.testing.assert_array_equal(view.rgb, originalRgb)
            draw.assert_called_once_with(
                view.rgb,
                observation.bbox,
                label="fuseScore=0.850/0.000",
            )

    def testWritesWrappedGeometryBoxOverOriginalErpRgb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = _recorder(Path(directory), stages={"geometry_box"})
            frame = _frame()
            observation = _projectedObservation(BBoxXYWH(10.0, 2.0, 4.0, 3.0))

            with patch(
                "track360.visualization.recorder.drawBoxRgb", wraps=drawBoxRgb
            ) as draw:
                paths = recorder.recordGeometryBoxes(frame, [observation])

            annotated = _readRgbPng(paths[0])
            np.testing.assert_array_equal(annotated[2, 10], FLUORESCENT_GREEN_RGB)
            np.testing.assert_array_equal(annotated[2, 0], FLUORESCENT_GREEN_RGB)
            np.testing.assert_array_equal(annotated[4, 1], FLUORESCENT_GREEN_RGB)
            np.testing.assert_array_equal(frame.rgb, np.zeros_like(frame.rgb))
            draw.assert_called_once_with(
                frame.rgb,
                observation.bbox,
                wrapHorizontal=True,
                label="score=0.850/0.700/0.800/1.00",
            )

    def testWritesStateEvaluatorScoreBelowFinalResultBox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = FramePacket(
                sequenceId=SequenceId("sequence/a"),
                frameIndex=FrameIndex(7),
                timestampNs=1,
                rgb=np.zeros((64, 160, 3), dtype=np.uint8),
            )
            result = _trackResult(BBoxXYWH(10.0, 10.0, 30.0, 20.0))
            recorder = ResultVisualizationRecorder(Path(directory))

            with patch(
                "track360.visualization.result.drawBoxRgb", wraps=drawBoxRgb
            ) as draw:
                path = recorder.record(frame, result, stateScore=0.625, roundCount=2)

            annotated = _readRgbPng(path)
            self.assertTrue(np.any(np.all(annotated[32:, :] == FLUORESCENT_GREEN_RGB, axis=2)))
            draw.assert_called_once_with(
                frame.rgb,
                result.bbox,
                wrapHorizontal=True,
                label="state=TRACKING/rounds=2/stateScore=0.6250",
            )

    def testFinalResultLabelSupportsEveryControllerState(self) -> None:
        rgb = np.zeros((64, 320, 3), dtype=np.uint8)
        bbox = BBoxXYWH(10.0, 10.0, 30.0, 20.0)

        for statusName in ("TRACKING", "UNCERTAIN", "LOST"):
            annotated = drawBoxRgb(
                rgb,
                bbox,
                label=f"state={statusName}/rounds=3/stateScore=0.6250",
            )
            self.assertTrue(np.any(np.all(annotated == FLUORESCENT_GREEN_RGB, axis=2)))

    def testInstanceIdDocumentOnlyUsesFrameZero(self) -> None:
        frame0 = _segmentationFrame(
            0,
            instance=np.array([[10, 10, 20, 0], [10, 20, 20, 0]], dtype=np.int32),
            semantic=np.array([[5, 5, 6, 3], [5, 6, 6, 3]], dtype=np.int32),
        )
        groups = collectInstanceIdGroups(frame0)

        self.assertEqual([group.semanticName for group in groups], ["concreteblock", "streetprops"])
        self.assertEqual(groups[0].instanceIds, (10,))
        self.assertEqual(groups[1].instanceIds, (20,))
        expected = "concreteblock 1 10\n\nstreetprops 1 20\n"
        self.assertEqual(formatInstanceIdDocument(groups), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = writeInstanceIdDocument(Path(directory) / "InstanceID.txt", groups)
            self.assertEqual(path.read_text(encoding="utf-8"), expected)

    def testInstanceIdDocumentRejectsNonInitialFrame(self) -> None:
        frame1 = _segmentationFrame(
            1,
            instance=np.array([[30]], dtype=np.int32),
            semantic=np.array([[6]], dtype=np.int32),
        )

        with self.assertRaisesRegex(Exception, "requires frameIndex 0"):
            collectInstanceIdGroups(frame1)


def _recorder(
    outputRoot: Path,
    *,
    enabled: bool = True,
    stages: set[str] | None = None,
) -> VisualizationRecorder:
    selectedStages = stages or {"local_rgb", "backend_box", "geometry_box"}
    return VisualizationRecorder(
        VisualizationConfig(
            enabled=enabled,
            outputRoot=outputRoot,
            stages=frozenset(selectedStages),
        )
    )


def _frame() -> FramePacket:
    return FramePacket(
        sequenceId=SequenceId("sequence/a"),
        frameIndex=FrameIndex(7),
        timestampNs=1,
        rgb=np.zeros((6, 12, 3), dtype=np.uint8),
    )


def _segmentationFrame(
    frameIndex: int,
    *,
    instance: np.ndarray,
    semantic: np.ndarray,
) -> FramePacket:
    return FramePacket(
        sequenceId=SequenceId("catalog"),
        frameIndex=FrameIndex(frameIndex),
        timestampNs=frameIndex,
        rgb=np.zeros((*instance.shape, 3), dtype=np.uint8),
        segmentation=SegmentationPlane(
            semantic=semantic,
            instance=instance,
            classNames={5: "concreteblock", 6: "streetprops"},
        ),
    )


def _view() -> LocalView:
    spec = ViewSpec(viewId=3, bfov=_bfov(), outputWidthPx=8, outputHeightPx=6)
    rgb = np.full((6, 8, 3), 40, dtype=np.uint8)
    return LocalView(spec=spec, rgb=rgb)


def _bfov() -> BFoV:
    center = SphericalPoint(x=1.0, y=0.0, z=0.0, yawRad=0.0, pitchRad=0.0)
    return BFoV(center=center, horizontalFovRad=1.0, verticalFovRad=1.0)


def _localObservation(bbox: BBoxXYWH) -> LocalObservation:
    return LocalObservation(
        viewId=3,
        bbox=bbox,
        modelScore=0.9,
        appearanceScore=0.8,
        fusedScore=0.85,
        latencyNs=1,
    )


def _projectedObservation(bbox: BBoxXYWH) -> ProjectedObservation:
    return ProjectedObservation(
        viewId=3,
        bfov=_bfov(),
        bbox=bbox,
        modelScore=0.9,
        appearanceScore=0.8,
        motionScore=0.7,
        scaleScore=0.6,
        fusedScore=0.85,
    )


def _trackResult(bbox: BBoxXYWH):
    from track360.core.types import ResultSource, TrackResult, TrackStatus

    return TrackResult(
        sequenceId=SequenceId("sequence/a"),
        frameIndex=FrameIndex(7),
        bbox=bbox,
        bfov=_bfov(),
        confidence=0.625,
        status=TrackStatus.TRACKING,
        valid=True,
        resultSource=ResultSource.OBSERVED_CONFIRMED,
    )


def _readRgbPng(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    offset = 8
    compressed = bytearray()
    widthPx = 0
    heightPx = 0
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            widthPx, heightPx = struct.unpack(">II", data[:8])
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"IEND":
            break
    rows = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(
        heightPx, 1 + widthPx * 3
    )
    np.testing.assert_array_equal(rows[:, 0], 0)
    return rows[:, 1:].reshape(heightPx, widthPx, 3)


if __name__ == "__main__":
    unittest.main()
