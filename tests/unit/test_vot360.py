import json
import math
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from track360.core.errors import DecodeError, ProtocolError
from track360.datasets import (
    Vot360Dataset,
    Vot360DataSource,
    openDataset,
    registeredDatasetFormats,
)
from track360.datasets.vot360 import FRAME_INTERVAL_NS

WIDTH = 64
HEIGHT = 32
ABSENT_BFOV = {"clon": 0, "clat": 0, "fov_h": 0, "fov_v": 0, "rotation": 0}
ABSENT_BBOX = {"cx": 0, "cy": 0, "w": 0, "h": 0, "rotation": 0}


def _label(clon: float, clat: float, cx: float, cy: float) -> dict[str, dict[str, float]]:
    return {
        "bfov": {"clon": clon, "clat": clat, "fov_h": 20.0, "fov_v": 10.0, "rotation": 0},
        "rbfov": {"clon": clon, "clat": clat, "fov_h": 22.0, "fov_v": 8.0, "rotation": -30.0},
        "bbox": {"cx": cx, "cy": cy, "w": 8, "h": 6, "rotation": 0},
        "rbbox": {"cx": cx, "cy": cy, "w": 9, "h": 5, "rotation": 30.0},
    }


LABELS = {
    "000000.jpg": _label(-90.0, 15.0, 15.5, 12.5),
    # The target is absent in the second frame.
    "000001.jpg": {
        "bfov": ABSENT_BFOV,
        "rbfov": ABSENT_BFOV,
        "bbox": ABSENT_BBOX,
        "rbbox": ABSENT_BBOX,
    },
    # The box runs past the left image border: it crosses the seam.
    "000002.jpg": _label(178.0, 0.0, 1.5, 15.5),
}


def _jpeg(value: int) -> bytes:
    bgr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    bgr[..., 2] = value  # red channel in BGR order
    encoded, payload = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert encoded
    return payload.tobytes()


def _writeDirectorySequence(root: Path, name: str) -> None:
    images = root / name / "image"
    images.mkdir(parents=True)
    for index, frameName in enumerate(LABELS):
        (images / frameName).write_bytes(_jpeg(60 * (index + 1)))
    (root / name / "label.json").write_text(json.dumps(LABELS), encoding="utf-8")


def _writeArchiveSequence(root: Path, name: str) -> None:
    with zipfile.ZipFile(root / f"{name}.zip", "w", zipfile.ZIP_STORED) as archive:
        for index, frameName in enumerate(LABELS):
            archive.writestr(f"{name}/image/{frameName}", _jpeg(60 * (index + 1)))
        archive.writestr(f"{name}/label.json", json.dumps(LABELS))


class Vot360Test(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        _writeDirectorySequence(self.root, "0001")
        _writeArchiveSequence(self.root, "0002")

    def tearDown(self) -> None:
        self._directory.cleanup()

    def testDatasetFindsDirectoriesAndArchives(self) -> None:
        dataset = Vot360Dataset(self.root)

        self.assertEqual(dataset.sequenceNames, ("0001", "0002"))
        self.assertEqual([sequence.frameCount for sequence in dataset], [3, 3])

    def testExtractedDirectoryWinsOverItsArchive(self) -> None:
        _writeArchiveSequence(self.root, "0001")

        sequence = Vot360Dataset(self.root).sequence("0001")

        self.assertTrue(sequence.location.is_dir())

    def testFramesDecodeAsRgbFromBothLayouts(self) -> None:
        for name in ("0001", "0002"):
            with self.subTest(sequence=name):
                sequence = Vot360Dataset(self.root).sequence(name)
                rgb = sequence.readRgb(1)
                sequence.close()

                self.assertEqual(sequence.frameSize, (WIDTH, HEIGHT))
                self.assertEqual(rgb.shape, (HEIGHT, WIDTH, 3))
                self.assertEqual(rgb.dtype, np.uint8)
                # JPEG is lossy, so allow a small error around the encoded red value.
                self.assertLess(abs(int(rgb[0, 0, 0]) - 120), 4)
                self.assertLess(int(rgb[0, 0, 2]), 4)

    def testAnnotationsUseProjectAnglesAndPixelEdgeBoxes(self) -> None:
        sequence = Vot360Dataset(self.root).sequence("0001")

        first = sequence.annotation(0)

        self.assertTrue(first.present)
        assert first.bfov is not None and first.rbfov is not None and first.bbox is not None
        self.assertAlmostEqual(first.bfov.center.yawRad, -math.pi / 2.0)
        self.assertAlmostEqual(first.bfov.center.pitchRad, math.radians(15.0))
        self.assertAlmostEqual(first.bfov.horizontalFovRad, math.radians(20.0))
        self.assertAlmostEqual(first.bfov.verticalFovRad, math.radians(10.0))
        self.assertEqual(first.bfov.rollRad, 0.0)
        self.assertAlmostEqual(first.rbfov.rollRad, math.radians(-30.0))
        # Pixel index 15.5 is the edge coordinate 16.0, which is yaw -90 degrees.
        self.assertEqual(
            (first.bbox.xPx, first.bbox.yPx, first.bbox.widthPx, first.bbox.heightPx),
            (12.0, 10.0, 8.0, 6.0),
        )
        self.assertEqual(sequence.initialBfov(), first.bfov)
        self.assertEqual(sequence.initialBbox(), first.bbox)

    def testAbsentTargetHasNoAnnotation(self) -> None:
        absent = Vot360Dataset(self.root).sequence("0001").annotation(1)

        self.assertFalse(absent.present)
        self.assertIsNone(absent.bfov)
        self.assertIsNone(absent.rbfov)
        self.assertIsNone(absent.bbox)

    def testSeamCrossingBoxWrapsIntoTheFrame(self) -> None:
        crossing = Vot360Dataset(self.root).sequence("0001").annotation(2).bbox

        assert crossing is not None
        self.assertEqual(crossing.xPx, 62.0)
        self.assertGreater(crossing.xPx + crossing.widthPx, WIDTH)

    def testGroundTruthMatchesTheOfficialToolkitLayout(self) -> None:
        sequence = Vot360Dataset(self.root).sequence("0001")

        bbox, bboxPresent = sequence.groundTruth("bbox")
        rbbox, _ = sequence.groundTruth("rbbox")
        bfov, bfovPresent = sequence.groundTruth("bfov")
        rbfov, _ = sequence.groundTruth("rbfov")

        np.testing.assert_array_equal(bbox[0], [11.5, 9.5, 8.0, 6.0])
        # The toolkit keeps coordinates outside the frame for seam-crossing boxes.
        np.testing.assert_array_equal(bbox[2], [-2.5, 12.5, 8.0, 6.0])
        np.testing.assert_array_equal(rbbox[0], [15.5, 12.5, 9.0, 5.0, 30.0])
        np.testing.assert_array_equal(bfov[0], [-90.0, 15.0, 20.0, 10.0, 0.0])
        np.testing.assert_array_equal(rbfov[0], [-90.0, 15.0, 22.0, 8.0, -30.0])
        np.testing.assert_array_equal(bboxPresent, [True, False, True])
        np.testing.assert_array_equal(bfovPresent, [True, False, True])
        self.assertEqual(bbox.shape, (3, 4))
        self.assertEqual(bfov.shape, (3, 5))

    def testDataSourceYieldsOrderedFramePackets(self) -> None:
        self.assertIn("360vot", registeredDatasetFormats())
        source = openDataset(str(self.root), format="360vot", sequenceId="0002")

        frames = []
        while (frame := source.read()) is not None:
            frames.append(frame)
        source.close()

        self.assertEqual(source.frameCount, 0)
        self.assertEqual([int(frame.frameIndex) for frame in frames], [0, 1, 2])
        self.assertEqual({str(frame.sequenceId) for frame in frames}, {"0002"})
        self.assertEqual(frames[2].timestampNs, 2 * FRAME_INTERVAL_NS)

    def testDataSourceHonorsMaxFrames(self) -> None:
        source = Vot360DataSource(maxFrames=2)
        source.open(str(self.root), "0001")

        self.assertEqual(source.frameCount, 2)
        self.assertIsNotNone(source.read())
        self.assertIsNotNone(source.read())
        self.assertIsNone(source.read())
        self.assertEqual(source.sequence.initialBfov().verticalFovRad, math.radians(10.0))
        source.close()

    def testDataSourceRequiresASequenceNameWhenAmbiguous(self) -> None:
        with self.assertRaisesRegex(DecodeError, "sequence name is required"):
            Vot360DataSource().open(str(self.root))
        with self.assertRaisesRegex(DecodeError, "unknown 360VOT sequence"):
            Vot360DataSource().open(str(self.root), "9999")
        with self.assertRaises(ProtocolError):
            Vot360DataSource().read()

    def testInvalidDataIsReported(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "out of range"):
            Vot360Dataset(self.root).sequence("0001").annotation(3)

        broken = dict(LABELS)
        broken["000002.jpg"] = {"bfov": ABSENT_BFOV}
        (self.root / "0001" / "label.json").write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaisesRegex(DecodeError, "lacks a valid 'bbox': 0001/000002.jpg"):
            Vot360Dataset(self.root).sequence("0001").annotation(0)

        (self.root / "0001" / "image" / "000000.jpg").write_bytes(b"not an image")
        (self.root / "0001" / "label.json").write_text(json.dumps(LABELS), encoding="utf-8")
        with self.assertRaisesRegex(DecodeError, "cannot decode image"):
            Vot360Dataset(self.root).sequence("0001").readRgb(0)

        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(DecodeError, "no 360VOT sequences"):
                Vot360Dataset(empty)


if __name__ == "__main__":
    unittest.main()
