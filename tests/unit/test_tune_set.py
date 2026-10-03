import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from track360.core.errors import DecodeError
from track360.datasets.mask_labels import labelFromMask, readTargetMask, writeSequenceLabels
from track360.datasets.tune_split import (
    excludedTrainClips,
    leakFreePool,
    readSequenceFile,
    selectTuneSet,
)
from track360.datasets.vot360 import Vot360Dataset
from track360.datasets.vots_info import loadVotsInfo

WIDTH = 360
HEIGHT = 180
HEADER = (
    "Sequence (User can ignore),360VOS ID,Target,Split,Muptiple,360VOT ID,"
    "IV(illumination variation),CB(cross border),FM(fast motion)"
)
INFO = f"""{HEADER}
aaaaaaaaaaa_0-10_0-35,001,bear,train,,1,,1,
bbbbbbbbbbb_start_0-39,002,bird,train,,,1,,
aaaaaaaaaaa_1-00_1-30,003,bear,train,,,,1,1
ccccccccccc,004,boat,train,1,,1,,1
ddddddddddd_2-40_end,005,car,train,,,,,1
e_eeeeeeeee_00-45_01-45,006,dog,train,,,1,1,1
fffffffffff,001,fox,test,,2,1,,
ggggggggggg,,goat,none,,3,,1,
test+train,,,,,,,,
"""


def _mask(columns: slice | list[int], rows: slice) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[rows, columns] = True
    return mask


def _writeTrainArchive(root: Path, name: str, masks: list[np.ndarray]) -> None:
    """A 360VOS training archive: ``image/`` and ``mask/`` at the top level, no labels."""
    with zipfile.ZipFile(root / f"{name}.zip", "w") as archive:
        for index, mask in enumerate(masks):
            image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            archive.writestr(f"image/{index:06d}.jpg", cv2.imencode(".jpg", image)[1].tobytes())
            colored = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            colored[mask] = (0, 0, 128)  # BGR for the first object's RGB (128, 0, 0)
            archive.writestr(f"mask/{index:06d}.png", cv2.imencode(".png", colored)[1].tobytes())


class VotsInfoTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        path = Path(self._directory.name) / "info.csv"
        path.write_text(INFO, encoding="utf-8")
        self.info = loadVotsInfo(path)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def testTableRowsAndAttributeCodesAreParsed(self) -> None:
        self.assertEqual(self.info.attributeNames, ("IV", "CB", "FM"))
        self.assertEqual(len(self.info.clips), 8)
        first = self.info.clips[0]
        self.assertEqual((first.vosId, first.votId, first.vosSplit), ("001", "0001", "train"))
        self.assertEqual(first.attributes, frozenset({"CB"}))
        self.assertTrue(self.info.clips[3].multipleTargets)
        self.assertEqual(
            self.info.votAttributes(),
            {"0001": {"CB"}, "0002": {"IV"}, "0003": {"CB"}},
        )
        self.assertEqual(self.info.vosAttributes("train")["006"], {"IV", "CB", "FM"})

    def testSourceVideoDropsTheTimeRange(self) -> None:
        videos = [clip.sourceVideo for clip in self.info.clips]
        self.assertEqual(videos[0], "aaaaaaaaaaa")
        self.assertEqual(videos[1], "bbbbbbbbbbb")
        self.assertEqual(videos[4], "ddddddddddd")
        # An underscore inside the video ID is not a time range.
        self.assertEqual(videos[5], "e_eeeeeeeee")

    def testClipsThatCouldLeakTheTestSetAreExcluded(self) -> None:
        excluded = {
            item.vosId: (item.reason, item.detail) for item in excludedTrainClips(self.info)
        }

        self.assertEqual(
            excluded,
            {
                "001": ("is_360vot_sequence", "0001"),
                "003": ("same_video_as_360vot_sequence", "0001"),
                "004": ("multiple_targets", ""),
            },
        )
        self.assertEqual([clip.vosId for clip in leakFreePool(self.info)], ["002", "005", "006"])

    def testTuneSelectionCoversAttributesAndSkipsLongClips(self) -> None:
        pool = leakFreePool(self.info)
        frames = {"002": 100, "005": 50, "006": 900}

        # 006 has the most attributes; the other two then tie, and the shorter one wins.
        two = selectTuneSet(pool, frames, size=2, maxFrames=1000)
        short = selectTuneSet(pool, frames, size=2, maxFrames=500)

        self.assertEqual([clip.vosId for clip in two], ["005", "006"])
        self.assertEqual([clip.vosId for clip in short], ["002", "005"])
        self.assertEqual(len(selectTuneSet(pool, frames, size=9, maxFrames=1000)), 3)

    def testSequenceFileIgnoresComments(self) -> None:
        path = Path(self._directory.name) / "tune.txt"
        path.write_text("# header\n006  # 931, boat\n\n009\n", encoding="utf-8")

        self.assertEqual(readSequenceFile(path), ["006", "009"])
        with self.assertRaises(DecodeError):
            readSequenceFile(path.with_name("missing.txt"))


class MaskLabelTest(unittest.TestCase):
    def testBoxAndBfovFollowTheMask(self) -> None:
        label = labelFromMask(_mask(slice(170, 190), slice(80, 100)))

        self.assertEqual(
            label["bbox"], {"cx": 179.5, "cy": 89.5, "w": 20.0, "h": 20.0, "rotation": 0}
        )
        self.assertAlmostEqual(label["bfov"]["clon"], 0.0, places=6)
        self.assertAlmostEqual(label["bfov"]["clat"], 0.0, places=6)
        # 20 of 360 columns at the equator span 20 degrees, minus half a pixel per side.
        self.assertAlmostEqual(label["bfov"]["fov_h"], 19.0, delta=0.2)
        self.assertEqual(label["rbfov"], label["bfov"])

    def testTargetOnTheSeamStaysOneBox(self) -> None:
        label = labelFromMask(_mask(list(range(350, 360)) + list(range(0, 10)), slice(80, 100)))

        self.assertEqual(label["bbox"]["w"], 20.0)
        self.assertEqual(label["bbox"]["cx"], 359.5)  # runs past the right border
        self.assertAlmostEqual(abs(label["bfov"]["clon"]), 180.0, delta=0.01)
        self.assertLess(label["bfov"]["fov_h"], 21.0)

    def testEmptyMaskIsAnAbsentTarget(self) -> None:
        label = labelFromMask(np.zeros((HEIGHT, WIDTH), dtype=bool))

        self.assertEqual(label["bbox"]["w"], 0)
        self.assertEqual(label["bfov"]["fov_h"], 0)

    def testTrainingArchiveIsReadWithLabelsFromALabelRoot(self) -> None:
        masks = [
            _mask(slice(100, 130), slice(60, 90)),
            np.zeros((HEIGHT, WIDTH), dtype=bool),
            _mask(slice(110, 140), slice(60, 90)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "train"
            labels = Path(directory) / "labels"
            root.mkdir()
            _writeTrainArchive(root, "006", masks)

            unlabelled = Vot360Dataset(root).sequence("006")
            self.assertFalse(unlabelled.hasLabels)
            self.assertEqual(unlabelled.frameCount, 3)
            np.testing.assert_array_equal(readTargetMask(unlabelled, 2), masks[2])
            path = writeSequenceLabels(unlabelled, labels)
            unlabelled.close()

            self.assertEqual(path, labels / "006" / "label.json")
            self.assertEqual(
                list(json.loads(path.read_text(encoding="utf-8"))),
                [f"{index:06d}.jpg" for index in range(3)],
            )
            sequence = Vot360Dataset(root, labels).sequence("006")
            self.assertTrue(sequence.hasLabels)
            self.assertEqual(sequence.readRgb(0).shape, (HEIGHT, WIDTH, 3))
            boxes, present = sequence.groundTruth("bbox")
            np.testing.assert_array_equal(present, [True, False, True])
            # Pixels 100..129 are [99.5, 129.5] as pixel indices and [100, 130] as edges.
            np.testing.assert_array_equal(boxes[0], [99.5, 59.5, 30.0, 30.0])
            self.assertEqual(sequence.initialBbox().xPx, 100.0)
            self.assertIsNone(sequence.annotation(1).bfov)
            sequence.close()

    def testSequenceWithoutATargetInFrameZeroIsRejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _writeTrainArchive(root, "007", [np.zeros((HEIGHT, WIDTH), dtype=bool)])
            sequence = Vot360Dataset(root).sequence("007")
            with self.assertRaisesRegex(DecodeError, "no target in frame 0"):
                writeSequenceLabels(sequence, root / "labels")
            sequence.close()


if __name__ == "__main__":
    unittest.main()
