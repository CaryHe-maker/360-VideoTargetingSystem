import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from track360.datasets.airsim360_source import AirSim360DataSource
from track360.datasets.pseudo_track_builder import PseudoTrackBuilder
from track360.visualization.png import writeRgbPng


class AirSim360DataTest(unittest.TestCase):
    def testSourceAndPseudoBoxRoundTrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset" / "sequence_a"
            for name in ("rgb", "instance"):
                (root / name).mkdir(parents=True, exist_ok=True)
            writeRgbPng(root / "rgb" / "000000.png", np.full((6, 12, 3), 64, dtype=np.uint8))
            instance = np.zeros((6, 12), dtype=np.uint8)
            instance[1:4, 10:12] = 7
            instance[1:4, 0:2] = 7
            np.save(root / "instance" / "000000.npy", instance)
            (root / "meta.json").write_text(
                json.dumps({"classNames": {"7": "target"}}),
                encoding="utf-8",
            )

            source = AirSim360DataSource()
            source.open(str(root.parent), "sequence_a")
            frame = source.read()
            assert frame is not None
            bbox, visible = PseudoTrackBuilder().buildPseudoGroundTruth(frame, 7)

            self.assertTrue(visible)
            self.assertEqual(int(frame.frameIndex), 0)
            self.assertEqual(frame.segmentation.instance.shape, (6, 12))
            self.assertGreaterEqual(bbox.widthPx, 1.0)
