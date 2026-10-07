import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from track360.core.config import loadConfig
from track360.runtime.reproducibility import (
    RUN_METADATA_SUFFIX,
    configHash,
    configSnapshot,
    writeRunMetadata,
)

ROOT = Path(__file__).resolve().parents[2]


class ReproducibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = loadConfig(ROOT / "configs" / "default.yaml")

    def testConfigHashIgnoresMachineSpecificPaths(self) -> None:
        moved = replace(
            self.config,
            sourcePath=Path("elsewhere") / "default.yaml",
            model=replace(self.config.model, weights=Path("elsewhere") / "weights.pth.tar"),
            visualization=replace(self.config.visualization, outputRoot=Path("elsewhere")),
        )

        self.assertEqual(configHash(moved), configHash(self.config))

    def testConfigHashChangesWithAnyTuningValue(self) -> None:
        changed = replace(
            self.config,
            backendTuning=replace(self.config.backendTuning, templateFovScale=2.0),
        )

        self.assertNotEqual(configHash(changed), configHash(self.config))

    def testRunMetadataIsWrittenNextToTheResult(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config,
                model=replace(self.config.model, weights=Path(directory) / "missing.pth.tar"),
            )
            result = Path(directory) / "result.txt"
            metadataPath = writeRunMetadata(result, config)
            payload = json.loads(metadataPath.read_text(encoding="utf-8"))

        self.assertEqual(metadataPath.name, "result.txt" + RUN_METADATA_SUFFIX)
        self.assertEqual(payload["configHash"], configHash(config))
        self.assertEqual(payload["config"], configSnapshot(config))
        self.assertIsNone(payload["weightsSha256"])
        self.assertEqual(payload["config"]["backendTuning"]["templateFovScale"], 2.5)
        self.assertIn("commit", payload["git"])
        self.assertIn("numpy", payload["packages"])


if __name__ == "__main__":
    unittest.main()
