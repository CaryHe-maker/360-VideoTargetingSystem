import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from track360.controller.score_calibration import loadScoreCalibration
from track360.core.errors import ConfigError


class ScoreCalibrationArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporaryDirectory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporaryDirectory.name)
        self.checkpoint = self.root / "stage3.pth"
        self.checkpoint.write_bytes(b"stage3 checkpoint fixture")
        self.artifact = self.root / "stage3.calibration.json"
        self.payload = _artifactPayload(self.checkpoint)

    def tearDown(self) -> None:
        self.temporaryDirectory.cleanup()

    def testLoadsArtifactBoundToCheckpointAndThresholds(self) -> None:
        self._write(self.payload)

        result = self._load()

        self.assertEqual(result.appearanceInput, "presence_quality_product")
        self.assertEqual(result.appearanceWeight, 0.5)

    def testRejectsUnknownFields(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["oldBetaParameters"] = [1.0, 1.0, 0.0]
        self._write(payload)

        with self.assertRaisesRegex(ConfigError, "unknown=.*oldBetaParameters"):
            self._load()

    def testRejectsNonFiniteParameters(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["appearance"]["alpha"] = float("nan")
        self._write(payload)

        with self.assertRaisesRegex(ConfigError, "appearance.alpha must be finite"):
            self._load()

    def testRejectsCheckpointHashMismatch(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["checkpointSha256"] = "0" * 64
        self._write(payload)

        with self.assertRaisesRegex(ConfigError, "checkpoint hash mismatch"):
            self._load()

    def testRejectsControllerThresholdMismatch(self) -> None:
        self._write(self.payload)

        with self.assertRaisesRegex(ConfigError, "candidateMinScore does not match"):
            loadScoreCalibration(
                self.artifact,
                checkpointPath=self.checkpoint,
                candidateMinScore=0.6,
                fusionSourceMinConfidence=0.740642,
            )

        with self.assertRaisesRegex(ConfigError, "fusionSourceMinConfidence does not match"):
            loadScoreCalibration(
                self.artifact,
                checkpointPath=self.checkpoint,
                candidateMinScore=0.597262,
                fusionSourceMinConfidence=0.75,
            )

    def _write(self, payload: dict[str, object]) -> None:
        self.artifact.write_text(json.dumps(payload), encoding="utf-8")

    def _load(self):
        return loadScoreCalibration(
            self.artifact,
            checkpointPath=self.checkpoint,
            candidateMinScore=0.597262,
            fusionSourceMinConfidence=0.740642,
        )


def _artifactPayload(checkpoint: Path) -> dict[str, object]:
    return {
        "format": "track360.score-calibration.v1",
        "checkpointSha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "manifestSha256": "1" * 64,
        "split": "calibration",
        "appearanceInput": "presence_quality_product",
        "appearance": {
            "method": "beta",
            "alpha": 0.9934308915428243,
            "beta": 1.8582728355600815,
            "intercept": 0.6623364310412592,
        },
        "singleScore": {"appearanceWeight": 0.5, "motionWeight": 0.5},
        "thresholds": {
            "candidateMinScore": 0.597262,
            "fusionSourceMinConfidence": 0.740642,
        },
        "fit": {
            "sampleCount": 100,
            "positiveCount": 50,
            "negativeCount": 50,
            "sequenceCount": 2,
            "rawBrier": 0.2,
            "calibratedBrier": 0.1,
            "rawEce": 0.15,
            "calibratedEce": 0.05,
            "prAuc": 0.8,
            "rocAuc": 0.85,
        },
    }


if __name__ == "__main__":
    unittest.main()
