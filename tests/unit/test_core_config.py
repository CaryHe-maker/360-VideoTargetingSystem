import tempfile
import unittest
from pathlib import Path

from track360.core.config import loadConfig
from track360.core.errors import ConfigError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class CoreConfigTest(unittest.TestCase):
    def testLoadConfigConvertsAnglesAndResolvesWeights(self) -> None:
        config = loadConfig(REPOSITORY_ROOT / "configs" / "default.yaml")

        self.assertAlmostEqual(config.geometry.minFovRad, 0.3490658503988659)
        self.assertEqual(config.geometry.boundarySamplesPerEdge, 65)
        self.assertEqual(config.tracking.windowLength, 5)
        self.assertTrue(config.tracking.sameFrameEscalationEnabled)
        self.assertEqual(config.tracking.maxAttemptsPerFrame, 2)
        self.assertEqual(config.tracking.maxViewsPerFrameTotal, 12)
        self.assertEqual(config.tracking.reacquireCooldownFrames, 2)
        self.assertAlmostEqual(config.evaluator.supportWeight, 0.25)
        self.assertEqual(config.evaluator.minReacquireViews, 2)
        self.assertAlmostEqual(config.evaluator.successRate, 0.90)
        self.assertAlmostEqual(config.evaluator.overlapThreshold, 0.70)
        self.assertAlmostEqual(config.evaluator.fusionSourceMinConfidence, 0.740642)
        self.assertEqual(config.evaluator.fusionBoxMode, "best_source")
        self.assertEqual(config.motion.minSamplesForVelocity, 2)
        self.assertAlmostEqual(config.motion.processNoiseRadPerSec, 0.04)
        self.assertEqual(config.model.precision, "fp32")
        self.assertEqual(
            config.model.weights,
            REPOSITORY_ROOT / "models" / "artrackv2_b_256.pth.tar",
        )
        self.assertIsNone(config.scoring.calibrationArtifact)
        self.assertTrue(config.scoring.requireCheckpointHashMatch)
        self.assertFalse(config.visualization.enabled)
        self.assertEqual(
            config.visualization.outputRoot,
            REPOSITORY_ROOT / "outputs" / "visualization",
        )
        self.assertEqual(
            config.visualization.stages,
            frozenset({"local_rgb", "backend_box", "geometry_box"}),
        )

    def testLoadConfigRejectsUnknownFields(self) -> None:
        source = (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
        source = source.replace("schemaVersion: 1", "schemaVersion: 1\nunknownField: true")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(source, encoding="utf-8")
            with self.assertRaises(ConfigError):
                loadConfig(path)

    def testLoadConfigRejectsRemovedFixedStateThresholds(self) -> None:
        source = (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
        source = source.replace(
            "  candidateMinScore: 0.597262",
            "  candidateMinScore: 0.597262\n  uncertainThreshold: 0.45",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(source, encoding="utf-8")
            with self.assertRaises(ConfigError):
                loadConfig(path)

    def testLoadConfigRejectsBudgetThatCannotFitCubeMap(self) -> None:
        source = (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
        source = source.replace("maxViewsPerFrameTotal: 12", "maxViewsPerFrameTotal: 5")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(source, encoding="utf-8")
            with self.assertRaises(ConfigError):
                loadConfig(path)

    def testLoadConfigRejectsUnknownVisualizationStage(self) -> None:
        source = (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
        source = source.replace("    - geometry_box", "    - geometry_box\n    - unknown")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(source, encoding="utf-8")
            with self.assertRaises(ConfigError):
                loadConfig(path)


if __name__ == "__main__":
    unittest.main()
