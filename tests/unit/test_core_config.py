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
        self.assertEqual(config.geometry.resampler, "opencv")
        self.assertTrue(config.backendTuning.acceptAnyCandidate)
        self.assertAlmostEqual(config.backendTuning.viewHorizontalFovCapRad, 1.5707963267948966)
        self.assertAlmostEqual(config.backendTuning.viewVerticalFovCapRad, 1.5707963267948966)
        self.assertEqual(config.reproducibility.seed, 0)
        self.assertTrue(config.reproducibility.deterministic)
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

    def testLoadConfigRejectsRemovedMultiViewSettings(self) -> None:
        for old, addition in (
            ("motion:\n", "evaluator:\n  fusionBoxMode: best_source\n"),
            ("backendTuning:\n", "recovery:\n  maxViewsPerFrame: 12\n"),
            ("  windowLength: 5\n", "  maxAttemptsPerFrame: 2\n"),
            ("  sphericalSearch: true\n", "  singleView: true\n"),
        ):
            self._assertRejected(old, addition + old)

    def testLoadConfigRejectsUnknownVisualizationStage(self) -> None:
        source = (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
        source = source.replace("    - geometry_box", "    - geometry_box\n    - unknown")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(source, encoding="utf-8")
            with self.assertRaises(ConfigError):
                loadConfig(path)

    def testLoadConfigRejectsUnimplementedPrecision(self) -> None:
        self._assertRejected("  precision: fp32", "  precision: fp16")

    def testLoadConfigRejectsRemovedSections(self) -> None:
        removed = "decisionGate:\n  motionScoreWeight: 0.25\n  scaleScoreWeight: 0.15\n"
        self._assertRejected("motion:\n", removed + "motion:\n")

    def testLoadConfigRejectsInvalidBackendTuning(self) -> None:
        self._assertRejected("  templateFovScale: 2.5", "  templateFovScale: 0.5")
        self._assertRejected("  viewHorizontalFovCapDeg: 90.0", "  viewHorizontalFovCapDeg: 200.0")
        self._assertRejected("  holdWeakBox: true", "  holdWeakBox: 1")
        self._assertRejected("  resampler: opencv", "  resampler: gpu")

    def _assertRejected(self, old: str, new: str) -> None:
        source = (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
        self.assertEqual(source.count(old), 1, old)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(source.replace(old, new), encoding="utf-8")
            with self.assertRaises(ConfigError):
                loadConfig(path)


if __name__ == "__main__":
    unittest.main()
