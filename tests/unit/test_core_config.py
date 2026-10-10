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
        self.assertFalse(config.visualization.enabled)
        self.assertFalse(config.backendTuning.lossHandling)
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
        source = source.replace("schemaVersion: 2", "schemaVersion: 2\nunknownField: true")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(source, encoding="utf-8")
            with self.assertRaises(ConfigError):
                loadConfig(path)

    def testLoadConfigRejectsRemovedFixedStateThresholds(self) -> None:
        source = (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
        source = source.replace(
            "  windowLength: 5",
            "  windowLength: 5\n  uncertainThreshold: 0.45",
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
        self._assertRejected("  lossHandling: false", "  lossHandling: 1")
        self._assertRejected("  lossActions: jump", "  lossActions: scan")

    def testLoadConfigRejectsOptionsRemovedBeforeTheRelease(self) -> None:
        for removed in (
            "  acceptAnyCandidate: true\n",
            "  useMotionScore: false\n",
            "  stateRule: relative\n",
            "  scanMode: zoom\n",
            "  zoomSpread: 2\n",
        ):
            self._assertRejected("backendTuning:\n", "backendTuning:\n" + removed)
        self._assertRejected("geometry:\n", "geometry:\n  resampler: opencv\n")
        self._assertRejected(
            "geometry:\n", "scoring:\n  calibrationArtifact: null\ngeometry:\n"
        )
        self._assertRejected("schemaVersion: 2", "schemaVersion: 1")

    def testTheShippedConfigsAreTheTwoBaselinesInTwoPrecisions(self) -> None:
        default = loadConfig(REPOSITORY_ROOT / "configs" / "default.yaml")
        loss = loadConfig(REPOSITORY_ROOT / "configs" / "loss_handling.yaml")
        fast = loadConfig(REPOSITORY_ROOT / "configs" / "fast_tf32.yaml")
        lossFast = loadConfig(REPOSITORY_ROOT / "configs" / "loss_handling_tf32.yaml")
        self.assertTrue(loss.backendTuning.lossHandling)
        self.assertAlmostEqual(loss.backendTuning.scanBudgetPerFrame, 0.5)
        self.assertEqual(fast.model.precision, "tf32")
        # The fast files differ from their baselines in the precision alone.
        self.assertEqual(fast.backendTuning, default.backendTuning)
        self.assertEqual(lossFast.backendTuning, loss.backendTuning)
        self.assertEqual(lossFast.model.precision, "tf32")

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
