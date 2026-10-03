"""Golden regression for the production runtime path on synthetic sequences.

The golden traces were recorded before the ``TRACK360_ARTRACK_*`` environment
switches moved into ``backendTuning``; every variant below reproduces one of the
recorded environment combinations through configuration alone.  Update a golden
file only for an intended behavior change, and say so in the commit.
"""

import hashlib
import json
import unittest
from dataclasses import replace
from math import pi
from pathlib import Path

import pytest
from synthetic_runtime import SCENARIOS, runScenario

from track360.core.config import AppConfig, BackendTuningConfig, loadConfig

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent / "golden"
FLOAT_TOLERANCE = 1e-6

VARIANT_OVERRIDES: dict[str, dict[str, object]] = {
    "adaptive": {"adaptiveViewCount": True},
    "gated": {"acceptAnyCandidate": False},
    "gated_single_round": {"acceptAnyCandidate": False, "singleRound": True},
    "gated_no_hold": {"acceptAnyCandidate": False, "holdWeakBox": False},
    "motion_on": {"useMotionScore": True},
    "single_view_caps": {
        "singleView": True,
        "singleViewHorizontalFovCapRad": 70.0 * pi / 180.0,
        "singleViewVerticalFovCapRad": 50.0 * pi / 180.0,
    },
    "direct": {"directMode": True},
    "mixed": {
        "fourViewFovCapRad": 60.0 * pi / 180.0,
        "fusionBoxMode": "reference_adaptive",
        "holdWeakBox": False,
        "allowSingleViewTemplate": False,
        "templateMinConfidence": 0.45,
        "templateFovScale": 3.0,
        "fusionSourceMinConfidence": 0.5,
        "fusionOverlap": 0.6,
    },
}


def _trace(config: AppConfig) -> dict[str, object]:
    return {scenario: runScenario(config, scenario) for scenario in SCENARIOS}


def _digest(trace: object) -> str:
    text = json.dumps(trace, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class GoldenRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.defaultConfig = loadConfig(ROOT / "configs" / "default.yaml")
        cls.legacyConfig = loadConfig(ROOT / "configs" / "tests" / "legacy_off.yaml")
        cls.digests = json.loads((GOLDEN / "digests.json").read_text(encoding="utf-8"))
        cls.defaultTrace = _trace(cls.defaultConfig)

    def testDataclassDefaultsMatchDefaultYaml(self) -> None:
        self.assertEqual(BackendTuningConfig(), self.defaultConfig.backendTuning)

    def testDefaultConfigMatchesGolden(self) -> None:
        self._assertMatchesGolden(self.defaultTrace, "default")

    def testLegacyOffConfigMatchesGolden(self) -> None:
        self._assertMatchesGolden(_trace(self.legacyConfig), "legacy_off")

    @pytest.mark.slow
    def testTuningVariantsMatchRecordedDigests(self) -> None:
        # Digests are exact, so they only apply where this platform reproduces the
        # recorded default trace bit for bit.
        if _digest(self.defaultTrace) != self.digests["default"]:
            self.skipTest("platform is not bit-compatible with the recorded golden traces")
        self.assertEqual(_digest(_trace(self.legacyConfig)), self.digests["legacy_off"])
        for name, overrides in VARIANT_OVERRIDES.items():
            with self.subTest(variant=name):
                config = replace(
                    self.defaultConfig,
                    backendTuning=replace(self.defaultConfig.backendTuning, **overrides),
                )
                self.assertEqual(_digest(_trace(config)), self.digests[name])

    def _assertMatchesGolden(self, actual: object, name: str) -> None:
        expected = json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))
        self._assertClose(actual, expected, name)

    def _assertClose(self, actual: object, expected: object, path: str) -> None:
        if isinstance(expected, dict):
            self.assertIsInstance(actual, dict, path)
            self.assertEqual(sorted(actual), sorted(expected), path)
            for key, value in expected.items():
                self._assertClose(actual[key], value, f"{path}.{key}")
        elif isinstance(expected, list):
            self.assertIsInstance(actual, list, path)
            self.assertEqual(len(actual), len(expected), path)
            for index, value in enumerate(expected):
                self._assertClose(actual[index], value, f"{path}[{index}]")
        elif isinstance(expected, float):
            self.assertAlmostEqual(actual, expected, delta=FLOAT_TOLERANCE, msg=path)
        else:
            self.assertEqual(actual, expected, path)


if __name__ == "__main__":
    unittest.main()
