"""The benchmark runner end to end on a small synthetic 360VOT-style dataset."""

import json
import tempfile
import unittest
from dataclasses import replace
from math import degrees
from pathlib import Path

import cv2
import numpy as np
from synthetic_runtime import FRAME_HEIGHT, FRAME_WIDTH, FakeARTrackSession, buildScenario

from track360.core.config import loadConfig
from track360.core.errors import DecodeError, ProtocolError
from track360.core.types import BBoxXYWH
from track360.geometry import SphericalGeometryImpl
from track360.io.vot360_results import readResultFile, resultPaths
from track360.runtime.benchmark import METHODS, evaluateResults, methodConfig, runBenchmark

ROOT = Path(__file__).resolve().parents[2]
FRAME_COUNT = 8
ABSENT = {"cx": 0, "cy": 0, "w": 0, "h": 0, "rotation": 0}
ABSENT_BFOV = {"clon": 0, "clat": 0, "fov_h": 0, "fov_v": 0, "rotation": 0}


def _writeSequence(root: Path, name: str, scenario: str) -> None:
    """Store the first frames of a synthetic scenario with labels from the drawn target."""
    geometry = SphericalGeometryImpl()
    frames, _ = buildScenario(scenario)
    images = root / name / "image"
    images.mkdir(parents=True)
    labels = {}
    for index, frame in enumerate(frames[:FRAME_COUNT]):
        fileName = f"{index:06d}.jpg"
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        assert cv2.imwrite(str(images / fileName), bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
        mask = (frame.rgb[..., 0] > 200) & (frame.rgb[..., 1] < 60)
        rows = np.flatnonzero(mask.any(axis=1))
        columns = np.flatnonzero(mask.any(axis=0))
        box = BBoxXYWH(
            float(columns[0]),
            float(rows[0]),
            float(columns[-1] + 1 - columns[0]),
            float(rows[-1] + 1 - rows[0]),
        )
        bfov = geometry.bboxToBfov(box, FRAME_WIDTH, FRAME_HEIGHT)
        pixelBox = {
            "cx": box.xPx + box.widthPx / 2.0 - 0.5,
            "cy": box.yPx + box.heightPx / 2.0 - 0.5,
            "w": box.widthPx,
            "h": box.heightPx,
            "rotation": 0,
        }
        sphereBox = {
            "clon": degrees(bfov.center.yawRad),
            "clat": degrees(bfov.center.pitchRad),
            "fov_h": degrees(bfov.horizontalFovRad),
            "fov_v": degrees(bfov.verticalFovRad),
            "rotation": 0,
        }
        labels[fileName] = {
            "bbox": pixelBox,
            "rbbox": pixelBox,
            "bfov": sphereBox,
            "rbfov": sphereBox,
        }
    (root / name / "label.json").write_text(json.dumps(labels), encoding="utf-8")


class BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.dataset = Path(cls._directory.name) / "dataset"
        cls.output = Path(cls._directory.name) / "output"
        _writeSequence(cls.dataset, "0001", "seam")
        _writeSequence(cls.dataset, "0002", "large")
        cls.config = loadConfig(ROOT / "configs" / "default.yaml")
        cls.sessions: list[FakeARTrackSession] = []

        def sessionFactory(modelConfig):
            session = FakeARTrackSession(modelConfig)
            cls.sessions.append(session)
            return session

        cls.sessionFactory = staticmethod(sessionFactory)
        cls.summaries = {
            method: runBenchmark(
                datasetRoot=cls.dataset,
                outputRoot=cls.output,
                method=method,
                config=cls.config,
                sessionFactory=sessionFactory,
            )
            for method in METHODS
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def testEveryMethodWritesCompleteResultFiles(self) -> None:
        for method, summary in self.summaries.items():
            with self.subTest(method=method):
                self.assertEqual([report.status for report in summary.reports], ["done"] * 2)
                self.assertEqual(summary.failures, ())
                for name in ("0001", "0002"):
                    bboxPath, bfovPath = resultPaths(self.output, method, name)
                    self.assertEqual(readResultFile(bboxPath).shape, (FRAME_COUNT, 4))
                    self.assertEqual(readResultFile(bfovPath).shape, (FRAME_COUNT, 5))
                report = summary.reports[0]
                self.assertEqual(report.frames, FRAME_COUNT)
                self.assertGreater(report.fps, 0.0)
                self.assertGreater(report.p95LatencyMs, 0.0)

    def testOneModelSessionServesAllSequencesOfAMethod(self) -> None:
        self.assertEqual(len(self.sessions), len(METHODS))
        for session in self.sessions:
            self.assertEqual([call["op"] for call in session.calls].count("close"), 1)
            self.assertEqual(session.calls[-1]["op"], "close")

    def testMethodsDifferInHowTheySearch(self) -> None:
        ours, single, direct = self.sessions
        self.assertEqual({c["views"] for c in ours.calls if c["op"] == "inferBatch"}, {4})
        self.assertEqual({c["views"] for c in single.calls if c["op"] == "inferBatch"}, {1})
        self.assertFalse([c for c in direct.calls if c["op"] == "inferBatch"])
        erpCalls = [c for c in direct.calls if c["op"] == "inferErp"]
        self.assertEqual(len(erpCalls), 2 * (FRAME_COUNT - 1))
        self.assertTrue(all(call["priors"] for call in erpCalls))
        self.assertTrue(methodConfig(self.config, "b2").backendTuning.singleView)
        self.assertEqual(methodConfig(self.config, "ours"), self.config)

    def testRunReportsRecordTheEffectiveConfiguration(self) -> None:
        run = json.loads(
            (self.output / "reports" / "b2" / "run.json").read_text(encoding="utf-8")
        )
        report = json.loads(
            (self.output / "reports" / "b2" / "0001.json").read_text(encoding="utf-8")
        )

        self.assertEqual(run["method"], "b2")
        self.assertEqual(run["sequences"], ["0001", "0002"])
        self.assertTrue(run["config"]["backendTuning"]["singleView"])
        self.assertIn("configHash", run)
        self.assertEqual(report["status"], "done")
        self.assertEqual(report["frames"], FRAME_COUNT)

    def testScoresAreComputedForEveryMethodAndRepresentation(self) -> None:
        scores = evaluateResults(datasetRoot=self.dataset, outputRoot=self.output)

        self.assertEqual(sorted(scores), sorted(METHODS))
        for method in METHODS:
            with self.subTest(method=method):
                self.assertEqual(sorted(scores[method]), ["bbox", "bfov"])
                bbox = scores[method]["bbox"]
                self.assertEqual((bbox.sequenceCount, bbox.frameCount), (2, 2 * FRAME_COUNT))
                # The fake backend finds the drawn target, so every method tracks it.
                self.assertGreater(bbox.success, 0.5)
                self.assertGreater(scores[method]["bfov"].anglePrecision, 0.5)
        only = evaluateResults(datasetRoot=self.dataset, outputRoot=self.output, methods=["b0"])
        self.assertEqual(list(only), ["b0"])
        with self.assertRaisesRegex(DecodeError, "no bbox results for method"):
            evaluateResults(datasetRoot=self.dataset, outputRoot=self.output, methods=["none"])

    def testResumeSkipsCompleteSequencesAndRerunsIncompleteOnes(self) -> None:
        bboxPath, _ = resultPaths(self.output, "ours", "0002")
        bboxPath.write_text("1,2,3,4\n", encoding="utf-8")

        summary = runBenchmark(
            datasetRoot=self.dataset,
            outputRoot=self.output,
            method="ours",
            config=self.config,
            sessionFactory=FakeARTrackSession,
        )

        self.assertEqual([report.status for report in summary.reports], ["skipped", "done"])
        self.assertEqual(readResultFile(bboxPath).shape, (FRAME_COUNT, 4))

    def testSequenceSelectionShardingAndFrameLimit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = runBenchmark(
                datasetRoot=self.dataset,
                outputRoot=directory,
                method="b0",
                config=self.config,
                maxFrames=3,
                shard=(1, 2),
                sessionFactory=FakeARTrackSession,
            )
            self.assertEqual([report.sequence for report in summary.reports], ["0002"])
            self.assertEqual(summary.reports[0].frames, 3)
            with self.assertRaisesRegex(ProtocolError, "result length"):
                evaluateResults(datasetRoot=self.dataset, outputRoot=directory)
            partial = evaluateResults(
                datasetRoot=self.dataset, outputRoot=directory, allowPartial=True
            )
            self.assertEqual(partial["b0"]["bbox"].frameCount, 3)
        for arguments in ({"sequences": ["9999"]}, {"shard": (2, 2)}, {"method": "unknown"}):
            options = {"method": "b0", **arguments}
            with self.subTest(arguments=arguments), self.assertRaises((DecodeError, ProtocolError)):
                runBenchmark(
                    datasetRoot=self.dataset,
                    outputRoot=self.output,
                    config=self.config,
                    sessionFactory=FakeARTrackSession,
                    **options,
                )

    def testAFailingSequenceDoesNotStopTheRun(self) -> None:
        class _FailsOnSecondSequence(FakeARTrackSession):
            encoded = 0

            def encodeTemplate(self, rgb, bbox):
                type(self).encoded += 1
                if type(self).encoded == 1:
                    raise RuntimeError("injected failure")
                return super().encodeTemplate(rgb, bbox)

        with tempfile.TemporaryDirectory() as directory:
            summary = runBenchmark(
                datasetRoot=self.dataset,
                outputRoot=directory,
                method="b0",
                config=replace(self.config),
                sessionFactory=_FailsOnSecondSequence,
            )

            self.assertEqual([report.status for report in summary.reports], ["failed", "done"])
            self.assertIn("injected failure", summary.failures[0].error)
            self.assertFalse(resultPaths(directory, "b0", "0001")[0].exists())
            self.assertTrue(resultPaths(directory, "b0", "0002")[0].exists())


if __name__ == "__main__":
    unittest.main()
