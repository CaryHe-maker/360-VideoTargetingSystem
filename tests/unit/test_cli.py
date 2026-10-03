import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from track360.cli import DEFAULT_CONFIG, _withDefaultConfig, main
from track360.runtime.track_airsim360 import EXIT_CONFIG
from track360.runtime.track_airsim360 import main as trackAirSim360Main


class CliTest(unittest.TestCase):
    def testDefaultConfigShipsWithRepository(self) -> None:
        self.assertTrue(DEFAULT_CONFIG.is_file())

    def testDefaultConfigIsAppendedOnlyWhenMissing(self) -> None:
        self.assertEqual(
            _withDefaultConfig(["--input", "a.mp4"]),
            ["--input", "a.mp4", "--config", str(DEFAULT_CONFIG)],
        )
        explicit = ["--config", "custom.yaml"]
        self.assertEqual(_withDefaultConfig(explicit), explicit)
        self.assertEqual(_withDefaultConfig(["--config=custom.yaml"]), ["--config=custom.yaml"])

    def testUsageListsCommands(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = main(["--help"])
        self.assertEqual(code, 0)
        for command in ("track", "airsim360", "list-instances"):
            self.assertIn(command, stream.getvalue())

    def testUnknownCommandFails(self) -> None:
        self.assertEqual(main(["unknown"]), 2)

    def testAirSim360EntryWritesTimeArtifactWithoutChangingExitCode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            resultRoot = Path(directory) / "result"

            code = trackAirSim360Main(
                [
                    "--dataset-root",
                    str(Path(directory) / "missing-dataset"),
                    "--target-instance",
                    "1",
                    "--output",
                    str(resultRoot / "tracking.txt"),
                    "--config",
                    str(Path(directory) / "missing-config.yaml"),
                ]
            )

            self.assertEqual(code, EXIT_CONFIG)
            payload = json.loads((resultRoot / "time.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], "track360.time.v1")
            self.assertEqual(payload["scope"], "tracking_processing")
            self.assertGreaterEqual(payload["elapsedNanoseconds"], 0)


if __name__ == "__main__":
    unittest.main()
