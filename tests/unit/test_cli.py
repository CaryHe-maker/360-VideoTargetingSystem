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
        for command in ("track", "download", "benchmark", "airsim360", "list-instances"):
            self.assertIn(command, stream.getvalue())

    def testUnknownCommandFails(self) -> None:
        self.assertEqual(main(["unknown"]), 2)

    def testTrackChoosesItsConfigurationByPresetAndPrecision(self) -> None:
        from track360.runtime.track_video import buildParser

        args = buildParser().parse_args(
            ["--input", "a.mp4", "--init-box", "1,2,3,4", "--output", "out.txt"]
        )
        self.assertEqual((args.preset, args.precision, args.config), ("default", None, None))
        args = buildParser().parse_args(
            [
                "--input",
                "a.mp4",
                "--init-bfov=-10,5,20,30",
                "--demo",
                "d.mp4",
                "--preset",
                "loss_handling",
                "--precision",
                "tf32",
            ]
        )
        self.assertEqual((args.preset, args.precision), ("loss_handling", "tf32"))
        self.assertEqual(args.init_bfov, "-10,5,20,30")

    def testBenchmarkIsTheToolOfTheCheckout(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream), self.assertRaises(SystemExit) as raised:
            main(["benchmark", "--help"])
        self.assertEqual(raised.exception.code, 0)
        for command in ("run", "eval", "archive", "compare"):
            self.assertIn(command, stream.getvalue())

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
