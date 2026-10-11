from __future__ import annotations

import hashlib
import tempfile
import threading
import unittest
from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from track360 import hub
from track360.core.errors import ModelError

PAYLOAD = bytes(range(256)) * 5000  # 1.28 MB: more than one block


class _RangeHandler(SimpleHTTPRequestHandler):
    """Serves files with support for ``Range: bytes=N-``, as a resuming client needs."""

    def log_message(self, *args: object) -> None:
        del args

    def do_GET(self) -> None:  # noqa: N802 - the name the base class calls
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        start = 0
        header = self.headers.get("Range")
        if header:
            start = int(header.split("=")[1].split("-")[0])
            if start >= len(data):
                self.send_error(416)
                return
            self.send_response(206)
        else:
            self.send_response(200)
        kind = "text/html" if path.suffix == ".html" else "application/octet-stream"
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        self.wfile.write(data[start:])


class DownloadWeightsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        (self.root / "served").mkdir()
        (self.root / "served" / "weights.bin").write_bytes(PAYLOAD)
        (self.root / "served" / "page.html").write_text("<html>sign in</html>")
        handler = partial(_RangeHandler, directory=str(self.root / "served"))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        entry = replace(
            hub.WEIGHTS["artrackv2_b_256"],
            sha256=hashlib.sha256(PAYLOAD).hexdigest(),
            sizeBytes=len(PAYLOAD),
            url=f"{self.base}/weights.bin",
        )
        patcher = patch.dict(hub.WEIGHTS, {"artrackv2_b_256": entry})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.messages: list[str] = []

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._directory.cleanup()

    def _download(self, target: Path, **options: object) -> Path:
        return hub.downloadWeights(target, report=self.messages.append, **options)

    def testDownloadsVerifiesAndKeepsAnExistingFile(self) -> None:
        target = self.root / "models" / "weights.pth.tar"
        self.assertEqual(self._download(target), target)
        self.assertEqual(target.read_bytes(), PAYLOAD)
        self.assertFalse(target.with_name(target.name + ".part").exists())
        self.assertTrue(hub.checkWeights(target))
        # A second call finds the verified file and fetches nothing.
        self.server.shutdown()
        self.assertEqual(self._download(target), target)
        self.assertIn("already there", self.messages[-1])

    def testAnInterruptedDownloadIsContinued(self) -> None:
        target = self.root / "weights.pth.tar"
        target.with_name(target.name + ".part").write_bytes(PAYLOAD[:300_000])
        self._download(target)
        self.assertEqual(target.read_bytes(), PAYLOAD)

    def testAWrongFileIsNotAccepted(self) -> None:
        (self.root / "served" / "other.bin").write_bytes(b"not the weights")
        target = self.root / "weights.pth.tar"
        with self.assertRaisesRegex(ModelError, "SHA-256"):
            self._download(target, url=f"{self.base}/other.bin")
        self.assertFalse(target.exists())
        # Unless the caller says the hash does not matter.
        target.with_name(target.name + ".part").unlink()
        self._download(target, url=f"{self.base}/other.bin", verify=False)
        self.assertEqual(target.read_bytes(), b"not the weights")
        with self.assertRaisesRegex(ModelError, "exists but is not"):
            self._download(target)

    def testAWebPageInsteadOfTheFileIsAnError(self) -> None:
        with self.assertRaisesRegex(ModelError, "web page"):
            self._download(self.root / "weights.pth.tar", url=f"{self.base}/page.html")

    def testAMissingSourceIsAnError(self) -> None:
        with self.assertRaisesRegex(ModelError, "HTTP 404"):
            self._download(self.root / "weights.pth.tar", url=f"{self.base}/absent.bin")

    def testTheCommandChecksAFile(self) -> None:
        target = self.root / "weights.pth.tar"
        target.write_bytes(PAYLOAD)
        self.assertEqual(hub.main(["--output", str(target), "--check"]), 0)
        target.write_bytes(b"x")
        self.assertEqual(hub.main(["--output", str(target), "--check"]), 1)
        self.assertEqual(hub.main(["--output", str(self.root / "absent"), "--check"]), 1)


if __name__ == "__main__":
    unittest.main()
