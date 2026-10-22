"""Resource guards must survive disappearing simulator temporary files."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cerebras_research.process import run_logged


class ProcessTests(unittest.TestCase):
    def test_disappearing_log_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch(
                "cerebras_research.process.os.walk", return_value=[(folder, [], ["gone.log"])]
            ):
                result = run_logged(
                    [sys.executable, "-c", "print('clean exit')"], root, root / "run.log"
                )
            self.assertEqual(result["exit_code"], 0)

    def test_timeout_preserves_log(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "Step exceeded"):
                run_logged(
                    [
                        sys.executable,
                        "-c",
                        "import time; print('started', flush=True); time.sleep(10)",
                    ],
                    root,
                    root / "run.log",
                    timeout=0.05,
                )
            self.assertIn("started", (root / "run.log").read_text())
