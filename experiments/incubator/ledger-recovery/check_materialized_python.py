"""Prepared runs must execute the helper source that their fingerprints describe."""

import os
import py_compile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cerebras_research.adapters import materialize


class MaterializedPythonTests(unittest.TestCase):
    def test_stale_bytecode_cannot_mask_fingerprinted_helper_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            origin = root / "source"
            origin.mkdir()
            helper = origin / "helper.py"
            helper.write_text("VALUE = 'old'\n")
            timestamp = helper.stat()
            py_compile.compile(str(helper), doraise=True)
            helper.write_text("VALUE = 'new'\n")
            os.utime(helper, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
            (origin / "kernel.csl").write_text("// source")
            work = root / "work"
            with patch("cerebras_research.adapters.baseline_source", return_value=(origin, {})):
                materialize({}, work)
            result = subprocess.check_output(
                [sys.executable, "-c", "import helper; print(helper.VALUE)"], cwd=work, text=True
            )
            self.assertEqual(result.strip(), "new")
