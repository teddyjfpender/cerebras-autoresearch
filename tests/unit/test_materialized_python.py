"""Prepared runs must execute the helper source that their fingerprints describe."""

import os
import py_compile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cerebras_research.adapters import candidate_init, materialize


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

    def test_candidate_initialization_carries_the_pinned_upstream_license(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkout = root / ".deps/waferllm"
            origin = checkout / "MeshGEMV"
            (origin / "src").mkdir(parents=True)
            (origin / "src/layout.csl").write_text("// pinned code")
            (checkout / "LICENSE").write_text("Upstream terms\n")
            profile = dict(adapter="meshgemv", id="fixture", variants=["baseline", "candidate"])
            with (
                patch("cerebras_research.adapters.ROOT", root),
                patch(
                    "cerebras_research.adapters.baseline_source",
                    return_value=(origin, dict(commit="pinned")),
                ),
            ):
                candidate = candidate_init(profile, "license-probe")
            self.assertEqual(
                (candidate / "LICENSE.upstream").read_bytes(), (checkout / "LICENSE").read_bytes()
            )
            self.assertEqual((candidate / "src/layout.csl").read_text(), "// pinned code")
