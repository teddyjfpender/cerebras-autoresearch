"""Failed or cancelled SDK steps retain their attempted command and elapsed cost."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cerebras_research.catalog import get_profile
from cerebras_research.ledger import receipt_row
from cerebras_research.paths import digest, write_json
from cerebras_research.runner import execute


class FailedStepTests(unittest.TestCase):
    def test_failure_and_cancellation_keep_step_metadata(self):
        for error in (RuntimeError("SDK step failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve()
                write_json(root / "config/sdk-install.json", {})

                def prepared(profile, work, variant, candidate):
                    work.mkdir()
                    (work / "kernel.csl").write_text("// source")
                    return ["cslc", "kernel.csl"], {}, "baseline"

                def failed_step(args, work, log, timeout):
                    binary = work / "out/bin/partial.elf"
                    binary.parent.mkdir(parents=True)
                    binary.write_bytes(b"synthetic partial compiler output")
                    raise error

                with (
                    redirect_stdout(io.StringIO()),
                    patch("cerebras_research.ledger.ROOT", root),
                    patch("cerebras_research.runner.ROOT", root),
                    patch("cerebras_research.runner.shutil.which", return_value="stub"),
                    patch(
                        "cerebras_research.runner.capture",
                        return_value=({"evaluator_files": {}}, {}),
                    ),
                    patch("cerebras_research.runner.prepare", side_effect=prepared),
                    patch("cerebras_research.runner.run_logged", side_effect=failed_step),
                ):
                    with self.assertRaises(type(error)):
                        execute(get_profile("pipeline.k8-n8"))
                receipt = next((root / "runs/v2").glob("*/result.json"))
                data = json.loads(receipt.read_text())
                status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
                self.assertEqual(data["status"], status)
                self.assertEqual(data["steps"][0]["status"], status)
                self.assertEqual(data["steps"][0]["command"], ["cslc", "kernel.csl"])
                self.assertEqual(data["steps"][0]["log"], "step-0.log")
                self.assertGreaterEqual(data["steps"][0]["wall_seconds"], 0)
                self.assertEqual(
                    data["metrics"]["experiment_wall_seconds"], data["steps"][0]["wall_seconds"]
                )
                self.assertEqual(data["cases"], [])
                binary = receipt.parent / "work/out/bin/partial.elf"
                self.assertEqual(data["artifacts"]["work/out/bin/partial.elf"], digest(binary))
                binary.write_bytes(b"changed executable artifact")
                with (
                    patch("cerebras_research.ledger.ROOT", root),
                    self.assertRaisesRegex(ValueError, "artifact changed"),
                ):
                    receipt_row(receipt, "live")
