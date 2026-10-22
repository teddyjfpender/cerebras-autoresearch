"""Regression gates for the staged evaluator changes; launches no SDK job."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cerebras_research import ledger, runner
from cerebras_research.catalog import get_profile
from cerebras_research.paths import write_json
from tests.support.records import make_record


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        mocked = patch("cerebras_research.ledger.ROOT", self.root)
        mocked.start()
        self.addCleanup(mocked.stop)

    def test_unchanged_recovery_skips_old_artifact_audit(self):
        path = make_record(self.root, "base")
        ledger.append_receipts([path])
        before = (self.root / "results/results.tsv").read_bytes()
        with patch("cerebras_research.ledger.receipt_row", side_effect=AssertionError("old audit")):
            self.assertEqual(ledger.append_receipts([path])["added"], 0)
        self.assertEqual(before, (self.root / "results/results.tsv").read_bytes())

    def test_incremental_skip_does_not_hide_corruption_from_explicit_audit(self):
        path = make_record(self.root, "base")
        ledger.append_receipts([path])
        (path.parent / "work/out/positive.npz").write_bytes(b"corrupt")
        self.assertEqual(ledger.append_receipts([path])["added"], 0)
        with self.assertRaises(ValueError):
            ledger.verify(artifacts=True)

    def test_changed_receipt_is_rejected_before_recovery(self):
        path = make_record(self.root, "base")
        ledger.append_receipts([path])
        data = json.loads(path.read_text())
        data["error"] = "changed"
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "already-recorded"):
            ledger.append_receipts([path])

    def test_new_receipt_still_passes_full_audit(self):
        old, new = (make_record(self.root, name) for name in ("old", "new"))
        ledger.append_receipts([old])
        with patch("cerebras_research.ledger.receipt_row", wraps=ledger.receipt_row) as replay:
            self.assertEqual(ledger.append_receipts([old, new])["added"], 1)
        self.assertEqual(replay.call_count, 1)
        self.assertEqual(replay.call_args.args[0], new)

    def test_runner_captures_auxiliary_python_and_rejects_mutation(self):
        (self.root / "config").mkdir()
        write_json(self.root / "config/sdk-install.json", {})

        def prepared(profile, work, variant, candidate):
            (work / "helpers").mkdir(parents=True)
            (work / "kernel.csl").write_text("// source")
            (work / "helpers/host_common.py").write_text("# auxiliary baseline helper")
            return ["cslc", "kernel.csl"], {}, "baseline"

        with (
            redirect_stdout(io.StringIO()),
            patch("cerebras_research.runner.ROOT", self.root),
            patch("cerebras_research.runner.shutil.which", return_value="stub"),
            patch("cerebras_research.runner.capture", return_value=({"evaluator_files": {}}, {})),
            patch("cerebras_research.runner.prepare", side_effect=prepared),
            patch(
                "cerebras_research.runner.run_logged",
                side_effect=RuntimeError("stop after capture"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after capture"):
                runner.execute(get_profile("pipeline.k8-n8"))
        receipt = next((self.root / "runs/v2").glob("*/result.json"))
        data = json.loads(receipt.read_text())
        self.assertEqual(set(data["harness_files"]), {"helpers/host_common.py"})
        (receipt.parent / "work/helpers/host_common.py").write_text("# changed after capture")
        with self.assertRaisesRegex(ValueError, "harness"):
            ledger.audit_run(receipt, data)


if __name__ == "__main__":
    unittest.main()
