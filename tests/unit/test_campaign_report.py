"""Source-bound campaign exports must reject stale plans and modified evidence."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cerebras_research.paths import digest, identity, write_json
from cerebras_research.scoring import compare
from tests.support.records import make_record
from tools.campaign_report import build_report


class CampaignReportTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        for name in ("tools.campaign_report.ROOT", "cerebras_research.ledger.ROOT"):
            mocked = patch(name, self.root)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.receipt = make_record(self.root, "baseline")
        record = json.loads(self.receipt.read_text())
        self.plan = dict(
            schema_version=1, seed=7, cells=[dict(profile="meshgemv.square64", candidates=[])]
        )
        self.state = dict(
            status="completed",
            plan_sha256=identity(self.plan),
            evaluator_sha256=record["provenance"]["evaluator_sha256"],
            runs=dict(base=dict(path="baseline/result.json", sha256=digest(self.receipt))),
            cells={
                "meshgemv.square64": dict(
                    status="no_confirmed_gain", baseline="baseline/result.json"
                )
            },
            decisions={},
        )
        self.plan_path = self.root / "plan.json"
        self.state_path = self.root / "state.json"
        self.save()

    def save(self):
        write_json(self.plan_path, self.plan)
        write_json(self.state_path, self.state)

    def test_terminal_report_replays_evidence_and_preserves_claim_boundary(self):
        report = build_report(self.plan_path, self.state_path)
        self.assertEqual(report["record_count"], 1)
        self.assertEqual(report["validated_case_count"], 6)
        self.assertEqual(
            report["records"][0]["provenance"]["evaluator_sha256"], self.state["evaluator_sha256"]
        )
        self.assertTrue(report["artifact_integrity_replayed"])
        self.assertFalse(report["numerical_arrays_recomputed"])
        self.assertFalse(report["qwen_tps_result"])

    def test_live_checkpoint_and_changed_plan_are_rejected(self):
        self.state["status"] = "running"
        self.save()
        with self.assertRaisesRegex(ValueError, "running"):
            build_report(self.plan_path, self.state_path)
        self.state["status"] = "completed"
        self.plan["seed"] = 8
        self.save()
        with self.assertRaisesRegex(ValueError, "plan"):
            build_report(self.plan_path, self.state_path)

    def test_changed_receipt_or_raw_artifact_is_rejected(self):
        data = json.loads(self.receipt.read_text())
        data["seed"] = 8
        write_json(self.receipt, data)
        with self.assertRaisesRegex(ValueError, "receipt"):
            build_report(self.plan_path, self.state_path)
        data["seed"] = 7
        write_json(self.receipt, data)
        self.state["runs"]["base"]["sha256"] = digest(self.receipt)
        self.save()
        (self.receipt.parent / "work/out/positive.npz").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Raw numerical artifact"):
            build_report(self.plan_path, self.state_path)

    def test_provisional_comparison_without_status_is_exported_and_replayed(self):
        candidate = make_record(self.root, "candidate", "candidate", cycles=80)
        report = compare(self.receipt, candidate)
        report.update(
            schema_version=2,
            record_type="comparison",
            inputs=dict(baseline="baseline/result.json", candidate="candidate/result.json"),
            budgets=dict(minimum_speedup=1.05, max_memory_growth=1.1),
        )
        write_json(self.root / "comparison.json", report)
        self.state["runs"]["candidate"] = dict(
            path="candidate/result.json", sha256=digest(candidate)
        )
        self.state["decisions"]["comparison"] = dict(report, receipt="comparison.json")
        self.save()
        result = build_report(self.plan_path, self.state_path)
        decision = next(r for r in result["records"] if r["kind"] == "comparison")
        self.assertEqual(decision["status"], "needs_fresh_confirmation")
        self.assertEqual(decision["speedup"], 1.25)

    def test_retried_attempts_are_preserved_beyond_the_latest_cache_entry(self):
        cancelled = make_record(self.root, "cancelled")
        data = json.loads(cancelled.read_text())
        data.update(status="cancelled", cases=[], metrics={})
        write_json(cancelled, data)
        self.state["attempts"] = [dict(path="cancelled/result.json", sha256=digest(cancelled))]
        self.save()
        report = build_report(self.plan_path, self.state_path)
        self.assertEqual(report["record_count"], 2)
        self.assertEqual(report["validated_case_count"], 6)
        self.assertEqual(report["executed_attempt_count"], 1)
        self.assertIn("cancelled", {r["status"] for r in report["records"]})
