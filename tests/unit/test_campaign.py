"""Batch research must reuse evidence safely and resume interrupted work."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cerebras_research.paths import write_json
from tests.support.records import make_record
from tools import run_campaign


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        prototype = make_record(self.root, "prototype")
        evaluator = json.loads(prototype.read_text())["provenance"]["evaluator_sha256"]
        for name in (
            "tools.run_campaign.ROOT",
            "cerebras_research.cli.ROOT",
            "cerebras_research.ledger.ROOT",
        ):
            context = patch(name, self.root)
            context.start()
            self.addCleanup(context.stop)
        context = patch("tools.run_campaign.evaluator_identity", return_value=evaluator)
        context.start()
        self.addCleanup(context.stop)
        self.plan = self.root / "plan.json"
        self.state = self.root / "state.json"
        write_json(
            self.plan,
            {
                "schema_version": 1,
                "seed": 7,
                "cells": [{"profile": "pipeline.k8-n8", "candidates": [{"variant": "fused"}]}],
            },
        )
        self.calls = []

    def execute(self, profile, variant, seed, screen, candidate, deadline, memory=100):
        self.calls.append((variant, seed, screen))
        path = make_record(
            self.root,
            "runs/v2/attempt-" + str(len(self.calls)),
            variant,
            seed,
            cycles=100 if variant == "baseline" else 80,
            memory=100 if variant == "baseline" else memory,
            profile_id=profile["id"],
        )
        if screen:
            data = json.loads(path.read_text())
            data.update(status="screen_passed", screen_only=True)
            write_json(path, data)
        return path

    def run_plan(self, execute=None, retry_failed=False):
        with (
            redirect_stdout(io.StringIO()),
            patch("tools.run_campaign.execute", side_effect=execute or self.execute),
        ):
            return run_campaign.campaign(self.plan, self.state, 600, retry_failed)

    def test_shared_baseline_confirms_and_completed_resume_does_no_work(self):
        result = self.run_plan()
        self.assertEqual(result["cells"]["pipeline.k8-n8"]["status"], "confirmed_improvement")
        self.assertEqual(len(self.calls), 5)
        self.run_plan()
        self.assertEqual(len(self.calls), 5)

    def test_status_reads_checkpoint_without_running_sdk_work(self):
        self.run_plan()
        with patch("tools.run_campaign.execute") as execute:
            summary = run_campaign.status(self.plan, self.state)
        execute.assert_not_called()
        self.assertEqual(summary["completed_runs"], 5)
        self.assertEqual(summary["cells"][0]["status"], "confirmed_improvement")

    def test_retry_failed_does_not_reprocess_a_confirmed_cell(self):
        self.run_plan()
        with patch("tools.run_campaign.load_validated") as load:
            self.run_plan(retry_failed=True)
        load.assert_not_called()
        self.assertEqual(len(self.calls), 5)

    def test_fast_screen_above_memory_budget_is_not_promoted(self):
        result = self.run_plan(lambda *args: self.execute(*args, memory=120))
        self.assertEqual(result["cells"]["pipeline.k8-n8"]["status"], "no_confirmed_gain")
        self.assertEqual(len(self.calls), 2)

    def test_failed_screen_retry_preserves_the_original_attempt(self):
        def fail_screen(*args):
            path = self.execute(*args)
            if args[3]:
                data = json.loads(path.read_text())
                data["status"] = "failed"
                write_json(path, data)
                raise RuntimeError("simulated SDK failure with a saved receipt")
            return path

        first = self.run_plan(fail_screen)
        self.assertEqual(first["attempts"][-1]["outcome"], "failed")
        result = self.run_plan(retry_failed=True)
        self.assertEqual(result["cells"]["pipeline.k8-n8"]["status"], "confirmed_improvement")
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(len(result["attempts"]), 6)
        self.assertEqual(result["attempts"][1]["outcome"], "failed")

    def test_cached_receipt_edit_is_rejected_even_after_campaign_completion(self):
        result = self.run_plan()
        pointer = next(iter(result["runs"].values()))
        path = self.root / pointer["path"]
        data = json.loads(path.read_text())
        data["note"] = "edited after measurement"
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "Cached receipt"):
            self.run_plan()

    def test_evaluator_change_cannot_reuse_campaign(self):
        self.run_plan()
        with patch("tools.run_campaign.evaluator_identity", return_value="different"):
            with self.assertRaisesRegex(ValueError, "evaluator"):
                self.run_plan()

    def test_interrupted_campaign_reuses_completed_runs_and_clears_active_error(self):
        def interrupt(*args):
            if len(self.calls) == 2:
                raise KeyboardInterrupt("interrupted fixture")
            return self.execute(*args)

        with self.assertRaises(KeyboardInterrupt):
            self.run_plan(interrupt)
        self.assertEqual(json.loads(self.state.read_text())["status"], "cancelled")
        result = self.run_plan()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.calls), 5)
        self.assertNotIn("error", result)
        self.assertEqual(len(result["errors"]), 1)

    def test_cancelled_receipt_is_checkpointed_and_automatically_retried(self):
        def cancel_screen(*args):
            path = self.execute(*args)
            if args[3]:
                data = json.loads(path.read_text())
                data["status"] = "cancelled"
                write_json(path, data)
                raise KeyboardInterrupt("cancelled fixture with a receipt")
            return path

        with self.assertRaises(KeyboardInterrupt):
            self.run_plan(cancel_screen)
        interrupted = json.loads(self.state.read_text())
        self.assertEqual(interrupted["attempts"][-1]["outcome"], "cancelled")
        result = self.run_plan()
        self.assertEqual(result["cells"]["pipeline.k8-n8"]["status"], "confirmed_improvement")
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(len(result["attempts"]), 6)
        self.assertEqual(result["attempts"][1]["outcome"], "cancelled")
