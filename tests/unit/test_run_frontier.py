"""The full frontier loop must preserve costly tradeoffs and resume evidence."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cerebras_research.paths import write_json
from tests.support.records import make_record
from tools import run_frontier


class FrontierCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        candidate = self.root / "experiments/candidates/large"
        candidate.mkdir(parents=True)
        (candidate / "kernel.csl").write_text("// synthetic candidate fixture")
        write_json(candidate / "candidate.json", {"approach": "synthetic-fixture"})
        prototype = make_record(self.root, "prototype")
        evaluator = json.loads(prototype.read_text())["provenance"]["evaluator_sha256"]
        for target, value in (
            ("tools.run_frontier.ROOT", self.root),
            ("cerebras_research.cli.ROOT", self.root),
            ("cerebras_research.ledger.ROOT", self.root),
        ):
            context = patch(target, value)
            context.start()
            self.addCleanup(context.stop)
        context = patch("tools.run_frontier.evaluator_identity", return_value=evaluator)
        context.start()
        self.addCleanup(context.stop)
        self.plan = self.root / "plan.json"
        self.state = self.root / "state.json"
        write_json(
            self.plan,
            dict(
                schema_version=1,
                kind="full_fresh_frontier_pilot",
                seed=7,
                candidates=["large"],
                profiles=[{"profile": "meshgemv.square64"}],
            ),
        )
        self.calls = []

    def execute(self, profile, variant, seed, screen, candidate, deadline):
        self.calls.append((variant, seed, screen))
        path = make_record(
            self.root,
            "runs/v2/attempt-" + str(len(self.calls)),
            variant,
            seed,
            cycles=100 if variant == "baseline" else 50,
            memory=100 if variant == "baseline" else 400,
            profile_id=profile["id"],
        )
        data = json.loads(path.read_text())
        data["compile_command"] = ["cslc", "kernel.csl"]
        write_json(path, data)
        return path

    def run_plan(self):
        with patch("tools.run_frontier.execute", side_effect=self.execute):
            return run_frontier.campaign(self.plan, self.state, 600)

    def test_fourfold_memory_gets_full_fresh_runs_and_separate_policy_rejection(self):
        result = self.run_plan()
        self.assertEqual(len(self.calls), 4)
        self.assertTrue(all(not call[2] for call in self.calls))
        self.assertEqual(result["status"], "completed")
        cell = result["cells"]["meshgemv.square64/large"]
        self.assertEqual(cell["status"], "fresh_paired_qualified")
        self.assertEqual(cell["default_policy_status"], "rejected")
        self.assertEqual(len(result["points"]), 1)
        self.run_plan()
        self.assertEqual(len(self.calls), 4)

    def test_cached_receipt_edit_cannot_be_reused(self):
        result = self.run_plan()
        path = self.root / result["points"][0]["fresh_candidate"]["path"]
        data = json.loads(path.read_text())
        data["note"] = "changed"
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "Cached receipt"):
            self.run_plan()

    def test_changed_candidate_cannot_reuse_completed_state(self):
        self.run_plan()
        (self.root / "experiments/candidates/large/kernel.csl").write_text("// edited")
        with self.assertRaisesRegex(ValueError, "candidate changed"):
            self.run_plan()

    def test_interruption_preserves_full_first_arm_for_resume(self):
        def interrupt(*args):
            if len(self.calls) == 3:
                raise KeyboardInterrupt("before fresh run starts")
            return self.execute(*args)

        with patch("tools.run_frontier.execute", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_frontier.campaign(self.plan, self.state, 600)
        stopped = json.loads(self.state.read_text())
        self.assertEqual(stopped["status"], "cancelled")
        self.assertEqual(len(stopped["points"]), 0)
        result = self.run_plan()
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["points"]), 1)
