"""A running checkpoint must not masquerade as proof that its worker is alive."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.run_campaign import status, worker_identity, worker_liveness


class CampaignLivenessTests(unittest.TestCase):
    def make_proc(self, folder, state="S", ticks=123):
        root = Path(folder)
        boot = root / "sys/kernel/random/boot_id"
        boot.parent.mkdir(parents=True, exist_ok=True)
        boot.write_text("test-boot\n")
        process = root / "42"
        process.mkdir(exist_ok=True)
        # Fields 3..22: comm can contain spaces and a closing parenthesis.
        fields = [state] + ["0"] * 18 + [str(ticks)]
        (process / "stat").write_text("42 (research ) worker) " + " ".join(fields))
        return root

    def test_live_exited_reused_and_missing_worker_are_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            proc = self.make_proc(folder)
            worker = worker_identity(42, proc)
            self.assertEqual(worker_liveness(worker, proc)["state"], "alive")
            self.make_proc(folder, state="Z")
            self.assertEqual(worker_liveness(worker, proc)["state"], "not_running")
            self.make_proc(folder, ticks=124)
            result = worker_liveness(worker, proc)
            self.assertEqual(result["state"], "not_running")
            self.assertIn("reused", result["reason"])
            (proc / "42/stat").unlink()
            self.assertEqual(worker_liveness(worker, proc)["state"], "not_running")

    def test_other_environment_user_and_legacy_checkpoint_are_unverified(self):
        with tempfile.TemporaryDirectory() as folder:
            proc = self.make_proc(folder)
            worker = worker_identity(42, proc)
            self.assertEqual(worker_liveness(None, proc)["state"], "unknown")
            self.assertEqual(worker_liveness("invalid identity", proc)["state"], "unknown")
            self.assertEqual(worker_liveness(worker, proc / "missing")["state"], "unknown")
            self.assertEqual(
                worker_liveness(dict(worker, uid=os.getuid() + 1), proc)["state"], "unknown"
            )
            (proc / "sys/kernel/random/boot_id").write_text("another-boot")
            self.assertEqual(worker_liveness(worker, proc)["state"], "unknown")

    def test_stale_running_checkpoint_is_reported_without_mutating_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            proc = self.make_proc(root / "proc")
            worker = worker_identity(42, proc)
            (proc / "42/stat").unlink()
            plan = root / "plan.json"
            state = root / "state.json"
            plan.write_text(json.dumps({"cells": []}))
            state.write_text(
                json.dumps(
                    {"status": "running", "worker": worker, "runs": {}, "attempts": [], "cells": {}}
                )
            )
            before = state.read_bytes()
            with patch(
                "tools.run_campaign.worker_liveness", side_effect=lambda w: worker_liveness(w, proc)
            ):
                summary = status(plan, state)
            self.assertEqual(summary["status"], "running")
            self.assertEqual(summary["worker_liveness"]["state"], "not_running")
            self.assertEqual(state.read_bytes(), before)

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "requires Linux process observations")
    def test_actual_child_identity_reports_exit(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            worker = worker_identity(child.pid)
            self.assertIsNotNone(worker)
            self.assertEqual(worker_liveness(worker)["state"], "alive")
            child.terminate()
            child.wait(timeout=5)
            self.assertEqual(worker_liveness(worker)["state"], "not_running")
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
