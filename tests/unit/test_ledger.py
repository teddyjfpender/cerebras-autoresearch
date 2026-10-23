"""Journal history, evidence, and promotion must survive ordinary failure modes."""

import argparse
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cerebras_research import cli, ledger
from cerebras_research.catalog import get_profile
from cerebras_research.paths import write_json
from cerebras_research.provenance import capture
from cerebras_research.runner import execute
from cerebras_research.scoring import confirm
from tests.support.records import make_record


class LedgerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for name in ("cerebras_research.ledger.ROOT", "cerebras_research.cli.ROOT"):
            mocked = patch(name, self.root)
            mocked.start()
            self.addCleanup(mocked.stop)

    def change(self, path, **values):
        data = json.loads(path.read_text())
        data.update(values)
        write_json(path, data)
        return path

    def test_import_is_idempotent_and_changed_receipt_is_rejected(self):
        path = make_record(self.root, "base")
        self.assertEqual(ledger.append_receipts([path])["added"], 1)
        content = (self.root / "results/results.tsv").read_bytes()
        self.assertEqual(ledger.append_receipts([path])["added"], 0)
        self.assertEqual(content, (self.root / "results/results.tsv").read_bytes())
        self.change(path, error="changed after journal append")
        with self.assertRaisesRegex(ValueError, "already-recorded"):
            ledger.append_receipts([path])

    def test_control_characters_do_not_create_extra_tsv_rows_or_cells(self):
        path = make_record(self.root, "base")
        self.change(path, research={"hypothesis": "first\tcell\nsecond line\rfinal"})
        ledger.append_receipts([path])
        rows = ledger.read_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["hypothesis"], r"first\tcell\nsecond line\rfinal")
        self.assertEqual(ledger.verify(artifacts=True)["rows"], 1)

    def test_broken_chain_is_rejected(self):
        ledger.append_receipts([make_record(self.root, "base")])
        path = self.root / "results/results.tsv"
        path.write_text(path.read_text().replace("validated", "corrupted", 1))
        with self.assertRaisesRegex(ValueError, "digest"):
            ledger.read_rows()

    def test_git_prefix_blocks_truncation_even_after_board_regeneration(self):
        paths = [make_record(self.root, name) for name in ("a", "b", "c")]
        ledger.append_receipts(paths[:2])

        def git(*args):
            return subprocess.check_output(["git", *args], cwd=self.root, stderr=subprocess.DEVNULL)

        git("init", "-q")
        git("add", "results")
        git(
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "anchor",
        )
        base = git("rev-parse", "HEAD").decode().strip()
        ledger.append_receipts(paths[2:])
        self.assertTrue(ledger.verify(base_ref=base)["git_prefix_checked"])
        path = self.root / "results/results.tsv"
        path.write_text("\n".join(path.read_text().splitlines()[:2]) + "\n")
        write_json(self.root / "results/leaderboard.json", ledger.board(ledger.read_rows()))
        with self.assertRaisesRegex(ValueError, "history"):
            ledger.verify(base_ref=base)

    def test_screens_references_and_failures_never_enter_measured_board(self):
        paths = []
        for state in (
            "validated",
            "screen_passed",
            "reference_validated",
            "diagnostic_completed",
            "failed",
        ):
            path = make_record(self.root, state, cycles=1)
            self.change(path, status=state, screen_only=state == "screen_passed")
            paths.append(path)
        ledger.append_receipts(paths)
        view = ledger.board(ledger.read_rows())
        self.assertEqual(len(view["measured_best_by_approach"]), 1)
        self.assertEqual(view["confirmed_improvements"], [])

    def test_measured_memory_cycle_front_preserves_tradeoffs(self):
        paths = []
        for name, cycles, memory in (
            ("small", 100, 100),
            ("fast", 80, 120),
            ("dominated", 110, 110),
        ):
            path = make_record(self.root, name, "candidate", cycles=cycles, memory=memory)
            self.change(path, research={"approach": name})
            paths.append(path)
        ledger.append_receipts(paths)
        view = ledger.board(ledger.read_rows())
        self.assertEqual(len(view["measured_best_by_approach"]), 3)
        self.assertEqual(
            {r["approach"] for r in view["measured_cycle_memory_front"]}, {"small", "fast"}
        )

    def test_false_confirmation_label_is_rejected(self):
        first = (
            make_record(self.root, "base"),
            make_record(self.root, "candidate", "candidate", cycles=80),
        )
        fresh = (
            make_record(self.root, "fresh-base", seed=8),
            make_record(self.root, "fresh-candidate", "candidate", seed=8, cycles=80),
        )
        report = confirm(*first, *fresh)
        report.update(schema_version=2, record_type="confirmation")
        path = self.root / "decision.json"
        write_json(path, report)
        self.change(fresh[0], seed=7)
        with self.assertRaises(ValueError):
            ledger.append_receipts([path])

    def test_real_confirmation_and_cli_decisions_are_persisted(self):
        paths = (
            make_record(self.root, "base"),
            make_record(self.root, "candidate", "candidate", cycles=80),
            make_record(self.root, "fresh-base", seed=8),
            make_record(self.root, "fresh-candidate", "candidate", seed=8, cycles=80),
        )
        report = confirm(*paths)
        args = argparse.Namespace(minimum_speedup=1.05, max_memory_growth=1.1)
        result = cli.save_decision(
            report,
            "confirmation",
            dict(
                zip(
                    ("initial_baseline", "initial_candidate", "fresh_baseline", "fresh_candidate"),
                    paths,
                )
            ),
            args,
        )
        self.assertTrue((self.root / result["receipt"]).is_file())
        self.assertEqual(len(ledger.board(ledger.read_rows())["confirmed_improvements"]), 1)
        self.assertTrue(ledger.verify(artifacts=True)["artifacts_checked"])

    def test_corrupt_historical_evidence_is_recorded_as_invalid_not_promoted(self):
        path = make_record(self.root, "old")
        (path.parent / "work/out/positive.npz").write_bytes(b"corrupt")
        ledger.append_receipts([path], "historical_import")
        row = ledger.read_rows()[0]
        self.assertEqual(row["outcome"], "evidence_invalid")
        self.assertEqual(row["reported_outcome"], "validated")
        self.assertEqual(ledger.board([row])["measured_best_by_approach"], [])

    def prepare_failure_record(self):
        (self.root / "config").mkdir()
        write_json(self.root / "config/sdk-install.json", {})
        with (
            redirect_stdout(io.StringIO()),
            patch("cerebras_research.runner.ROOT", self.root),
            patch("cerebras_research.runner.shutil.which", return_value="stub"),
            patch("cerebras_research.runner.capture", return_value=({"evaluator_files": {}}, {})),
            patch(
                "cerebras_research.runner.prepare", side_effect=RuntimeError("compile setup failed")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "compile setup failed"):
                execute(get_profile("pipeline.k8-n8"))
        return ledger.read_rows()[0]

    def test_runner_records_prepare_failure_without_sdk_execution(self):
        row = self.prepare_failure_record()
        self.assertEqual(row["outcome"], "failed")
        self.assertIn("compile setup failed", row["note"])

    def test_broken_terminal_does_not_prevent_journaling_or_mask_original_failure(self):
        with patch("builtins.print", side_effect=BrokenPipeError("closed terminal")):
            row = self.prepare_failure_record()
        self.assertEqual(row["outcome"], "failed")
        self.assertIn("compile setup failed", row["note"])

    def test_source_capsule_tampering_blocks_promotion(self):
        path = make_record(self.root, "base")
        (path.parent / "evaluator-source/evaluator.py").write_text("# changed judge")
        with self.assertRaisesRegex(ValueError, "capsule"):
            ledger.append_receipts([path])

    def test_suite_confirmation_is_replayed_before_recording(self):
        profiles = ["meshgemv.square64", "meshgemv.ffn-up-scaled"]
        suites = []
        for name, variant, seed, cycles in (
            ("base", "baseline", 7, 100),
            ("candidate", "candidate", 7, 80),
            ("fresh-base", "baseline", 8, 100),
            ("fresh-candidate", "candidate", 8, 80),
        ):
            runs = {}
            for profile in profiles:
                run = make_record(
                    self.root, name + "/" + profile, variant, seed, cycles, profile_id=profile
                )
                runs[profile] = str(run.relative_to(self.root))
            path = self.root / (name + ".json")
            write_json(
                path,
                {"schema_version": 2, "status": "completed", "profiles": profiles, "runs": runs},
            )
            suites.append(path)
        args = argparse.Namespace(
            minimum_speedup=1.05, max_memory_growth=1.1, regression_limit=1.02
        )
        report = cli.confirm_suites(*suites, args)
        cli.save_decision(
            report,
            "suite_confirmation",
            dict(
                zip(
                    ("initial_baseline", "initial_candidate", "fresh_baseline", "fresh_candidate"),
                    suites,
                )
            ),
            args,
        )
        self.assertEqual(len(ledger.board(ledger.read_rows())["confirmed_suite_events"]), 1)
        self.assertEqual(ledger.verify(artifacts=True)["rows"], 1)

    def test_incomplete_receipt_is_reported_and_not_mislabelled_as_finished(self):
        path = make_record(self.root, "unfinished")
        self.change(path, status="running")
        result = ledger.append_receipts([path])
        self.assertEqual(result["added"], 0)
        self.assertEqual(result["skipped_receipts"], [str(path.relative_to(self.root))])

    def test_candidate_requires_falsification_and_evaluator_digest_covers_driver(self):
        driver = self.root / "benchmarks/drivers/oracle.py"
        driver.parent.mkdir(parents=True)
        driver.write_text("# first oracle")
        for name in ("sdk-run", "sdk-unpacked", "unpacked-sdk-bin/cslc"):
            wrapper = self.root / "tools" / name
            wrapper.parent.mkdir(parents=True, exist_ok=True)
            wrapper.write_text("# fixture SDK entrypoint")
        candidate = self.root / "candidate"
        candidate.mkdir()
        manifest = {
            "approach": "meshgemv.unroll",
            "hypothesis": "Remove map overhead",
            "falsification": "TODO: define",
        }
        write_json(candidate / "candidate.json", manifest)
        with (
            patch("cerebras_research.provenance.ROOT", self.root),
            patch("cerebras_research.provenance.subprocess.check_output", return_value=b"fixture"),
        ):
            with self.assertRaisesRegex(ValueError, "falsification"):
                capture(get_profile("meshgemv.square64"), "candidate", candidate)
            manifest["falsification"] = "Reject absent cycle gain or excess memory"
            write_json(candidate / "candidate.json", manifest)
            first, research = capture(get_profile("meshgemv.square64"), "candidate", candidate)
            driver.write_text("# changed oracle")
            second, _ = capture(get_profile("meshgemv.square64"), "candidate", candidate)
            wrapper.write_text("# changed SDK entrypoint")
            third, _ = capture(get_profile("meshgemv.square64"), "candidate", candidate)
        self.assertEqual(research["approach"], manifest["approach"])
        self.assertNotEqual(first["evaluator_sha256"], second["evaluator_sha256"])
        self.assertNotEqual(second["evaluator_sha256"], third["evaluator_sha256"])

    def test_best_confirmed_view_preserves_incumbent_and_separates_evaluator_tracks(self):
        paths = (
            make_record(self.root, "base"),
            make_record(self.root, "candidate", "candidate", cycles=80),
            make_record(self.root, "fresh-base", seed=8),
            make_record(self.root, "fresh-candidate", "candidate", seed=8, cycles=80),
        )
        report = confirm(*paths)
        args = argparse.Namespace(minimum_speedup=1.05, max_memory_growth=1.1)
        cli.save_decision(
            report,
            "confirmation",
            dict(
                zip(
                    ("initial_baseline", "initial_candidate", "fresh_baseline", "fresh_candidate"),
                    paths,
                )
            ),
            args,
        )
        incumbent = ledger.read_rows()[0]
        slower = dict(incumbent, primary_value="90", receipt="slower.json", event_id="slower")
        other = dict(
            incumbent,
            primary_value="1",
            evaluator_sha256="another-evaluator",
            receipt="other.json",
            event_id="other",
        )
        provisional = dict(
            incumbent,
            kind="run",
            outcome="validated",
            decision="",
            primary_value="0.5",
            receipt="provisional.json",
            event_id="provisional",
        )
        lucky = dict(
            incumbent, primary_value="60", speedup="1.15", receipt="lucky.json", event_id="lucky"
        )
        view = ledger.board([incumbent, slower, other, lucky, provisional])
        self.assertEqual(len(view["confirmed_improvements"]), 4)
        selected = {row["evaluator_sha256"]: row for row in view["best_confirmed_by_track"]}
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[incumbent["evaluator_sha256"]]["receipt"], incumbent["receipt"])
        self.assertEqual(selected["another-evaluator"]["receipt"], "other.json")


if __name__ == "__main__":
    unittest.main()
