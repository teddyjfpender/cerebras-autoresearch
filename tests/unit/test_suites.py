import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from cerebras_research.cli import confirm_suites, main
from cerebras_research.paths import digest, tree_digest, write_json
from tests.support.records import make_record


class SuiteConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profiles = ["meshgemv.square64", "meshgemv.ffn-up-scaled"]
        self.args = argparse.Namespace(
            minimum_speedup=1.05, max_memory_growth=1.1, regression_limit=1.02
        )

    def suite(self, name, variant="baseline", seed=7, cycles=(100, 100)):
        runs = {}
        for profile, value in zip(self.profiles, cycles):
            path = make_record(
                self.root, name + "/" + profile, variant, seed, value, profile_id=profile
            )
            runs[profile] = str(path.relative_to(self.root))
        path = self.root / (name + ".json")
        write_json(path, {"status": "completed", "profiles": self.profiles, "runs": runs})
        return path

    def pairs(self, fresh_seed=8):
        return (
            self.suite("base"),
            self.suite("candidate", "candidate", cycles=(90, 98)),
            self.suite("fresh-base", seed=fresh_seed),
            self.suite("fresh-candidate", "candidate", fresh_seed, (91, 98)),
        )

    def test_fresh_suite_can_confirm_with_gain_below_five_percent_in_one_cell(self):
        with patch("cerebras_research.cli.ROOT", self.root):
            result = confirm_suites(*self.pairs(), self.args)
        self.assertEqual(result["status"], "confirmed_suite_improvement")
        self.assertLess(result["cell_evidence"][self.profiles[1]]["initial_speedup"], 1.05)

    def test_reused_seed_is_rejected_for_suite(self):
        with patch("cerebras_research.cli.ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "seed"):
                confirm_suites(*self.pairs(fresh_seed=7), self.args)

    def test_missing_cell_is_rejected_even_when_suite_says_completed(self):
        paths = self.pairs()
        data = json.loads(paths[1].read_text())
        data["runs"].pop(self.profiles[0])
        write_json(paths[1], data)
        with patch("cerebras_research.cli.ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "coverage"):
                confirm_suites(*paths, self.args)

    def test_different_candidate_source_cannot_confirm(self):
        paths = self.pairs()
        suite = json.loads(paths[3].read_text())
        path = self.root / suite["runs"][self.profiles[0]]
        source = path.parent / "work/kernel.csl"
        source.write_text("// changed after initial pair")
        data = json.loads(path.read_text())
        data["source_sha256"] = tree_digest(path.parent / "work")
        data["artifacts"]["work/kernel.csl"] = digest(source)
        write_json(path, data)
        with patch("cerebras_research.cli.ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                confirm_suites(*paths, self.args)

    def test_nonfinite_budget_is_rejected_before_execution(self):
        with redirect_stderr(io.StringIO()), patch("cerebras_research.cli.execute") as execute:
            result = main(["run", "meshgemv.square64", "--max-seconds", "nan"])
            self.assertEqual(result, 1)
            execute.assert_not_called()
