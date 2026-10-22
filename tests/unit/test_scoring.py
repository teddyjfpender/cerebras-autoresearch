"""Promotion gates must reject tempting but invalid improvements."""

import json
import tempfile
import unittest
from pathlib import Path

from cerebras_research.paths import write_json
from cerebras_research.scoring import aggregate, compare, confirm, load_validated
from tests.support.records import make_record


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def run_record(self, *args, **kwargs):
        return make_record(self.folder, *args, **kwargs)

    def mutate(self, path, change):
        data = json.loads(path.read_text())
        change(data)
        write_json(path, data)

    def test_confirm_requires_fresh_matched_pair(self):
        first = self.run_record("base"), self.run_record("candidate", "candidate", cycles=80)
        fresh = (
            self.run_record("fresh-base", seed=8),
            self.run_record("fresh-candidate", "candidate", seed=8, cycles=80),
        )
        self.assertEqual(confirm(*first, *fresh)["status"], "confirmed_improvement")

    def test_same_seed_cannot_confirm(self):
        first = self.run_record("base"), self.run_record("candidate", "candidate", cycles=80)
        with self.assertRaisesRegex(ValueError, "seed"):
            confirm(
                *first,
                self.run_record("fresh-base"),
                self.run_record("fresh-candidate", "candidate", cycles=80),
            )

    def test_fresh_confirmation_cannot_change_compiler_arguments(self):
        first = self.run_record("base"), self.run_record("candidate", "candidate", cycles=80)
        fresh = (
            self.run_record("fresh-base", seed=8),
            self.run_record("fresh-candidate", "candidate", seed=8, cycles=80),
        )
        self.mutate(fresh[1], lambda data: data["compile_command"].append("--changed-flag"))
        with self.assertRaisesRegex(ValueError, "compile_command"):
            confirm(*first, *fresh)

    def test_fast_but_larger_memory_is_rejected(self):
        result = compare(
            self.run_record("base"),
            self.run_record("candidate", "candidate", cycles=50, memory=120),
        )
        self.assertFalse(result["eligible"])

    def test_screen_reference_and_diagnostics_cannot_qualify(self):
        for status in (
            "screen_passed",
            "reference_validated",
            "diagnostic_completed",
            "failed",
            "running",
        ):
            with self.subTest(status=status):
                path = self.run_record(status)
                self.mutate(path, lambda d: d.update(status=status))
                with self.assertRaises(ValueError):
                    load_validated(path)

    def test_changed_precision_or_missing_case_is_rejected(self):
        path = self.run_record("base")
        self.mutate(path, lambda d: d["profile"].update(precision="fp32"))
        with self.assertRaisesRegex(ValueError, "Profile"):
            load_validated(path)
        path = self.run_record("missing")
        self.mutate(path, lambda d: d["cases"].pop())
        with self.assertRaisesRegex(ValueError, "cases"):
            load_validated(path)

    def test_false_measurement_summary_is_rejected(self):
        for field in ("max_local_cycles", "max_program_object_bytes_per_pe"):
            with self.subTest(field=field):
                path = self.run_record(field)
                self.mutate(path, lambda d: d["metrics"].update({field: 1}))
                with self.assertRaises(ValueError):
                    load_validated(path)

    def test_artifact_corruption_and_source_changes_are_rejected(self):
        for file in ("work/out/positive.npz", "work/kernel.csl", "work/driver.py"):
            with self.subTest(file=file):
                path = self.run_record(str(len(list(self.folder.iterdir()))))
                (path.parent / file).write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    load_validated(path)

    def test_forged_case_receipt_is_rejected(self):
        path = self.run_record("base")
        self.mutate(path, lambda d: d["cases"][0].update(max_fraction_of_error_bound=0.01))
        with self.assertRaisesRegex(ValueError, "Case receipt"):
            load_validated(path)

    def test_suite_average_cannot_hide_regression_or_missing_coverage(self):
        cells = [
            {"profile": "a", "speedup": 1.3, "memory_ratio": 1, "max_memory_growth": 1.1},
            {"profile": "b", "speedup": 0.95, "memory_ratio": 1, "max_memory_growth": 1.1},
        ]
        result = aggregate(cells, ["a", "b"])
        self.assertGreater(result["geometric_mean_speedup"], 1.05)
        self.assertFalse(result["eligible"])
        with self.assertRaisesRegex(ValueError, "coverage"):
            aggregate(cells[:1], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
