"""Memory growth alone cannot exclude a fully qualified Pareto point."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cerebras_research.paths import digest, write_json
from tests.support.records import make_record
from tools import paired_frontier
from tools.paired_frontier import mark_frontier


def point(name, cycles, memory, track="same-contract"):
    return dict(candidate=name, cycles=cycles, memory_bytes_per_pe=memory, track_sha256=track)


class FrontierTests(unittest.TestCase):
    def test_fast_large_and_slow_small_both_survive(self):
        rows = mark_frontier(
            [point("small", 100, 10), point("fast-large", 60, 30), point("dominated", 110, 20)]
        )
        self.assertEqual([r["pareto_optimal"] for r in rows], [True, True, False])

    def test_equal_coordinates_and_incomparable_workloads(self):
        rows = mark_frontier(
            [point("a", 100, 10), point("tie", 100, 10), point("other", 1, 1, "other-contract")]
        )
        self.assertTrue(all(row["pareto_optimal"] for row in rows))

    def test_faster_and_smaller_dominates_both_tradeoffs(self):
        rows = mark_frontier(
            [point("small", 100, 10), point("fast-large", 60, 30), point("better", 50, 9)]
        )
        self.assertEqual([r["pareto_optimal"] for r in rows], [False, False, True])


class PairedEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        paths = [
            make_record(self.root, "base"),
            make_record(self.root, "candidate", "candidate", cycles=50, memory=400),
            make_record(self.root, "fresh-base", seed=8),
            make_record(self.root, "fresh-candidate", "candidate", seed=8, cycles=50, memory=400),
        ]
        for path in paths:
            record = json.loads(path.read_text())
            record["compile_command"] = ["cslc", "kernel.csl"]
            write_json(path, record)
        self.point = {
            arm: {"path": str(path.relative_to(self.root)), "sha256": digest(path)}
            for arm, path in zip(paired_frontier.ARMS, paths)
        }
        self.patch = patch.object(paired_frontier, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def report(self):
        return paired_frontier.build_report({"schema_version": 1, "points": [self.point]})

    def test_memory_policy_rejection_still_yields_qualified_frontier_point(self):
        report = self.report()
        baseline, candidate = report["points"]
        self.assertTrue(baseline["pareto_optimal"])
        self.assertTrue(candidate["pareto_optimal"])
        self.assertEqual(candidate["memory_ratio"], 4)
        self.assertFalse(candidate["default_policy_eligible"])
        self.assertEqual(candidate["qualification"], "matched_full_six_case_fresh_pair")

    def test_changed_receipt_is_rejected(self):
        self.point["fresh_candidate"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            self.report()

    def test_reused_seed_is_rejected(self):
        self.point["fresh_baseline"] = self.point["initial_baseline"]
        self.point["fresh_candidate"] = self.point["initial_candidate"]
        with self.assertRaisesRegex(ValueError, "seed"):
            self.report()
