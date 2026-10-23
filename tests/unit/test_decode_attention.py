"""SDK-free gates for distributed attention control and source preparation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cerebras_research import ledger
from cerebras_research.adapters import baseline_source, prepare, verify_timing_contract
from cerebras_research.catalog import get_profile, get_suite
from cerebras_research.paths import ROOT
from cerebras_research.scoring import compare, confirm
from tests.support.records import make_record


class DecodeAttentionTests(unittest.TestCase):
    def test_real_head_dimensions_and_distributed_contexts(self):
        suite = get_suite("decode-attention")
        self.assertEqual(len(suite), 5)
        self.assertEqual({p["dimensions"]["D"] for p in suite}, {64, 128})
        self.assertIn(128, {p["dimensions"]["P"] for p in suite})
        for profile in suite:
            self.assertEqual(profile["metric"], "coordinator_cycles")
            self.assertEqual(profile["pe_grid"], [profile["dimensions"]["P"], 1])
            self.assertEqual(profile["oracle_contract"], "sharded_gqa_decode_fp32_v1")

    def test_all_candidates_prepare_on_every_shape(self):
        for profile in get_suite("decode-attention"):
            for name in (
                None,
                "attention-vector-exp",
                "attention-polynomial-exp",
                "attention-dsr",
                "attention-dsr-normalize",
            ):
                with (
                    self.subTest(profile=profile["id"], candidate=name),
                    tempfile.TemporaryDirectory() as folder,
                ):
                    work = Path(folder) / "work"
                    command, _, _ = prepare(
                        profile,
                        work,
                        "candidate" if name else "baseline",
                        ROOT / "experiments/candidates" / name if name else None,
                    )
                    self.assertIn("--arch=wse3", command)
                    self.assertTrue((work / "decode_attention_oracle.py").exists())

    def test_control_cache_masking_and_dispatch_cannot_move(self):
        profile = get_profile("decode-attention.d64-ctx128")
        root = baseline_source(profile)[0]
        source = (root / "src/pe.csl").read_text()
        for old, new in (
            ("clock.get_timestamp(&begin);", ""),
            ("step < T", "step < 1"),
            ("V[dim*S+position%S] = NEW_V[step*D+dim];", ""),
            ("slot < valid", "slot < S"),
            ("mpi.reduce_fadds(ROOT", "mpi.reduce_fadds(0"),
            ("store_output();", ""),
        ):
            # The local compute body can change; replace the last occurrence,
            # which belongs to immutable cache/control for the causal-mask case.
            index = source.rfind(old)
            self.assertGreaterEqual(index, 0)
            with self.subTest(old=old), self.assertRaisesRegex(ValueError, "protected"):
                verify_timing_contract(
                    source, source[:index] + source[index:].replace(old, new, 1), "decode_attention"
                )
        layout = (root / "src/layout.csl").read_text()
        with self.assertRaisesRegex(ValueError, "protected"):
            verify_timing_contract(
                layout,
                layout.replace("@set_rectangle(P, 1)", "@set_rectangle(1, 1)"),
                "decode_attention",
            )

    def test_local_optimization_remains_editable(self):
        profile = get_profile("decode-attention.d64-ctx128")
        source = (baseline_source(profile)[0] / "src/pe.csl").read_text()
        verify_timing_contract(
            source,
            source.replace("return math.exp(x);", "return math.exp(x) * 1.0;"),
            "decode_attention",
        )

    def test_coordinator_metric_drives_gates_and_leaderboard(self):
        # Synthetic gate fixtures, not measured performance evidence.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [
                make_record(
                    root,
                    name,
                    variant=variant,
                    seed=seed,
                    cycles=cycles,
                    local_cycles=1000,
                    profile_id="decode-attention.d64-ctx128",
                )
                for name, variant, seed, cycles in (
                    ("base", "baseline", 7, 100),
                    ("candidate", "candidate", 7, 10),
                    ("fresh-base", "baseline", 8, 100),
                    ("fresh-candidate", "candidate", 8, 10),
                )
            ]
            self.assertEqual(compare(*paths[:2])["speedup"], 10)
            self.assertEqual(
                confirm(*paths, minimum_speedup=5.01)["status"], "confirmed_improvement"
            )
            with patch("cerebras_research.ledger.ROOT", root):
                ledger.append_receipts(paths[:2])
                view = ledger.board(ledger.read_rows())
                self.assertEqual(len(view["measured_cycle_memory_front"]), 1)
                self.assertEqual(
                    view["measured_cycle_memory_front"][0]["primary_metric"], "coordinator_cycles"
                )


if __name__ == "__main__":
    unittest.main()
