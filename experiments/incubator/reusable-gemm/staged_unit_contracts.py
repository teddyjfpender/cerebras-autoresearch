"""Catalog and protected sequence controls for the proposed adapter."""

import unittest

from cerebras_research.adapters import baseline_source, verify_timing_contract
from cerebras_research.catalog import get_suite


class ReusableGemmContractTests(unittest.TestCase):
    def test_full_sequence_dimensions_and_work_are_registered(self):
        profiles = get_suite("reusable-gemm")
        self.assertEqual(len(profiles), 3)
        for profile in profiles:
            dims = profile["dimensions"]
            self.assertEqual(profile["adapter"], "reusable_gemm")
            self.assertEqual(profile["pe_grid"], [4, 4])
            self.assertEqual(profile["oracle_contract"], "resident_gemm_sequence_fp16_v1")
            self.assertEqual(dims["T"], 4)
            self.assertEqual(profile["timing_protocol"]["repeats"], 1)
            self.assertEqual(
                profile["work_units"], 2 * dims["T"] * dims["M"] * dims["K"] * dims["N"]
            )

    def test_timer_every_frame_and_weight_snapshot_are_protected(self):
        profile = get_suite("reusable-gemm")[0]
        source = (baseline_source(profile)[0] / "src/meshgemm.csl").read_text()
        for old, new in (
            ("frame < T", "frame < 1"),
            ("timestamp.get_timestamp(&tscStartBuffer);", ""),
            ("@fmovh(weights, working_w0_dsd);", ""),
        ):
            with self.subTest(old=old), self.assertRaisesRegex(ValueError, "protected"):
                verify_timing_contract(source, source.replace(old, new), "reusable_gemm")
