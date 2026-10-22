"""Structural gates for whole-block source and independent benchmark profiles."""

import unittest
from pathlib import Path

from cerebras_research.adapters import baseline_source, verify_timing_contract
from cerebras_research.catalog import get_profile, get_suite


class ResidentContractTests(unittest.TestCase):
    def test_all_four_tracks_are_in_the_challenge_suite(self):
        profiles = get_suite("inference-blocks")
        self.assertEqual(len(profiles), 8)
        self.assertEqual(
            {p["adapter"] for p in profiles}, {"meshgemv", "meshgemm", "gated_mlp", "decoder"}
        )
        self.assertTrue(all(p["validation"] == "independent_six_case" for p in profiles))

    def test_token_timer_and_cache_snapshots_are_protected(self):
        profile = get_profile("decoder.h16-f32-ctx4")
        source = (baseline_source(profile)[0] / "src/pe.csl").read_text()
        for original, replacement in (
            ("frame < T", "frame < 1"),
            ("snapshot(frame);", ""),
            ("HISTORY[frame*2*C*H+i] = CACHE[i];", ""),
        ):
            with self.subTest(original=original), self.assertRaisesRegex(ValueError, "protected"):
                verify_timing_contract(source, source.replace(original, replacement), "decoder")
        verify_timing_contract(
            source, source.replace("var maximum: f32", "var maximum: f32"), "decoder"
        )

    def test_sources_expose_every_output_and_full_cache(self):
        for name in ("gated-mlp.h16-f32", "decoder.h16-f32-ctx4"):
            profile = get_profile(name)
            root = baseline_source(profile)[0]
            layout = (root / "src/layout.csl").read_text()
            self.assertIn('@export_name("Y", [*]f32, false)', layout)
            self.assertEqual(profile["precision"], "fp32")
            self.assertEqual(profile["dimensions"]["T"], 4)
            if profile["adapter"] == "decoder":
                self.assertIn('@export_name("HISTORY", [*]f32, false)', layout)
                self.assertIn('@export_name("CACHE", [*]f32, true)', layout)
                self.assertTrue(Path(root / "src/pe.csl").is_file())


if __name__ == "__main__":
    unittest.main()
