import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cerebras_research.adapters import prepare, verify_timing_contract
from cerebras_research.catalog import get_profile, load_catalog
from cerebras_research.paths import ROOT


class BF16ChainTests(unittest.TestCase):
    def test_materialization_requests_native_bf16_and_distinct_driver(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder) / "work"
            args, _, _ = prepare(get_profile("resident-swiglu-bf16.ffn-scaled"), work, "baseline")
            self.assertIn("--fp16-format=bf16", args)
            self.assertIn("--params=P:4,Mt:4,Ht:20,Ft:68,T:4", args)
            self.assertTrue((work / "resident_swiglu_bf16.py").exists())
            source = (work / "src/meshgemm.csl").read_text()
            self.assertIn("var res_tile: [Mt*Ft]f32 = @zeros([Mt*Ft]f32);", source)
            self.assertIn("@fmachs(", source)
            self.assertNotIn("@fmach(", source)
            self.assertIn("const u = @as(f32, U_HISTORY[frame*Mt*Ft+i]);", source)
            self.assertIn("Y_HISTORY[frame*Mt*Ht+i] = @as(fp, @as(f32, D_HISTORY", source)
            self.assertEqual(source.count("@fs2h(target,"), 3)

    def test_precision_oracle_work_and_transport_contract_cannot_drift(self):
        original = get_profile("resident-swiglu-bf16.small")
        variants = [
            dict(precision="bf16"),
            dict(oracle_contract="resident_swiglu_sequence_fp16_fp32_v1"),
            dict(work_units=1),
            dict(dimensions=dict(M=4, H=4, F=4, T=4), work_units=1536),
        ]
        for change in variants:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                profile = copy.deepcopy(original)
                profile.update(change)
                (root / "profile.json").write_text(
                    json.dumps(dict(schema_version=1, profiles=[profile]))
                )
                catalog = root / "catalog.json"
                catalog.write_text(json.dumps(dict(schema_version=1, includes=["profile.json"])))
                with (
                    patch("cerebras_research.catalog.CATALOG", catalog),
                    patch("cerebras_research.catalog.ROOT", root),
                ):
                    with self.assertRaises(ValueError):
                        load_catalog()

    def test_complete_chain_controls_and_stage_evidence_are_protected(self):
        source = (ROOT / "kernels/composed/resident-swiglu-bf16/src/meshgemm.csl").read_text()
        for name in ("sequence_host", "save_stage", "exit", "meshgemm_host"):
            changed = source.replace("fn " + name + "(", "fn changed_" + name + "(")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "protected"):
                verify_timing_contract(source, changed, "resident_swiglu_bf16")


if __name__ == "__main__":
    unittest.main()
