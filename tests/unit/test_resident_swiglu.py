import copy
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cerebras_research.adapters import functions, prepare, verify_timing_contract
from cerebras_research.catalog import get_profile, load_catalog
from cerebras_research.paths import ROOT


class ResidentSwiGLUTests(unittest.TestCase):
    def test_profiles_count_only_projection_arithmetic_and_lock_layout(self):
        for name in ("resident-swiglu.small", "resident-swiglu.ffn-scaled"):
            profile = get_profile(name)
            dims = profile["dimensions"]
            self.assertEqual(profile["pe_grid"], [4, 4])
            self.assertEqual(profile["precision"], "fp16-fp32")
            self.assertEqual(profile["work_unit"], "projection_arithmetic_operations")
            self.assertEqual(
                profile["work_units"], 6 * dims["T"] * dims["M"] * dims["H"] * dims["F"]
            )

    def test_invalid_shape_state_precision_and_work_count_are_rejected(self):
        original = get_profile("resident-swiglu.small")
        variants = [
            dict(pe_grid=[1, 1]),
            dict(precision="fp16"),
            dict(oracle_contract="wrong"),
            dict(work_units=1),
            dict(work_unit="all_operations"),
            dict(dimensions=dict(M=16, H=16, F=32, T=3)),
            dict(dimensions=dict(M=16, H=16, F=30, T=4)),
            dict(dimensions=dict(M=16, H=32, F=16, T=4)),
            dict(dimensions=dict(M=16, H=16, F=2048, T=4)),
        ]
        for changes in variants:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                profile = copy.deepcopy(original)
                profile.update(changes)
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

    def test_all_chain_controls_and_evidence_stores_are_protected(self):
        source = (ROOT / "kernels/composed/resident-swiglu/src/meshgemm.csl").read_text()
        for name in (
            "sequence_host",
            "save_stage",
            "exit",
            "init_task",
            "meshgemm_host",
            "meshgemm_entry",
            "f_memcpy_timestamps",
            "f_reference_timestamps",
        ):
            body = functions(source)[name]
            changed = source.replace(body, body.replace("{", "{ changed();", 1))
            with self.subTest(function=name), self.assertRaisesRegex(ValueError, "protected"):
                verify_timing_contract(source, changed, "resident_swiglu")
        body = functions(source)["mm_compute"]
        verify_timing_contract(
            source, source.replace(body, body.replace("{", "{ faster();", 1)), "resident_swiglu"
        )

    def test_every_exported_pointer_is_global_and_matches_layout_mutability(self):
        folder = ROOT / "kernels/composed/resident-swiglu/src"
        source = (folder / "meshgemm.csl").read_text()
        layout = (folder / "layout.csl").read_text()
        exports = dict(
            (name, ptr) for ptr, name in re.findall(r'@export_symbol\((\w+), "(\w+)"\)', source)
        )
        declarations = dict(
            (name, kind)
            for kind, name in re.findall(r"^(var|const) (\w+): \[\*\](?:f16|f32)", source, re.M)
        )
        names = re.findall(r'@export_name\("(\w+)", \[\*\](?:f16|f32), (true|false)\)', layout)
        self.assertGreaterEqual(len(names), 13)
        for name, mutable in names:
            with self.subTest(symbol=name):
                self.assertIn(name, exports)
                self.assertEqual(
                    declarations.get(exports[name]), "var" if mutable == "true" else "const"
                )

    def test_materialized_sources_and_params_include_complete_chain(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "work"
            args, _, _ = prepare(get_profile("resident-swiglu.ffn-scaled"), target, "baseline")
            self.assertIn("--params=P:4,Mt:4,Ht:20,Ft:68,T:4", args)
            self.assertTrue((target / "resident_swiglu.py").is_file())
            self.assertTrue((target / "resident_swiglu_oracle.py").is_file())
            source = (target / "src/meshgemm.csl").read_text()
            self.assertIn("var Y_HISTORY: [T*Mt*Ht]f32;", source)
            self.assertIn("var W_HISTORY_3:", source)
            self.assertIn("nonlinear_product();", source)


if __name__ == "__main__":
    unittest.main()
