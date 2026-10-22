import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cerebras_research.adapters import functions, materialize, verify_timing_contract
from cerebras_research.catalog import get_profile, get_suite, load_catalog
from cerebras_research.paths import contained
from cerebras_research.process import simulator_lease


class CatalogTests(unittest.TestCase):
    def test_broad_coverage_has_explicit_validation_and_metric(self):
        profiles = load_catalog()
        self.assertGreaterEqual(len({p["family"] for p in profiles.values()}), 8)
        self.assertGreaterEqual(len(profiles), 25)
        for profile in profiles.values():
            self.assertIn(
                profile["validation"],
                ("independent_six_case", "upstream_reference", "diagnostic_only"),
            )
            self.assertTrue(profile["work_unit"])
        self.assertTrue(get_suite("regression"))

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                contained(folder, "../private")

    def test_timing_and_repeat_controls_are_protected(self):
        original = "fn compute() void { clock.start(); pipeline(); clock.stop(); }\nfn pipeline() void { work(); }"
        changed = original.replace("clock.start();", "")
        with self.assertRaisesRegex(ValueError, "protected"):
            verify_timing_contract(original, changed, "pipeline")
        verify_timing_contract(original, original.replace("work();", "faster_work();"), "pipeline")
        self.assertIn("compute", functions(original))

    def test_cold_gemm_profile_documents_correct_timing_scope(self):
        profile = get_profile("meshgemm.cold-tiny")
        self.assertEqual(profile["timing_protocol"]["warmups"], 0)
        self.assertEqual(profile["timing_protocol"]["repeats"], 1)

    def test_manifest_cannot_misstate_driver_grid_or_repeat_count(self):
        original = get_profile("meshgemv.square64")
        for field, value in (
            ("pe_grid", [4, 3]),
            ("timing_protocol", {"warmups": 0, "repeats": 100}),
            ("timeout_seconds", float("nan")),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                profile = copy.deepcopy(original)
                profile[field] = value
                (root / "profile.json").write_text(
                    json.dumps({"schema_version": 1, "profiles": [profile]})
                )
                catalog = root / "catalog.json"
                catalog.write_text(json.dumps({"schema_version": 1, "includes": ["profile.json"]}))
                with (
                    patch("cerebras_research.catalog.CATALOG", catalog),
                    patch("cerebras_research.catalog.ROOT", root),
                ):
                    with self.assertRaises(ValueError):
                        load_catalog()

    def test_simulator_lease_rejects_overlapping_runner(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("cerebras_research.process.ROOT", Path(folder)):
                with simulator_lease():
                    with self.assertRaises(RuntimeError):
                        with simulator_lease():
                            pass

    def test_shared_library_is_inside_the_isolated_source_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "upstream/example"
            (source / "src").mkdir(parents=True)
            original = '@import_module("../../benchmark-libs/shared/pe.csl");'
            (source / "src/layout.csl").write_text(original)
            library = root / "upstream/benchmark-libs/shared"
            library.mkdir(parents=True)
            dynamic_path = 'fn module() type { return "../../benchmark-libs/shared/wse3/pe.csl"; }'
            (library / "pe.csl").write_text(dynamic_path)
            with patch("cerebras_research.adapters.baseline_source", return_value=(source, {})):
                materialize({"shared_libraries": ["shared"]}, root / "work")
            self.assertEqual((source / "src/layout.csl").read_text(), original)
            self.assertEqual(
                original.replace("../../", "../"),
                (root / "work/src/layout.csl").read_text(),
            )
            self.assertEqual(
                dynamic_path.replace("../../", "../"),
                (root / "work/benchmark-libs/shared/pe.csl").read_text(),
            )
            self.assertEqual((library / "pe.csl").read_text(), dynamic_path)


if __name__ == "__main__":
    unittest.main()
