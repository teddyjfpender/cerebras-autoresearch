import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cerebras_research.adapters import prepare, verify_timing_contract
from cerebras_research.catalog import get_profile, load_catalog
from cerebras_research.paths import ROOT


class RecurrentGDNTests(unittest.TestCase):
    def test_native_precision_and_complete_state_source(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder) / "work"
            args, _, _ = prepare(get_profile("recurrent-gdn.k32-v16"), work, "baseline")
            self.assertIn("--fp16-format=bf16", args)
            self.assertIn("--params=K:32,V:16,T:4", args)
            self.assertTrue((work / "recurrent_gdn.py").is_file())
            text = (work / "src/pe.csl").read_text()
            self.assertIn("var STATE_HISTORY: [T*R*K*V]f32;", text)
            self.assertIn("var STATE: [R*K*V]f32;", text)
            self.assertIn("var Y_HISTORY: [T*R*V]fp;", text)
            self.assertIn("@fmovs(state_dsd, initial_dsd);", text)
            self.assertIn("one_frame(frame); save_frame(frame);", text)

    def test_profile_precision_grouping_work_and_state_are_fixed(self):
        original = get_profile("recurrent-gdn.k16-v16")
        for change in (
            dict(precision="bf16"),
            dict(work_units=1),
            dict(dimensions=dict(K=16, V=16, T=4, R=1)),
            dict(pe_grid=[1, 1]),
            dict(oracle_contract="other"),
        ):
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

    def test_whole_sequence_and_full_state_saves_are_protected(self):
        source = (ROOT / "kernels/attention/recurrent-gdn/src/pe.csl").read_text()
        for name in ("sequence_host", "save_frame"):
            changed = source.replace("fn " + name + "(", "fn changed_" + name + "(")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "protected"):
                verify_timing_contract(source, changed, "recurrent_gdn")
