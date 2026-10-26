"""The recurrent core mapper checks every mixed-precision array and timer bit."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
sys.path.insert(0, str(Path(__file__).parent))
import check_recurrent_gdn_replay as replay_fixture
import numpy as np
from recurrent_gdn import capture_contract, output_shapes

from cerebras_research.catalog import get_profile
from cerebras_research.paths import digest
from tools import replay_snapshots as audit


class RecurrentSnapshotTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        out = root / "work/out"
        out.mkdir(parents=True)
        arrays = replay_fixture.RecurrentReplayTests().fixture()
        names = ("Q_CANON", "K_CANON", "V_CANON", "G_CANON", "BETA_CANON", "S0")
        data = {name: arrays["input_" + name] for name in names}
        shapes, expected = capture_contract(16, 16, data)
        values = {name: arrays["canonical_" + name] for name in names}
        values.update(
            {
                name: arrays["FINAL_STATE" if name == "STATE" else name]
                for name in output_shapes(16, 16)
            }
        )
        values["times"] = arrays["timer_core_u16"]
        captured = {
            name: value.ravel().view(np.uint16).view(np.float16).copy()
            for name, value in values.items()
        }
        core = out / "case-signed.core"
        core.write_bytes(b"fixture: core parsing itself uses separately qualified helper")
        artifact = out / "case-signed.npz"
        np.savez_compressed(artifact, **arrays)
        receipt = root / "result.json"
        record = dict(
            profile=get_profile("recurrent-gdn.k16-v16"),
            artifacts={"work/out/case-signed.core": digest(core)},
        )
        case = dict(case="signed", artifact="case-signed.npz", artifact_sha256=digest(artifact))
        return receipt, record, case, captured, expected

    def test_complete_mixed_array_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt, record, case, captured, expected = self.fixture(folder)
            with patch.object(audit, "read_core_grid_arrays", return_value=captured) as capture:
                result = audit.audit_recurrent_case(record, receipt, case)
            self.assertTrue(result["complete_output_state_and_timer_match"])
            self.assertEqual(result["half_words_checked"], 16 * sum(expected.values()))
            self.assertEqual(capture.call_args.args[2:], (4, 4, expected))

    def test_single_core_word_corruption_rejects_for_every_symbol(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt, record, case, captured, _ = self.fixture(folder)
            for name in captured:
                changed = {k: v.copy() for k, v in captured.items()}
                changed[name].view(np.uint16)[0] ^= 1
                with (
                    self.subTest(name=name),
                    patch.object(audit, "read_core_grid_arrays", return_value=changed),
                ):
                    with self.assertRaises((ValueError, AssertionError)):
                        audit.audit_recurrent_case(record, receipt, case)

    def test_core_digest_checked_before_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt, record, case, _, _ = self.fixture(folder)
            (receipt.parent / "work/out/case-signed.core").write_bytes(b"changed")
            with patch.object(audit, "read_core_grid_arrays") as capture:
                with self.assertRaisesRegex(ValueError, "core is missing or changed"):
                    audit.audit_recurrent_case(record, receipt, case)
                capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
