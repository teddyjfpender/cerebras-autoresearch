"""BF16 raw replay detects rehashed precision, shape, numerical and state corruption."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from check_resident_swiglu_bf16_contracts import reference
from resident_swiglu_bf16_oracle import evidence, inputs, packed_inputs

from cerebras_research.catalog import get_profile
from cerebras_research.paths import digest
from tools.replay_numerics import check_case


class BF16ReplayTests(unittest.TestCase):
    def fixture(self):
        values = inputs(4, 16, 16, 32, 125, "signed")
        stages = reference(*values)
        packed = packed_inputs(4, *values)
        history = np.broadcast_to(packed["WD_CANON"][:, :, None], (4, 4, 4, 32)).copy()
        arrays, _, bound = evidence(
            4, *values, stages, {k: v.ravel() for k, v in packed.items()}, history
        )
        arrays.update(
            timer_u32=np.tile(np.array([0, 100 << 16, 0], np.uint32), 16), error_bound=bound
        )
        return arrays

    def check(self, arrays):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "case.npz"
            np.savez_compressed(path, **arrays)
            error = np.abs(arrays["actual"] - arrays["expected"])
            report = dict(
                case="signed",
                seed=125,
                passed=True,
                artifact_sha256=digest(path),
                max_abs_error=float(error.max()),
                max_fraction_of_error_bound=float((error / arrays["error_bound"]).max()),
                exact_state_match=True,
                metrics=dict(max_local_cycles=100.0, mean_local_cycles=100.0, pe_cycle_spread=0.0),
            )
            return check_case(get_profile("resident-swiglu-bf16.small"), report, path)

    def test_complete_bf16_chain_replays_from_raw_words(self):
        self.check(self.fixture())

    def test_rehashed_precision_state_expected_and_bound_changes_rejected(self):
        for kind in ("dtype", "history", "canonical", "actual", "expected", "bound"):
            arrays = self.fixture()
            if kind == "dtype":
                arrays["bits_y"] = arrays["bits_y"].astype(np.float32)
            elif kind == "history":
                arrays["working_w_history"][-1, -1, 0, -1] ^= 1
            elif kind == "canonical":
                arrays["canonical_X_FRAMES"][-1] ^= 0x8000
            elif kind == "actual":
                arrays["actual"][-1] += 1
            elif kind == "expected":
                arrays["expected"][-1] += 1
            else:
                arrays["error_bound"] *= 100
            with self.subTest(kind=kind), self.assertRaises((ValueError, AssertionError)):
                self.check(arrays)

    def test_profile_cannot_claim_four_frames_from_short_inputs(self):
        arrays = self.fixture()
        arrays["x_bits"] = arrays["x_bits"][:2]
        with self.assertRaises(ValueError):
            self.check(arrays)


if __name__ == "__main__":
    unittest.main()
