"""Independent raw replay rejects rehashed recurrent states, inputs and timers."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from check_recurrent_gdn_contracts import simulation
from recurrent_gdn import pack_timer
from recurrent_gdn_oracle import evidence, inputs

from cerebras_research.catalog import get_profile
from cerebras_research.paths import digest
from tools.replay_numerics import check_case


class RecurrentReplayTests(unittest.TestCase):
    def fixture(self, seed=125):
        data = inputs(16, 16, seed, "signed")
        actual = simulation(data, 16, 16)
        arrays, _, bound = evidence(16, 16, data, actual, data)
        words = np.zeros((4, 4, 6), np.uint16)
        words[:, :, 3] = 100
        arrays.update(
            error_bound=bound,
            timer_u32=pack_timer(words),
            timer_memcpy_u16=words.copy(),
            timer_core_u16=words.copy(),
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
            return check_case(get_profile("recurrent-gdn.k16-v16"), report, path)

    def test_complete_operator_replays(self):
        self.check(self.fixture())

    def test_rehashed_state_precision_canonical_bounds_and_timers_reject(self):
        for kind in (
            "history",
            "canonical",
            "final_bit",
            "actual",
            "expected",
            "bound",
            "timer",
            "dtype",
            "labels",
            "offsets",
        ):
            arrays = self.fixture()
            if kind == "history":
                arrays["STATE_HISTORY"][0, 0, 0, 0, 0, 0] += 0.25
            elif kind == "canonical":
                arrays["canonical_S0"].view(np.uint32).reshape(-1)[0] ^= 0x80000000
            elif kind == "final_bit":
                arrays["FINAL_STATE"].view(np.uint32).reshape(-1)[0] ^= 1
            elif kind == "actual":
                arrays["actual"][-1] += 1
            elif kind == "expected":
                arrays["expected"][-1] += 1
            elif kind == "bound":
                arrays["error_bound"] *= 100
            elif kind == "labels":
                arrays["check_labels"][0] = "tampered"
            elif kind == "offsets":
                arrays["check_offsets"][1] += 1
            elif kind == "timer":
                arrays["timer_core_u16"][0, 0, 3] += 1
            else:
                arrays["STATE_HISTORY"] = arrays["STATE_HISTORY"].astype(np.float16)
            with self.subTest(kind=kind), self.assertRaises((ValueError, AssertionError)):
                self.check(arrays)

    def test_declared_seed_and_every_frame_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "seed/family"):
            self.check(self.fixture(seed=126))
        arrays = self.fixture()
        arrays["STATE_HISTORY"] = arrays["STATE_HISTORY"][:, :, :3]
        with self.assertRaises(ValueError):
            self.check(arrays)


if __name__ == "__main__":
    unittest.main()
