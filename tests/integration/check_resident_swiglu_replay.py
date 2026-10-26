"""Corrupt saved chain evidence and require the independent replay to reject it."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))

import numpy as np
from resident_swiglu_oracle import evidence, inputs, packed_inputs, silu

from cerebras_research.catalog import get_profile
from cerebras_research.paths import digest
from tools.replay_numerics import check_case


class ChainReplayTests(unittest.TestCase):
    def check(self, report, path):
        report["artifact_sha256"] = digest(path)
        return check_case(get_profile("resident-swiglu.small"), report, path)

    def fixture(self, directory):
        values = inputs(4, 16, 16, 32, 125, "signed")
        x, wg, wu, wd = values
        g = (x.astype(np.float64) @ wg.astype(np.float64)).astype(np.float16)
        u = (x.astype(np.float64) @ wu.astype(np.float64)).astype(np.float16)
        v = (silu(g) * u.astype(np.float64)).astype(np.float16)
        d = (v.astype(np.float64) @ wd.astype(np.float64)).astype(np.float16)
        stages = dict(g=g, u=u, v=v, d=d, y=d.astype(np.float32) + x.astype(np.float32))
        packed = packed_inputs(4, *values)
        history = np.broadcast_to(packed["WD_CANON"][:, :, None], (4, 4, 4, 32)).copy()
        arrays, error, bound = evidence(
            4, *values, stages, {k: v.ravel() for k, v in packed.items()}, history
        )
        words = np.tile(np.array([0, 100 << 16, 0], np.uint32), 16)
        arrays.update(timer_u32=words, error_bound=bound)
        path = Path(directory) / "case.npz"
        np.savez_compressed(path, **arrays)
        report = dict(
            case="signed",
            seed=125,
            passed=True,
            max_abs_error=float(error.max()),
            max_fraction_of_error_bound=float((error / bound).max()),
            exact_state_match=True,
            metrics=dict(max_local_cycles=100.0, mean_local_cycles=100.0, pe_cycle_spread=0.0),
        )
        return arrays, report, path

    def test_correct_full_chain_replays(self):
        with tempfile.TemporaryDirectory() as folder:
            _, report, path = self.fixture(folder)
            self.check(report, path)

    def test_flattened_actual_cannot_disagree_with_saved_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            arrays, report, path = self.fixture(folder)
            arrays["actual"][0] += 1
            np.savez_compressed(path, **arrays)
            with self.assertRaises((AssertionError, ValueError)):
                self.check(report, path)

    def test_final_frame_and_history_corruption_are_rejected(self):
        for kind in (
            "stage_y",
            "working_w_history",
            "canonical_X_FRAMES",
            "error_bound",
            "expected",
        ):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                arrays, report, path = self.fixture(folder)
                arrays[kind].reshape(-1)[-1] += 1
                np.savez_compressed(path, **arrays)
                with self.assertRaises((AssertionError, ValueError)):
                    self.check(report, path)


if __name__ == "__main__":
    unittest.main()
