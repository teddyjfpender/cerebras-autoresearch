"""Pre-device corruption, precision, layout and full-chain propagation gates."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))

import numpy as np  # noqa: E402
from common import CASES  # noqa: E402
from oracle import evidence, inputs, packed_inputs, silu  # noqa: E402
from reusable_gemm_oracle import pack_inputs, unpack_outputs  # noqa: E402


def reference(x, wg, wu, wd):
    g = (x.astype(np.float64) @ wg.astype(np.float64)).astype(np.float16)
    u = (x.astype(np.float64) @ wu.astype(np.float64)).astype(np.float16)
    v = (silu(g) * u.astype(np.float64)).astype(np.float16)
    d = (v.astype(np.float64) @ wd.astype(np.float64)).astype(np.float16)
    return dict(g=g, u=u, v=v, d=d, y=d.astype(np.float32) + x.astype(np.float32))


class OracleTests(unittest.TestCase):
    def fixture(self, case="signed", h=16, f=32):
        values = inputs(4, 16, h, f, 7301, case)
        packed = packed_inputs(4, *values)
        history = np.broadcast_to(packed["WD_CANON"][:, :, None], (4, 4, 4, f * h // 16)).copy()
        return values, reference(*values), {k: v.ravel().copy() for k, v in packed.items()}, history

    def test_all_families_both_shapes(self):
        for h, f in [(16, 32), (80, 272)]:
            for case in CASES:
                with self.subTest(h=h, f=f, case=case):
                    values, stages, canonical, history = self.fixture(case, h, f)
                    arrays, error, bound = evidence(4, *values, stages, canonical, history)
                    self.assertTrue(arrays["exact_state_match"])
                    self.assertTrue(np.all(error <= bound))

    def test_every_stage_corruption_rejected(self):
        for name in "guvdy":
            with self.subTest(stage=name):
                values, stages, canonical, history = self.fixture()
                stages[name].flat[0] += 10
                _, error, bound = evidence(4, *values, stages, canonical, history)
                self.assertFalse(np.all(error <= bound))

    def test_signed_zero_canonical_and_every_weight(self):
        for name in ["X_FRAMES", "WG_CANON", "WU_CANON", "WD_CANON"]:
            values, stages, canonical, history = self.fixture("zero")
            canonical[name].view(np.uint16)[0] ^= 0x8000
            arrays, _, _ = evidence(4, *values, stages, canonical, history)
            self.assertFalse(arrays["exact_state_match"])

    def test_every_frame_weight_state_rejected(self):
        for frame in range(4):
            values, stages, canonical, history = self.fixture()
            history[3, 2, frame, -1] += 1
            arrays, _, _ = evidence(4, *values, stages, canonical, history)
            self.assertFalse(arrays["exact_state_match"])

    def test_projection_output_layout_is_next_input_layout(self):
        t, m, f = 4, 16, 272
        values = np.arange(t * m * f, dtype=np.float32).reshape(t, m, f).astype(np.float16)
        raw = values.reshape(t, 4, 4, 4, 68).transpose(1, 3, 0, 4, 2).copy()
        next_x, _ = pack_inputs(4, 4, 68, 20, values, np.zeros((f, 80), np.float16))
        self.assertEqual(next_x.tobytes(), raw.tobytes())
        np.testing.assert_array_equal(unpack_outputs(4, 4, 68, t, raw), values)

    def test_full_chain_has_independent_ideal_output_check(self):
        values, stages, canonical, history = self.fixture()
        arrays, _, bound = evidence(4, *values, stages, canonical, history)
        y_size = stages["y"].size
        self.assertEqual(arrays["actual"].size, sum(a.size for a in stages.values()) + y_size)
        self.assertTrue(np.all(bound[-y_size:] > 0))
        self.assertFalse(np.array_equal(arrays["expected"][-y_size:], stages["y"].ravel()))

    def test_shape_precision_and_nonfinite_rejected(self):
        for kind in ("shape", "dtype", "nonfinite"):
            values, stages, canonical, history = self.fixture()
            if kind == "shape":
                stages["g"] = stages["g"][:-1]
            elif kind == "dtype":
                stages["d"] = stages["d"].astype(np.float32)
            else:
                stages["u"].flat[0] = np.inf
            with self.assertRaises(ValueError):
                evidence(4, *values, stages, canonical, history)


if __name__ == "__main__":
    unittest.main()
