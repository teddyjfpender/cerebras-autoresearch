"""Pre-device corruption, precision, layout and full-chain propagation gates."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))

import numpy as np  # noqa: E402
import resident_swiglu as driver
from common import CASES  # noqa: E402
from resident_swiglu_oracle import evidence, inputs, packed_inputs, silu  # noqa: E402
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


class ChainLifecycleTests(unittest.TestCase):
    def run_main(self, folder):
        path = Path(folder)
        (path / "out.json").write_text(json.dumps(dict(params=dict(P=4, Mt=4, Ht=4, Ft=8, T=4))))
        before = Path.cwd()
        try:
            with patch.object(
                sys,
                "argv",
                ["resident_swiglu.py", "--name", str(path), "--case", "signed", "--seed", "7"],
            ):
                driver.main()
        finally:
            os.chdir(before)

    def test_shutdown_fault_prevents_every_pass_and_snapshot(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(driver, "get_platform"),
            patch.object(driver, "SdkRuntime") as runtime,
            patch.object(driver, "read_core_arrays") as capture,
            patch.object(driver, "save_case") as save,
        ):
            runtime.return_value.stop.side_effect = RuntimeError("shutdown fault")
            with self.assertRaisesRegex(RuntimeError, "shutdown fault"):
                self.run_main(folder)
            self.assertEqual(runtime.return_value.memcpy_d2h.call_args.args[4:7], (4, 4, 3))
            capture.assert_not_called()
            save.assert_not_called()

    def test_stale_core_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(driver, "SdkRuntime") as runtime:
            (Path(folder) / "out.core").touch()
            with self.assertRaisesRegex(RuntimeError, "stale core"):
                self.run_main(folder)
            runtime.assert_not_called()

    def test_fp32_output_snapshot_keeps_every_word_and_layout(self):
        values = (np.arange(4 * 16 * 16, dtype=np.float32).reshape(4, 16, 16) / 17).copy()
        values.reshape(-1)[0] = -0.0
        packed = values.reshape(4, 4, 4, 4, 4).transpose(1, 3, 0, 4, 2).copy().reshape(-1)
        decoded = driver.unpack(packed.view(np.float16), 4, 4, 4, 4, np.float32)
        self.assertEqual(decoded.dtype, np.float32)
        self.assertEqual(decoded.tobytes(), values.tobytes())


if __name__ == "__main__":
    unittest.main()
