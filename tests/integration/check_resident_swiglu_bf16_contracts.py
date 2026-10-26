"""Before-device gates for BF16 bits, FP32 accumulation, stages and state."""

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
import resident_swiglu_bf16 as driver  # noqa: E402
from common import CASES  # noqa: E402
from resident_swiglu_bf16_oracle import (  # noqa: E402
    decode,
    encode,
    evidence,
    inputs,
    linear,
    packed_inputs,
    silu,
)


def reference(x, wg, wu, wd):
    x64, wg64, wu64, wd64 = (decode(a).astype(np.float64) for a in (x, wg, wu, wd))
    g, u = (encode(a.astype(np.float32)) for a in (x64 @ wg64, x64 @ wu64))
    v = encode((silu(decode(g)) * decode(u).astype(np.float64)).astype(np.float32))
    d = encode((decode(v).astype(np.float64) @ wd64).astype(np.float32))
    y = encode((decode(d) + decode(x)).astype(np.float32))
    return dict(g=g, u=u, v=v, d=d, y=y)


class BF16OracleTests(unittest.TestCase):
    def fixture(self, case="signed", h=16, f=32):
        values = inputs(4, 16, h, f, 7301, case)
        packed = packed_inputs(4, *values)
        history = np.broadcast_to(packed["WD_CANON"][:, :, None], (4, 4, 4, f * h // 16)).copy()
        return values, reference(*values), {k: v.ravel().copy() for k, v in packed.items()}, history

    def test_all_finite_bf16_words_round_trip_including_signed_zero(self):
        words = np.arange(65536, dtype=np.uint16)
        finite = words[(words & 0x7F80) != 0x7F80]
        self.assertEqual(len(finite), 65280)
        self.assertEqual(encode(decode(finite)).tobytes(), finite.tobytes())

    def test_nearest_even_ties_match_native_fixture(self):
        words = np.array(
            [
                0x3F808000,
                0x3F818000,
                0xBF808000,
                0xBF818000,
                0x00000000,
                0x80000000,
                0x00008000,
                0x00018000,
            ],
            np.uint32,
        )
        expected = np.array([0x3F80, 0x3F82, 0xBF80, 0xBF82, 0, 0x8000, 0, 2], np.uint16)
        np.testing.assert_array_equal(encode(words.view(np.float32)), expected)

    def test_wrong_precision_nonfinite_and_bf16_overflow_rejected(self):
        for values in (
            np.array([1.0], np.float64),
            np.array([np.inf], np.float32),
            np.array([np.finfo(np.float32).max], np.float32),
        ):
            with self.assertRaises(ValueError):
                encode(values)
        for values in (
            np.array([0.0], np.float16),
            np.array([0x7F80], np.uint16),
            np.array([0xFFC1], np.uint16),
        ):
            with self.assertRaises(ValueError):
                decode(values)

    def test_full_chain_all_six_families_both_shapes(self):
        for h, f in [(16, 32), (80, 272)]:
            for case in CASES:
                with self.subTest(h=h, f=f, case=case):
                    values, stages, canonical, history = self.fixture(case, h, f)
                    arrays, error, bound = evidence(4, *values, stages, canonical, history)
                    self.assertTrue(arrays["exact_state_match"])
                    self.assertTrue(np.all(error <= bound))
                    self.assertTrue(np.all(bound > 0))

    def test_every_stage_and_final_frame_corruption_rejected(self):
        for name in "guvdy":
            values, stages, canonical, history = self.fixture()
            decoded = decode(stages[name][-1])
            index = int(np.abs(decoded).argmax())
            stages[name][-1].reshape(-1)[index] ^= 0x8000
            arrays, error, bound = evidence(4, *values, stages, canonical, history)
            self.assertTrue(arrays["exact_state_match"])
            self.assertFalse(np.all(error <= bound), name)

    def test_canonical_signed_zero_and_all_weight_bits_are_exact(self):
        for name in ("X_FRAMES", "WG_CANON", "WU_CANON", "WD_CANON"):
            values, stages, canonical, history = self.fixture("zero")
            canonical[name][-1] ^= 0x8000
            arrays, _, _ = evidence(4, *values, stages, canonical, history)
            self.assertFalse(arrays["exact_state_match"])

    def test_every_past_weight_frame_is_checked(self):
        for frame in range(4):
            values, stages, canonical, history = self.fixture()
            history[-1, -1, frame, -1] ^= 1
            arrays, _, _ = evidence(4, *values, stages, canonical, history)
            self.assertFalse(arrays["exact_state_match"])

    def test_bf16_accumulation_does_not_satisfy_fp32_accumulator_contract(self):
        a = encode(np.ones((1, 272), np.float32))
        b = encode(np.full((272, 1), 1 / 128, np.float32))
        expected, bound = linear(a, b, 4)
        accum = encode(np.zeros((1, 1), np.float32))
        for k in range(272):
            accum = encode(
                (decode(accum) + decode(a[:, k : k + 1]) * decode(b[k : k + 1])).astype(np.float32)
            )
        self.assertEqual(float(decode(accum)[0, 0]), 2.0)
        self.assertEqual(float(expected[0, 0]), 2.125)
        self.assertTrue(np.any(np.abs(decode(accum).astype(np.float64) - expected) > bound))

    def test_distinct_nonzero_frames_and_complete_output_precision(self):
        values, stages, canonical, history = self.fixture("positive")
        self.assertEqual(len({a.tobytes() for a in values[0]}), 4)
        stages["y"] = decode(stages["y"])
        with self.assertRaises(ValueError):
            evidence(4, *values, stages, canonical, history)


class BF16DriverTests(unittest.TestCase):
    def run_main(self, folder):
        path = Path(folder)
        (path / "out.json").write_text(json.dumps(dict(params=dict(P=4, Mt=4, Ht=4, Ft=8, T=4))))
        before = Path.cwd()
        try:
            with patch.object(
                sys, "argv", ["bf16.py", "--name", str(path), "--case", "signed", "--seed", "7"]
            ):
                driver.main()
        finally:
            os.chdir(before)

    def test_shutdown_fault_never_saves_a_pass_or_reads_a_core(self):
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
            capture.assert_not_called()
            save.assert_not_called()
            self.assertEqual(runtime.return_value.memcpy_d2h.call_args.args[4:7], (4, 4, 3))

    def test_stale_core_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(driver, "SdkRuntime") as runtime:
            (Path(folder) / "out.core").touch()
            with self.assertRaisesRegex(RuntimeError, "stale core"):
                self.run_main(folder)
            runtime.assert_not_called()

    def test_uint16_stage_layout_preserves_every_bit_and_frame(self):
        values = np.arange(4 * 16 * 32, dtype=np.uint16).reshape(4, 16, 32)
        values.reshape(-1)[0] = 0x8000
        packed = values.reshape(4, 4, 4, 4, 8).transpose(1, 3, 0, 4, 2).copy()
        decoded = driver.unpack(packed.view(np.float16), 4, 4, 8, 4)
        self.assertEqual(decoded.dtype, np.uint16)
        self.assertEqual(decoded.tobytes(), values.tobytes())


if __name__ == "__main__":
    unittest.main()
