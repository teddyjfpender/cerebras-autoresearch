"""SDK-side oracle properties: run with tools/sdk-run cs_python <this file>."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
from common import CASES, matrix_inputs, save_case, timing


def packed(start, end):
    return [start & 0xFFFFFFFF, (start >> 32) | ((end & 0xFFFF) << 16), end >> 16]


class DriverContractTests(unittest.TestCase):
    def test_timestamp_wrap_and_per_pe_duration(self):
        words = np.array([packed((1 << 48) - 3, 7), packed(100, 130)], np.uint32)
        report = timing(words, 2)
        self.assertEqual(report["max_local_cycles"], 15)
        self.assertEqual(report["mean_local_cycles"], 10)
        self.assertEqual(report["pe_cycle_spread"], 10)

    def test_timestamp_words_are_not_converted_through_float_nan(self):
        start = 0x7FC00001
        self.assertEqual(
            timing(np.array([packed(start, start + 101)], np.uint32), 1)["max_local_cycles"], 101
        )

    def test_matrix_cases_have_exact_quantized_semantics(self):
        for dtype in (np.float16, np.float32):
            for case in CASES:
                x, w = matrix_inputs(5, 8, 12, 7, case, dtype)
                self.assertEqual(x.dtype, dtype)
                self.assertTrue(np.isfinite(x).all() and np.isfinite(w).all())
                expected = x.astype(np.float64) @ w.astype(np.float64)
                if case in ("zero", "cancellation"):
                    np.testing.assert_array_equal(expected, np.zeros((5, 12)))
                elif case == "one_hot":
                    np.testing.assert_array_equal(expected, w[:5].astype(np.float64))

    def test_failed_case_preserves_raw_output(self):
        with tempfile.TemporaryDirectory() as folder:
            actual, expected = np.array([2.0]), np.array([0.0])
            with self.assertRaises(AssertionError):
                save_case(
                    folder,
                    "signed",
                    {"actual": actual, "expected": expected},
                    np.abs(actual - expected),
                    np.ones(1),
                    np.array([packed(10, 20)], np.uint32),
                    1,
                    7,
                )
            with np.load(Path(folder) / "case-signed.npz") as raw:
                np.testing.assert_array_equal(raw["actual"], actual)


if __name__ == "__main__":
    unittest.main()
