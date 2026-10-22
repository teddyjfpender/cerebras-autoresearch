"""SDK-side oracle properties: run with tools/sdk-run cs_python <this file>."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
from common import CASES, matrix_inputs, save_case, timing
from resident_oracle import evidence, inputs, reference


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

    def test_complete_blocks_have_finite_all_case_references(self):
        dims = dict(H=16, F=32, T=4, C=8, L=4, A=2)
        for kind in ("gated_mlp", "decoder"):
            for case in CASES:
                data = inputs(kind, dims, 7, case)
                expected = reference(kind, dims, data)
                actual = {name: value.astype(np.float32) for name, value in expected.items()}
                arrays, error, bound = evidence(kind, dims, data, actual)
                self.assertTrue(np.isfinite(arrays["expected"]).all())
                self.assertTrue(np.all(error <= bound))
                if kind == "decoder":
                    self.assertEqual(actual["HISTORY"].shape, (4, 2, 8, 16))
                    np.testing.assert_array_equal(
                        data["CACHE"], inputs(kind, dims, 7, case)["CACHE"]
                    )

    def test_decoder_oracle_rejects_early_output_and_state_corruption(self):
        dims = dict(H=16, F=32, T=4, C=10, L=4, A=2)
        data = inputs("decoder", dims, 11, "signed")
        expected = reference("decoder", dims, data)
        # All defects leave the final token output unchanged.
        for name, index in (
            ("Y", (0, 0)),
            ("HISTORY", (0, 0, 4, 0)),
            ("HISTORY", (0, 1, 9, 0)),
            ("CACHE", (0, 0, 0)),
        ):
            with self.subTest(name=name, index=index):
                actual = {key: value.astype(np.float32) for key, value in expected.items()}
                actual[name][index] += 0.01
                _, error, bound = evidence("decoder", dims, data, actual)
                self.assertTrue(np.any(error > bound))
        actual = {key: value.astype(np.float32) for key, value in expected.items()}
        actual["Y"] = actual["Y"][-1:]
        with self.assertRaisesRegex(ValueError, "shape"):
            evidence("decoder", dims, data, actual)

    def test_decoder_reads_cache_and_updates_each_token(self):
        dims = dict(H=16, F=32, T=4, C=10, L=4, A=2)
        data = inputs("decoder", dims, 13, "signed")
        expected = reference("decoder", dims, data)
        changed = {key: value.copy() for key, value in data.items()}
        changed["CACHE"][:, :4] += 2
        self.assertGreater(
            np.max(np.abs(reference("decoder", dims, changed)["Y"] - expected["Y"])), 0.01
        )
        for step in range(4):
            state = expected["HISTORY"][step]
            np.testing.assert_array_equal(state[:, :4], data["CACHE"][:, :4])
            np.testing.assert_array_equal(state[:, 5 + step :], data["CACHE"][:, 5 + step :])
        self.assertGreater(np.max(np.abs(expected["HISTORY"][0] - expected["HISTORY"][-1])), 0.01)

    def test_gated_mlp_is_sensitive_to_both_projections_and_residual(self):
        dims = dict(H=16, F=32, T=4)
        data = inputs("gated_mlp", dims, 17, "signed")
        expected = reference("gated_mlp", dims, data)["Y"]
        for key in ("WG", "WU", "WD", "SCALE1"):
            changed = {name: value.copy() for name, value in data.items()}
            changed[key][:] = 0
            self.assertGreater(
                np.max(np.abs(reference("gated_mlp", dims, changed)["Y"] - expected)), 0.001
            )
        changed = {name: value.copy() for name, value in data.items()}
        changed["WD"][:] = 0
        np.testing.assert_array_equal(reference("gated_mlp", dims, changed)["Y"], data["X"])

    def test_decoder_matches_analytic_uniform_attention_sequence(self):
        dims = dict(H=4, F=4, T=4, C=6, L=1, A=1)
        data = inputs("decoder", dims, 23, "signed")
        data["X"][:] = np.eye(4, dtype=np.float32)
        for name in ("SCALE1", "SCALE2"):
            data[name][:] = 1
        for name in ("WQ", "WK", "WG", "WU", "WD"):
            data[name][:] = 0
        data["WV"][:] = np.eye(4)
        data["WO"][:] = np.eye(4)
        data["CACHE"][:] = 0
        data["CACHE"][1, 0] = [0.25, 0.5, 0.75, 1.0]
        result = reference("decoder", dims, data)
        accumulated = data["CACHE"][1, 0].astype(np.float64)
        for step in range(4):
            normalized = data["X"][step].astype(np.float64) / np.sqrt(0.25 + 1e-5)
            accumulated += normalized
            wanted = data["X"][step] + accumulated / (step + 2)
            np.testing.assert_allclose(result["Y"][step], wanted, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(result["HISTORY"][step, 1, step + 1], normalized, rtol=1e-13)

    def test_gated_mlp_matches_diagonal_closed_form(self):
        dims = dict(H=2, F=2, T=4)
        data = inputs("gated_mlp", dims, 29, "signed")
        data["SCALE1"][:] = 1
        for name in ("WG", "WU", "WD"):
            data[name][:] = np.eye(2)
        x = data["X"].astype(np.float64)
        normalized = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-5)
        wanted = x + normalized * normalized / (1 + np.exp(-normalized))
        np.testing.assert_allclose(reference("gated_mlp", dims, data)["Y"], wanted, rtol=1e-13)

    def test_untouched_cache_requires_exact_bits_and_preserves_failure(self):
        dims = dict(H=4, F=4, T=4, C=6, L=1, A=1)
        data = inputs("decoder", dims, 31, "signed")
        data["CACHE"][0, 0, 0] = np.float32(0.0)
        for replacement in (np.float32(-0.0), np.nextafter(np.float32(0), np.float32(1))):
            expected = reference("decoder", dims, data)
            actual = {key: value.astype(np.float32) for key, value in expected.items()}
            actual["CACHE"][0, 0, 0] = replacement
            arrays, error, bound = evidence("decoder", dims, data, actual)
            self.assertFalse(bool(arrays["exact_state_match"]))
            # The ordinary tolerance alone would accept either defect.
            self.assertTrue(np.all(error <= bound))
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(AssertionError):
                    save_case(
                        folder,
                        "signed",
                        arrays,
                        error,
                        bound,
                        np.array([packed(10, 20)], np.uint32),
                        1,
                        31,
                    )
                with np.load(Path(folder) / "case-signed.npz") as raw:
                    self.assertFalse(bool(raw["exact_state_match"]))


if __name__ == "__main__":
    unittest.main()
