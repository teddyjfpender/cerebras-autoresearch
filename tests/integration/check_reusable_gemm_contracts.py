"""Pure packing checks in SDK NumPy; no simulator launch or qualification claim."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
import reusable_gemm as driver  # noqa: E402
from reusable_gemm import read_symbol, read_weight_history  # noqa: E402
from reusable_gemm_oracle import evidence, pack_inputs, unpack_outputs  # noqa: E402


class SequencePackingTests(unittest.TestCase):
    def run_main(self, folder):
        path = Path(folder)
        (path / "out.json").write_text(
            json.dumps({"params": {"P": 4, "Mt": 4, "Kt": 4, "Nt": 4, "T": 4}})
        )
        before = Path.cwd()
        try:
            with patch.object(
                sys,
                "argv",
                ["reusable_gemm.py", "--name", str(path), "--case", "positive", "--seed", "7"],
            ):
                driver.main()
        finally:
            os.chdir(before)

    def test_shutdown_failure_cannot_save_a_case_or_read_a_snapshot(self):
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
            fence = runtime.return_value.memcpy_d2h.call_args
            self.assertIsNotNone(fence)
            self.assertEqual(fence.args[4:7], (4, 4, 3))
            self.assertFalse(fence.kwargs["nonblock"])
            capture.assert_not_called()
            save.assert_not_called()

    def test_stale_transient_core_is_rejected_before_device_execution(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(driver, "SdkRuntime") as runtime:
            (Path(folder) / "out.core").touch()
            with self.assertRaisesRegex(RuntimeError, "stale simulator core"):
                self.run_main(folder)
            runtime.assert_not_called()

    def test_frame_alias_readback_reconstructs_every_state_word_in_pe_frame_order(self):
        p, t, count = 4, 4, 1360
        calls = []

        class FakeRuntime:
            def get_id(self, symbol):
                return symbol

            def memcpy_d2h(self, destination, symbol, x, y, width, height, length, **options):
                frame = int(symbol.rsplit("_", 1)[1])
                calls.append((frame, width * height * length, length))
                destination[:] = [
                    (((row * p + column) * t + frame) * count + index) & 0xFFFF
                    for row in range(y, y + height)
                    for column in range(x, x + width)
                    for index in range(length)
                ]

        result = read_weight_history(FakeRuntime(), p, t, count, {})
        expected = np.arange(p * p * t * count, dtype=np.uint32).astype(np.uint16)
        np.testing.assert_array_equal(result.view(np.uint16), expected)
        self.assertEqual({frame for frame, _, _ in calls}, set(range(t)))
        self.assertEqual(sum(size for _, size, _ in calls), expected.size)
        self.assertTrue(all(size <= 32768 and length == count for _, size, length in calls))

    def check_readback(self, count):
        calls = []

        class FakeRuntime:
            def get_id(self, symbol):
                return 7

            def memcpy_d2h(self, destination, identifier, x, y, width, height, length, **options):
                calls.append((x, y, width, height, length))
                destination[:] = [
                    ((row * 4 + column) * count + index) & 0xFFFF
                    for row in range(y, y + height)
                    for column in range(x, x + width)
                    for index in range(length)
                ]

        actual = read_symbol(FakeRuntime(), "W_HISTORY", 4, count, {})
        expected = np.arange(16 * count, dtype=np.uint32).astype(np.uint16)
        np.testing.assert_array_equal(actual.view(np.uint16), expected)
        self.assertEqual(sum(w * h * n for _, _, w, h, n in calls), 16 * count)
        self.assertTrue(all(w * h * n <= 32768 for _, _, w, h, n in calls))
        return calls

    def test_large_readback_preserves_every_pe_word_across_bounded_requests(self):
        self.assertGreater(len(self.check_readback(5440)), 1)

    def test_small_readback_keeps_one_request_and_exact_half_bits(self):
        self.assertEqual(len(self.check_readback(64)), 1)

    def test_impossible_readback_is_rejected_before_transfer(self):
        with self.assertRaisesRegex(ValueError, "transport word budget"):
            read_symbol(None, "W_HISTORY", 4, 10000, {})

    def test_every_input_frame_has_the_required_pe_and_column_major_layout(self):
        frames = np.arange(4 * 16 * 16, dtype=np.float16).reshape(4, 16, 16)
        weights = np.arange(16 * 16, dtype=np.float16).reshape(16, 16)
        packed, _ = pack_inputs(4, 4, 4, 4, frames, weights)
        for pe_row in range(4):
            for pe_col in range(4):
                for frame in range(4):
                    for k in range(4):
                        for row in range(4):
                            self.assertEqual(
                                packed[pe_row, pe_col, frame, k * 4 + row],
                                frames[frame, pe_row * 4 + row, pe_col * 4 + k],
                            )

    def test_canonical_weight_tiles_use_the_known_four_pe_permutation(self):
        frames = np.zeros((4, 16, 16), np.float16)
        weights = np.arange(16 * 16, dtype=np.float16).reshape(16, 16)
        _, packed = pack_inputs(4, 4, 4, 4, frames, weights)
        origins = [[0, 1, 2, 3], [1, 3, 0, 2], [2, 0, 3, 1], [3, 2, 1, 0]]
        for pe_row in range(4):
            for pe_col in range(4):
                for k in range(4):
                    for col in range(4):
                        self.assertEqual(
                            packed[pe_row, pe_col, k * 4 + col],
                            weights[origins[pe_row][pe_col] * 4 + k, pe_col * 4 + col],
                        )

    def test_output_collection_preserves_every_global_element_and_frame(self):
        wanted = np.arange(4 * 16 * 16, dtype=np.float16).reshape(4, 16, 16)
        tiles = np.empty((4, 4, 4, 16), np.float16)
        for pe_row in range(4):
            for pe_col in range(4):
                for frame in range(4):
                    for col in range(4):
                        for row in range(4):
                            tiles[pe_row, pe_col, frame, col * 4 + row] = wanted[
                                frame, pe_row * 4 + row, pe_col * 4 + col
                            ]
        np.testing.assert_array_equal(unpack_outputs(4, 4, 4, 4, tiles.ravel()), wanted)

    def test_last_frame_output_and_middle_weight_state_are_checked(self):
        frames = (np.arange(4 * 16 * 16).reshape(4, 16, 16) / 1024).astype(np.float16)
        weights = np.eye(16, dtype=np.float16)
        px, pw = pack_inputs(4, 4, 4, 4, frames, weights)
        actual = frames.astype(np.float64).copy()
        history = np.broadcast_to(pw[:, :, None], (4, 4, 4, 16)).copy()
        arrays, error, bound = evidence(4, frames, weights, actual, history, px.ravel(), pw.ravel())
        self.assertTrue(bool(arrays["exact_state_match"]))
        self.assertTrue(np.all(error <= bound))
        actual[-1, 0, 0] += 1
        _, error, bound = evidence(4, frames, weights, actual, history, px.ravel(), pw.ravel())
        self.assertFalse(np.all(error <= bound))
        history[0, 0, 1, 0] += 1
        arrays, _, _ = evidence(4, frames, weights, actual, history, px.ravel(), pw.ravel())
        self.assertFalse(bool(arrays["exact_state_match"]))

    def test_canonical_signed_zero_change_is_rejected(self):
        frames = np.zeros((4, 16, 16), np.float16)
        weights = np.eye(16, dtype=np.float16)
        px, pw = pack_inputs(4, 4, 4, 4, frames, weights)
        history = np.broadcast_to(pw[:, :, None], (4, 4, 4, 16)).copy()
        after = px.ravel().copy()
        after[0] = np.float16(-0.0)
        arrays, _, _ = evidence(
            4, frames, weights, frames.astype(np.float64), history, after, pw.ravel()
        )
        self.assertFalse(bool(arrays["exact_state_match"]))


if __name__ == "__main__":
    unittest.main()
