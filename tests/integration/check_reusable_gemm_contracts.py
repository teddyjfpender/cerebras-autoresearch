"""Pure packing checks in SDK NumPy; no simulator launch or qualification claim."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
from reusable_gemm_oracle import evidence, pack_inputs, unpack_outputs  # noqa: E402


class SequencePackingTests(unittest.TestCase):
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
