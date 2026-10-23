"""Unscored native layout tests, run with the staged SDK NumPy dependencies."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))

import numpy as np
import recurrent_gdn_native_oracle as oracle
from recurrent_gdn_native_layout import Grid, profile


class NativeLayout(unittest.TestCase):
    def test_all_column_and_state_words_roundtrip(self):
        for t in (4, 8):
            grid = profile(t)
            for dtype in (np.float32, np.uint16):
                cols = np.arange(t * 3 * 128, dtype=dtype).reshape(t, 3, 128)
                state = np.arange(t * 3 * 128 * 128, dtype=dtype).reshape(t, 3, 128, 128)
                self.assertEqual(cols.tobytes(), grid.logical_columns(grid.columns(cols)).tobytes())
                self.assertEqual(
                    state.tobytes(), grid.logical_columns(grid.columns(state, True), True).tobytes()
                )
                self.assertEqual(
                    state[0].tobytes(), grid.logical_initial(grid.initial(state[0])).tobytes()
                )

    def test_every_key_and_head_replica_is_checked(self):
        for t in (4, 8):
            grid = profile(t)
            raw = oracle.inputs(t, 20261038, "signed")
            for name, pack, unpack in (
                ("Q_CANON", grid.key, grid.logical_key),
                ("G_CANON", grid.per_head, grid.logical_head),
            ):
                a = pack(raw[name])
                self.assertEqual(raw[name].tobytes(), unpack(a).tobytes())
                a.reshape(-1)[-1] += 1
                with self.assertRaises(ValueError):
                    unpack(a)

    def test_declared_global_head_column_mapping(self):
        for t in (4, 8):
            grid = profile(t)
            raw = oracle.inputs(t, 20261038, "signed")
            physical = grid.inputs(raw)
            for y in range(grid.height):
                for x in range(grid.width):
                    index = y * grid.width + x
                    head = index // grid.shards
                    first = (index % grid.shards) * grid.tile_v
                    columns = slice(first, first + grid.tile_v)
                    np.testing.assert_array_equal(
                        physical["S0"][y, x, 0], raw["S0"][head, :, columns]
                    )
                    np.testing.assert_array_equal(
                        physical["V_CANON"][y, x, :, 0], raw["V_CANON"][:, head, columns]
                    )
                    for name in ("Q_CANON", "K_CANON"):
                        self.assertEqual(physical[name][y, x].tobytes(), raw[name].tobytes())
                    for name in ("G_CANON", "BETA_CANON"):
                        self.assertEqual(
                            physical[name][y, x, :, 0].tobytes(), raw[name][:, head].tobytes()
                        )

    def test_wrong_area_or_incomplete_shapes_rejected(self):
        with self.assertRaises(ValueError):
            Grid(4, 8, 4, 4)
        grid = profile(4)
        with self.assertRaises(ValueError):
            grid.logical_initial(np.zeros((6, 8, 1, 127, 8), np.float32))
        with self.assertRaises(ValueError):
            grid.key(np.zeros((4, 127), np.float32))


if __name__ == "__main__":
    unittest.main()
