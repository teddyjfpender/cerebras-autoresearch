import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
sys.path.insert(0, str(Path(__file__).parent))

import unittest

import numpy as np
from recurrent_gdn_full_layout import profile
from recurrent_gdn_full_oracle import inputs


class FullLayoutTests(unittest.TestCase):
    def test_every_declared_global_group_head_and_column(self):
        grid = profile(2)
        data = inputs(2, 125, "signed")
        physical = grid.inputs(data)
        for y in range(24):
            for x in range(32):
                index = y * 32 + x
                group = index // 48
                head = (index // 16) % 3
                column = (index % 16) * 8
                np.testing.assert_array_equal(
                    physical["S0"][y, x, 0], data["S0"][group, head, :, column : column + 8]
                )
                np.testing.assert_array_equal(
                    physical["V_CANON"][y, x, :, 0],
                    data["V_CANON"][group, :, head, column : column + 8],
                )
                for n in ("Q_CANON", "K_CANON"):
                    self.assertEqual(physical[n][y, x].tobytes(), data[n][group].tobytes())
                for n in ("G_CANON", "BETA_CANON"):
                    self.assertEqual(
                        physical[n][y, x, :, 0].tobytes(), data[n][group, :, head].tobytes()
                    )

    def test_state_and_output_column_roundtrips(self):
        from check_full_gdn_contracts import simulation

        grid = profile(2)
        data = inputs(2, 125, "signed")
        a = simulation(data, 2)
        captured = grid.inputs(data)
        captured.update(
            Q_HISTORY=grid.key(a["Q_HISTORY"]),
            K_HISTORY=grid.key(a["K_HISTORY"]),
            A_HISTORY=grid.per_head(a["A_HISTORY"]),
            DELTA_HISTORY=grid.columns(a["DELTA_HISTORY"]),
            STATE_HISTORY=grid.columns(a["STATE_HISTORY"], True),
            Y_HISTORY=grid.columns(a["Y_HISTORY"]),
            STATE=grid.initial(a["FINAL_STATE"]),
        )
        actual, canonical, _ = grid.evidence(data, captured)
        for n in a:
            self.assertEqual(a[n].tobytes(), actual[n].tobytes())
        captured["Q_CANON"].reshape(-1)[-1] ^= 1
        with self.assertRaises(ValueError):
            grid.evidence(data, captured)

    def test_changed_budget_rejected(self):
        with self.assertRaises(ValueError):
            profile(4)


if __name__ == "__main__":
    unittest.main()
