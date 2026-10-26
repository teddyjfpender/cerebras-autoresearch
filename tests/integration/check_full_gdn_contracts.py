import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
sys.path.insert(0, str(Path(__file__).parent))

import copy
import unittest

import numpy as np
from check_native_gdn_contracts import simulation as single_simulation
from recurrent_gdn_full_oracle import STAGES, G, evidence, inputs


def simulation(data, tokens):
    groups = [single_simulation({n: a[g] for n, a in data.items()}, tokens) for g in range(G)]
    return {n: np.stack([a[n] for a in groups]) for n in STAGES}


class FullOracleTests(unittest.TestCase):
    def test_all_groups_and_both_full_states(self):
        data = inputs(2, 125, "signed")
        actual = simulation(data, 2)
        arrays, error, bound = evidence(2, data, actual, data)
        self.assertTrue(arrays["exact_state_match"])
        self.assertTrue(np.all(error <= bound))
        self.assertEqual(len(arrays["check_labels"]), 208)
        self.assertEqual(arrays["check_offsets"][-1], error.size)

    def test_missing_or_corrupted_last_group_state_rejected(self):
        data = inputs(2, 125, "signed")
        actual = simulation(data, 2)
        missing = {n: a[:-1] for n, a in actual.items()}
        with self.assertRaises(ValueError):
            evidence(2, data, missing, data)
        actual["STATE_HISTORY"][-1, 0, -1, -1, -1] += 0.5
        _, error, bound = evidence(2, data, actual, data)
        self.assertTrue(np.any(error > bound))

    def test_fixed_token_count_and_immutable_last_group(self):
        with self.assertRaises(ValueError):
            inputs(4, 125, "signed")
        data = inputs(2, 125, "signed")
        actual = simulation(data, 2)
        canonical = copy.deepcopy(data)
        canonical["S0"].view(np.uint32).reshape(-1)[-1] ^= 1
        arrays, _, _ = evidence(2, data, actual, canonical)
        self.assertFalse(arrays["exact_state_match"])


if __name__ == "__main__":
    unittest.main()
