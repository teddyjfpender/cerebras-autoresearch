"""Native-width oracle falsification; no device measurement."""

import copy
import unittest

import numpy as np
from common import CASES
from oracle import R, evidence, inputs, normalized, validated
from recurrent_gdn_oracle import kv, outer
from resident_swiglu_bf16_oracle import decode, encode


def simulation(data, tokens):
    k, v, T = 128, 128, tokens
    q, key, values = (decode(data[name]) for name in ("Q_CANON", "K_CANON", "V_CANON"))
    qsum = np.zeros(q.shape[:-1], np.float32)
    ksum = np.zeros_like(qsum)
    for i in range(k):
        qsum = (qsum + q[..., i] * q[..., i]).astype(np.float32)
        ksum = (ksum + key[..., i] * key[..., i]).astype(np.float32)
    q = (q / np.sqrt(qsum + np.float32(1e-6))[..., None]).astype(np.float32)
    q = (q / np.sqrt(np.float32(k))).astype(np.float32)
    key = (key / np.sqrt(ksum + np.float32(1e-6))[..., None]).astype(np.float32)
    a = np.exp(data["G_CANON"]).astype(np.float32)
    state = data["S0"].copy()
    states, deltas, outputs = [], [], []
    for frame in range(T):
        state = (state * a[frame, :, None, None]).astype(np.float32)
        memory = np.zeros((R, v), np.float32)
        for i in range(k):
            memory = (memory + key[frame, None, i, None] * state[..., i, :]).astype(np.float32)
        delta = ((values[frame] - memory) * data["BETA_CANON"][frame, :, None]).astype(np.float32)
        state = (state + outer(key[frame], delta)).astype(np.float32)
        result = np.zeros_like(memory)
        for i in range(k):
            result = (result + q[frame, None, i, None] * state[..., i, :]).astype(np.float32)
        outputs.append(encode(result))
        states.append(state.copy())
        deltas.append(delta.copy())
    return dict(
        Q_HISTORY=q,
        K_HISTORY=key,
        A_HISTORY=a,
        DELTA_HISTORY=np.stack(deltas, axis=0),
        STATE_HISTORY=np.stack(states, axis=0),
        Y_HISTORY=np.stack(outputs, axis=0),
        FINAL_STATE=state.copy(),
    )


class NativeOracleTests(unittest.TestCase):
    def test_every_family_and_token_budget(self):
        for t in (4, 8):
            for case in CASES:
                with self.subTest(tokens=t, case=case):
                    data = inputs(t, 20261038, case)
                    actual = simulation(data, t)
                    arrays, error, bound = evidence(t, data, actual, data)
                    self.assertTrue(arrays["exact_state_match"])
                    self.assertTrue(np.all(error <= bound), float((error / bound).max()))
                    self.assertEqual(len(arrays["check_labels"]), 3 + 5 * t)
                    self.assertEqual(arrays["check_offsets"][-1], error.size)

    def test_transition_matrix_independent_identity(self):
        for t in (4, 8):
            data = validated(inputs(t, 91, "mixed_scale"), t)
            q, key, a = normalized(data, 128)
            direct = data["S0"].copy()
            matrix = direct.copy()
            for frame in range(t):
                k = key[frame]
                b = data["BETA_CANON"][frame]
                v = data["V_CANON"][frame]
                d = a[frame, :, None, None] * direct
                delta = b[:, None] * (v - kv(k, d))
                direct = d + outer(k, delta)
                transition = a[frame, :, None, None] * (
                    np.eye(128) - b[:, None, None] * k[None, :, None] * k[None, None, :]
                )
                matrix = np.matmul(transition, matrix) + outer(k, b[:, None] * v)
                np.testing.assert_allclose(direct, matrix, rtol=3e-12, atol=3e-12)

    def test_bf16_state_or_omitted_prefix_rejected(self):
        data = inputs(4, 8, "signed")
        actual = simulation(data, 4)
        changed = copy.deepcopy(actual)
        changed["STATE_HISTORY"] = decode(encode(changed["STATE_HISTORY"]))
        changed["FINAL_STATE"] = changed["STATE_HISTORY"][-1].copy()
        arrays, error, bound = evidence(4, data, changed, data)
        self.assertTrue(np.any(error > bound))
        changed = copy.deepcopy(actual)
        changed["STATE_HISTORY"] = changed["STATE_HISTORY"][:-1]
        with self.assertRaises(ValueError):
            evidence(4, data, changed, data)
        changed = copy.deepcopy(actual)
        changed["FINAL_STATE"][0, 0, 0] += 1
        arrays, _, _ = evidence(4, data, changed, data)
        self.assertFalse(arrays["exact_state_match"])

    def test_all_immutable_bits_and_fixed_dimensions(self):
        data = inputs(4, 8, "zero")
        actual = simulation(data, 4)
        changed = copy.deepcopy(data)
        changed["Q_CANON"].reshape(-1)[1] = 0
        arrays, _, _ = evidence(4, data, actual, changed)
        self.assertFalse(arrays["exact_state_match"])
        with self.assertRaises(ValueError):
            inputs(3, 8, "signed")
        changed = copy.deepcopy(data)
        changed["S0"] = changed["S0"].astype(np.float64)
        with self.assertRaises(ValueError):
            evidence(4, changed, actual, changed)


if __name__ == "__main__":
    unittest.main()
