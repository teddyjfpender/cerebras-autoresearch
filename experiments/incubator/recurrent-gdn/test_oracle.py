"""Host-only state/oracle falsification; no SDK execution or cycle claim."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
sys.path.insert(0, str(Path(__file__).parent))
from common import CASES  # noqa: E402
from oracle import P, R, T, evidence, inputs, kv, normalized, outer, validated  # noqa: E402
from resident_swiglu_bf16_oracle import decode, encode  # noqa: E402


def simulation(data, k, v):
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
        state = (state * a[:, :, frame, :, None, None]).astype(np.float32)
        memory = np.zeros((P, P, R, v), np.float32)
        for i in range(k):
            memory = (memory + key[:, :, frame, None, i, None] * state[..., i, :]).astype(
                np.float32
            )
        delta = ((values[:, :, frame] - memory) * data["BETA_CANON"][:, :, frame, :, None]).astype(
            np.float32
        )
        state = (state + outer(key[:, :, frame], delta)).astype(np.float32)
        result = np.zeros_like(memory)
        for i in range(k):
            result = (result + q[:, :, frame, None, i, None] * state[..., i, :]).astype(np.float32)
        outputs.append(encode(result))
        states.append(state.copy())
        deltas.append(delta.copy())
    return dict(
        Q_HISTORY=q,
        K_HISTORY=key,
        A_HISTORY=a,
        DELTA_HISTORY=np.stack(deltas, axis=2),
        STATE_HISTORY=np.stack(states, axis=2),
        Y_HISTORY=np.stack(outputs, axis=2),
        FINAL_STATE=state.copy(),
    )


def fails(k, v, data, actual, canonical=None):
    arrays, error, bound = evidence(k, v, data, actual, canonical or data)
    return not bool(arrays["exact_state_match"]) or not np.all(error <= bound)


class RecurrentOracleTests(unittest.TestCase):
    def test_all_families_both_profiles(self):
        for k in (16, 32):
            for case in CASES:
                with self.subTest(k=k, case=case):
                    data = inputs(k, 16, 123, case)
                    self.assertFalse(fails(k, 16, data, simulation(data, k, 16)))

    def test_independent_transition_matrix_matches_recurrence(self):
        for case in CASES:
            data = validated(inputs(16, 16, 99, case), 16, 16)
            q, key, a = normalized(data, 16)
            transition_state = data["S0"].copy()
            state = transition_state.copy()
            for frame in range(T):
                k = key[:, :, frame]
                beta = data["BETA_CANON"][:, :, frame]
                value = data["V_CANON"][:, :, frame]
                d = a[:, :, frame, :, None, None] * state
                delta = beta[..., None] * (value - kv(k, d))
                state = d + outer(k, delta)
                transition = (
                    np.eye(16)
                    - beta[..., None, None] * k[:, :, None, :, None] * k[:, :, None, None, :]
                )
                transition *= a[:, :, frame, :, None, None]
                transition_state = transition @ transition_state + outer(k, beta[..., None] * value)
                np.testing.assert_allclose(state, transition_state, rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(
                    kv(q[:, :, frame], state), kv(q[:, :, frame], transition_state), atol=1e-14
                )

    def test_every_state_frame_and_intermediate_is_checked(self):
        data = inputs(32, 16, 321, "signed")
        reference = simulation(data, 32, 16)
        for name in ("Q_HISTORY", "K_HISTORY", "A_HISTORY", "DELTA_HISTORY", "STATE_HISTORY"):
            for frame in range(T):
                with self.subTest(name=name, frame=frame):
                    changed = copy.deepcopy(reference)
                    changed[name][0, 0, frame].reshape(-1)[0] += np.float32(0.1)
                    self.assertTrue(fails(32, 16, data, changed))

    def test_bf16_output_and_fp32_state_precision(self):
        data = inputs(16, 16, 312, "signed")
        reference = simulation(data, 16, 16)
        for frame in range(T):
            changed = copy.deepcopy(reference)
            changed["Y_HISTORY"][0, 0, frame, 0, 0] = encode(np.array([2], np.float32))[0]
            self.assertTrue(fails(16, 16, data, changed))
        changed = copy.deepcopy(reference)
        changed["STATE_HISTORY"] = decode(encode(changed["STATE_HISTORY"]))
        self.assertTrue(fails(16, 16, data, changed))

    def test_immutable_words_including_signed_zero(self):
        data = inputs(16, 16, 41, "zero")
        actual = simulation(data, 16, 16)
        for name in data:
            changed = copy.deepcopy(data)
            if changed[name].dtype == np.uint16:
                changed[name].reshape(-1)[0] ^= 0x8000
            else:
                changed[name].view(np.uint32).reshape(-1)[0] ^= 0x80000000
            self.assertTrue(fails(16, 16, data, actual, changed))

    def test_nonzero_initial_state_and_zero_queries(self):
        data = inputs(16, 16, 41, "zero")
        actual = simulation(data, 16, 16)
        self.assertTrue(np.any(data["S0"] != 0))
        self.assertTrue(np.all(decode(actual["Y_HISTORY"]) == 0))
        self.assertTrue(np.any(actual["STATE_HISTORY"][:, :, -1] != data["S0"]))
        changed = copy.deepcopy(actual)
        changed["STATE_HISTORY"][:] = 0
        self.assertTrue(fails(16, 16, data, changed))

    def test_missing_wrong_dtype_nonfinite(self):
        data = inputs(16, 16, 73, "signed")
        actual = simulation(data, 16, 16)
        for name in actual:
            changed = copy.deepcopy(actual)
            changed.pop(name)
            with self.assertRaises(ValueError):
                evidence(16, 16, data, changed, data)
            changed = copy.deepcopy(actual)
            changed[name] = changed[name][:, :, :-1]
            with self.assertRaises(ValueError):
                evidence(16, 16, data, changed, data)
        changed = copy.deepcopy(actual)
        changed["STATE_HISTORY"] = changed["STATE_HISTORY"].astype(np.float16)
        with self.assertRaises(ValueError):
            evidence(16, 16, data, changed, data)
        changed = copy.deepcopy(actual)
        changed["Q_HISTORY"].reshape(-1)[0] = np.inf
        with self.assertRaises(ValueError):
            evidence(16, 16, data, changed, data)
        changed = copy.deepcopy(actual)
        changed["Y_HISTORY"].reshape(-1)[0] = 0x7F80
        with self.assertRaises(ValueError):
            evidence(16, 16, data, changed, data)


if __name__ == "__main__":
    unittest.main()
