"""NumPy-only attention properties; neither compiles CSL nor measures WSE cycles."""

import copy
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/drivers"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from common import CASES, save_case, timing
from decode_attention_oracle import evidence, inputs, reference

from cerebras_research.catalog import get_suite
from tools.replay_numerics import check_case


def polynomial_exp(x):
    """Float32, unfused arithmetic mirror for stressing approximation error only."""
    x = np.asarray(x, dtype=np.float32)
    # Device branches skip the polynomial outside this interval. Avoid evaluating
    # its unused host-side branch on extreme masked scores.
    r = np.clip(x, -16, 0) * np.float32(0.0625)
    v = np.full_like(x, 1 / 3628800)
    for coefficient in (
        1 / 362880,
        1 / 40320,
        1 / 5040,
        1 / 720,
        1 / 120,
        1 / 24,
        1 / 6,
        1 / 2,
        1,
        1,
    ):
        v = v * r + np.float32(coefficient)
    for _ in range(4):
        v = v * v
    return np.where((x >= -16) & (x <= 0), v, np.exp(x))


def newton_reciprocal(x):
    """Unfused FP32 model of the positive-denominator reciprocal candidate."""
    x = np.asarray(x, dtype=np.float32)
    if not np.all(np.isfinite(x) & (x >= 1) & (x <= 8192)):
        raise ValueError("Outside the stable-softmax reciprocal domain")
    y = (np.uint32(0x7F000000) - x.view(np.uint32)).view(np.float32)
    for _ in range(4):
        y = y * (np.float32(2) - x * y)
    return y


def distributed_model(dims, data, exp=np.exp, reciprocal=None):
    """CPU model of the sharding/reduction scheme, separate from dense oracle."""
    p, s, d, g, t = (dims[k] for k in ("P", "S", "D", "G", "T"))
    keys, values = data["K"].copy(), data["V"].copy()
    result = np.zeros((t, g, d), np.float32)
    for step in range(t):
        position = dims["L"] + step
        keys[position // s, position % s] = data["NEW_K"][step]
        values[position // s, position % s] = data["NEW_V"][step]
        logits, maxima = [], []
        for shard in range(p):
            valid = min(s, max(0, position - shard * s + 1))
            score = (data["Q"][step] @ keys[shard, :valid].T) / np.float32(np.sqrt(d))
            logits.append(score)
            maxima.append(score.max(axis=1) if valid else np.full(g, -3.4e38, np.float32))
        maximum = np.max(maxima, axis=0)
        denominator = np.zeros(g, np.float32)
        numerator = np.zeros((g, d), np.float32)
        for shard, score in enumerate(logits):
            weights = exp(score - maximum[:, None])
            denominator += weights.sum(axis=1)
            numerator += weights @ values[shard, : score.shape[1]]
        result[step] = (
            numerator / denominator[:, None]
            if reciprocal is None
            else numerator * reciprocal(denominator)[:, None]
        )
    return dict(Y=result, K=keys, V=values)


class AttentionProperties(unittest.TestCase):
    def setUp(self):
        self.dims = get_suite("decode-attention")[0]["dimensions"]

    def test_finer_sharding_preserves_the_global_workload(self):
        profiles = [p for p in get_suite("decode-attention") if "ctx128" in p["id"]]
        base = inputs(profiles[0]["dimensions"], 19, "signed")
        expected = reference(profiles[0]["dimensions"], base)["Y"]
        for profile in profiles[1:]:
            data = inputs(profile["dimensions"], 19, "signed")
            for name in base:
                np.testing.assert_array_equal(data[name].ravel(), base[name].ravel())
            np.testing.assert_array_equal(reference(profile["dimensions"], data)["Y"], expected)

    def test_all_shapes_cases_and_fresh_seeds(self):
        for profile in get_suite("decode-attention"):
            dims = profile["dimensions"]
            for seed in (7, 808):
                for case in CASES:
                    data = inputs(dims, seed, case)
                    for exp in (np.exp, polynomial_exp):
                        with self.subTest(
                            profile=profile["id"], seed=seed, case=case, exp=exp.__name__
                        ):
                            actual = distributed_model(dims, data, exp)
                            arrays, error, bound = evidence(dims, data, actual)
                            self.assertTrue(bool(arrays["exact_state_match"]))
                            self.assertTrue(np.all(error <= bound), float(np.max(error / bound)))

    def test_every_normalized_fp32_mantissa_has_a_bounded_reciprocal_error(self):
        values = np.arange(0x3F800000, 0x40000000, dtype=np.uint32).view(np.float32)
        relative = np.abs(newton_reciprocal(values).astype(np.float64) * values - 1)
        self.assertLess(float(relative.max()), 3e-7)
        # Exact powers-of-two scaling covers the supported denominator range.
        boundaries = np.array([1, 2, 127, 128, 4096, 8192], np.float32)
        relative = np.abs(newton_reciprocal(boundaries).astype(np.float64) * boundaries - 1)
        self.assertLess(float(relative.max()), 3e-7)

    def test_newton_normalization_preserves_the_complete_attention_contract(self):
        for profile in get_suite("decode-attention"):
            dims = profile["dimensions"]
            for seed in (7, 808):
                for case in CASES:
                    data = inputs(dims, seed, case)
                    actual = distributed_model(dims, data, reciprocal=newton_reciprocal)
                    arrays, error, bound = evidence(dims, data, actual)
                    with self.subTest(profile=profile["id"], seed=seed, case=case):
                        self.assertTrue(bool(arrays["exact_state_match"]))
                        self.assertTrue(np.all(error <= bound), float(np.max(error / bound)))

    def test_early_output_and_final_cache_defects_fail(self):
        data = inputs(self.dims, 7, "signed")
        actual = {k: v.astype(np.float32) for k, v in reference(self.dims, data).items()}
        for defect in ("early_output", "updated_slot", "untouched_slot"):
            bad = copy.deepcopy(actual)
            if defect == "early_output":
                bad["Y"][0, 0, 0] += 1
            elif defect == "updated_slot":
                position = self.dims["L"]
                bad["K"][position // self.dims["S"], position % self.dims["S"], 0] += 1
            else:
                bad["V"][0, 0, 0] += 1
            arrays, error, bound = evidence(self.dims, data, bad)
            self.assertFalse(np.all(error <= bound) and bool(arrays["exact_state_match"]))

    def test_future_slots_do_not_affect_earlier_outputs(self):
        dims = get_suite("decode-attention")[1]["dimensions"]
        data = inputs(dims, 19, "signed")
        changed = copy.deepcopy(data)
        for name in ("K", "V"):
            changed[name].reshape(-1, dims["D"])[dims["L"] + dims["T"] :] = 1e20
        np.testing.assert_array_equal(reference(dims, data)["Y"], reference(dims, changed)["Y"])

    def test_zero_queries_require_nonzero_attention_work(self):
        data = inputs(self.dims, 11, "zero")
        expected = reference(self.dims, data)
        self.assertGreater(float(np.abs(expected["Y"]).max()), 0)

    def test_cache_signed_zero_is_checked_by_bits(self):
        data = inputs(self.dims, 7, "signed")
        data["K"][0, 0, 0] = -0.0
        actual = {k: v.astype(np.float32) for k, v in reference(self.dims, data).items()}
        actual["K"][0, 0, 0] = 0.0
        arrays, _, _ = evidence(self.dims, data, actual)
        self.assertFalse(bool(arrays["exact_state_match"]))

    def test_polynomial_interval_and_fallback_boundaries(self):
        values = np.linspace(-16, 0, 100001, dtype=np.float32)
        expected = np.exp(values.astype(np.float64))
        relative = np.abs(polynomial_exp(values) / expected - 1)
        self.assertLess(float(relative.max()), 5e-6)
        outside = np.array(
            [
                -100,
                -17,
                np.nextafter(np.float32(-16), np.float32(-17)),
                np.nextafter(np.float32(0), np.float32(1)),
                0.1,
                1,
            ],
            np.float32,
        )
        np.testing.assert_array_equal(polynomial_exp(outside), np.exp(outside))

    def test_coordinator_clock_is_not_the_largest_local_interval(self):
        # Packed 48-bit timestamps: local intervals 100 and 40, coordinator PE1.
        raw = np.array([[0, 100 << 16, 0], [0, 40 << 16, 0]], np.uint32)
        measured = timing(raw, 1, coordinator=1)
        self.assertEqual(measured["max_local_cycles"], 100)
        self.assertEqual(measured["coordinator_cycles"], 40)
        with self.assertRaises(ValueError):
            timing(raw, 1, coordinator=2)

    def test_synthetic_raw_evidence_replays_coordinator_and_state(self):
        # A temporary gate fixture: these clocks are not device measurements.
        profile = get_suite("decode-attention")[0]
        data = inputs(profile["dimensions"], 7, "signed")
        actual = distributed_model(profile["dimensions"], data)
        arrays, error, bound = evidence(profile["dimensions"], data, actual)
        raw = np.array([[0, n << 16, 0] for n in (100, 80, 70, 40)], np.uint32)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            report = save_case(folder, "signed", arrays, error, bound, raw, 1, 7, coordinator=3)
            checked = check_case(profile, report, Path(folder) / report["artifact"])
            self.assertEqual(checked["metrics"]["coordinator_cycles"], 40)


if __name__ == "__main__":
    unittest.main()
