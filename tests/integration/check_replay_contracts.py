"""Raw replay must detect errors even when accompanying hashes are recomputed."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
from common import timing
from resident_oracle import evidence, inputs, reference
from reusable_gemm_oracle import evidence as sequence_evidence
from reusable_gemm_oracle import pack_inputs

from cerebras_research.paths import digest
from tools.replay_numerics import check_case, check_summary


def packed(start=100, end=200):
    return np.array(
        [start & 0xFFFFFFFF, (start >> 32) | ((end & 65535) << 16), end >> 16], np.uint32
    )


class ReplayTests(unittest.TestCase):
    def linear(self):
        x = np.array([[1, 2, 3, 4]], np.float16)
        w = (np.arange(16).reshape(4, 4) / 16).astype(np.float16)
        expected = x.astype(np.float64) @ w.astype(np.float64)
        bound = (5 / 2043) * (
            np.abs(x.astype(np.float64)) @ np.abs(w.astype(np.float64))
        ) + 8 * 2.0**-24
        profile = dict(
            adapter="meshgemv",
            dimensions=dict(M=4, N=4),
            pe_grid=[4, 4],
            timing_protocol=dict(repeats=10),
        )
        arrays = dict(
            x=x,
            w=w,
            expected=expected,
            actual=np.tile(expected, (4, 1)),
            error_bound=bound,
            timer_u32=np.tile(packed(), (16, 1)),
        )
        return profile, arrays

    def check(self, profile, arrays, mutate=None, metrics=None, summary=None):
        if mutate:
            mutate(arrays)
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder) / "case.npz"
            np.savez_compressed(artifact, **arrays)
            case = dict(
                case="signed",
                artifact_sha256=digest(artifact),
                metrics=metrics
                or timing(arrays["timer_u32"], profile["timing_protocol"]["repeats"]),
            )
            error = np.abs(arrays["actual"] - arrays["expected"])
            case.update(
                max_abs_error=float(error.max()),
                max_fraction_of_error_bound=float((error / arrays["error_bound"]).max()),
            )
            case.update(summary or {})
            return check_case(profile, case, artifact)

    def test_quantized_linear_evidence_replays(self):
        result = self.check(*self.linear())
        self.assertEqual(result["max_abs_error_recomputed"], 0)
        self.assertEqual(result["metrics"]["max_local_cycles"], 10)

    def test_forged_expected_bound_or_actual_is_rejected_with_valid_digest(self):
        for name, change in (("expected", 1), ("error_bound", 100), ("actual", 1)):
            with self.subTest(name=name), self.assertRaises((AssertionError, ValueError)):

                def corrupt(arrays):
                    if name == "error_bound":
                        arrays[name] *= change
                    else:
                        arrays[name] += change

                self.check(*self.linear(), mutate=corrupt)

    def test_reported_timestamps_must_match_raw_integer_words(self):
        with self.assertRaisesRegex(ValueError, "timestamps"):
            self.check(
                *self.linear(),
                metrics=dict(max_local_cycles=11, mean_local_cycles=10, pe_cycle_spread=0),
            )

    def test_declared_input_precision_is_checked(self):
        with self.assertRaisesRegex(ValueError, "FP16"):
            self.check(
                *self.linear(),
                mutate=lambda arrays: arrays.update(x=arrays["x"].astype(np.float64)),
            )

    def test_decoder_early_cache_corruption_is_not_hidden_by_saved_success(self):
        dims = dict(H=4, F=4, T=4, C=6, L=1, A=1)
        profile = dict(
            adapter="decoder",
            dimensions=dims,
            oracle_contract="resident_transformer_fp32_v1",
            timing_protocol=dict(repeats=1),
        )
        data = inputs("decoder", dims, 41, "signed")
        actual = {
            key: value.astype(np.float32) for key, value in reference("decoder", dims, data).items()
        }
        arrays, _, bound = evidence("decoder", dims, data, actual)
        arrays.update(error_bound=bound, timer_u32=packed())
        self.check(profile, arrays)
        arrays["actual_HISTORY"][0, 0, 0, 0] += 0.1
        arrays["actual"] = np.concatenate(
            [
                arrays["actual_Y"].ravel(),
                arrays["actual_HISTORY"].ravel(),
                arrays["actual_CACHE"].ravel(),
            ]
        ).astype(np.float64)
        # The artifact is rehashed and its saved success flag remains true.
        self.assertTrue(bool(arrays["exact_state_match"]))
        with self.assertRaisesRegex(ValueError, "state"):
            self.check(profile, arrays)

    def test_replay_rejects_two_frames_claimed_as_a_four_frame_workload(self):
        dims = dict(H=4, F=4, T=4)
        short = dict(dims, T=2)
        profile = dict(
            adapter="gated_mlp",
            dimensions=dims,
            oracle_contract="resident_transformer_fp32_v1",
            timing_protocol=dict(repeats=1),
        )
        data = inputs("gated_mlp", short, 43, "signed")
        actual = {
            key: value.astype(np.float32)
            for key, value in reference("gated_mlp", short, data).items()
        }
        arrays, _, bound = evidence("gated_mlp", short, data, actual)
        arrays.update(error_bound=bound, timer_u32=packed())
        with self.assertRaisesRegex(ValueError, "shape"):
            self.check(profile, arrays)

    def sequence(self):
        dims = dict(M=16, K=16, N=16, T=4)
        profile = dict(
            adapter="reusable_gemm",
            dimensions=dims,
            pe_grid=[4, 4],
            oracle_contract="resident_gemm_sequence_fp16_v1",
            timing_protocol=dict(repeats=1),
        )
        x = (np.arange(4 * 16 * 16).reshape(4, 16, 16) / 1024).astype(np.float16)
        w = np.eye(16, dtype=np.float16)
        px, pw = pack_inputs(4, 4, 4, 4, x, w)
        history = np.broadcast_to(pw[:, :, None], (4, 4, 4, 16)).copy()
        arrays, _, bound = sequence_evidence(
            4, x, w, x.astype(np.float64), history, px.ravel(), pw.ravel()
        )
        arrays.update(error_bound=bound, timer_u32=np.tile(packed(), (16, 1)))
        return profile, arrays

    def test_every_resident_gemm_frame_and_state_replays(self):
        result = self.check(*self.sequence())
        self.assertEqual(result["max_abs_error_recomputed"], 0)
        self.assertTrue(result["exact_state_match"])

    def test_resident_gemm_middle_state_corruption_rejected_despite_saved_success(self):
        profile, arrays = self.sequence()
        arrays["working_w_history"][0, 0, 1, 0] += 1
        self.assertTrue(bool(arrays["exact_state_match"]))
        with self.assertRaisesRegex(ValueError, "state"):
            self.check(profile, arrays)

    def test_resident_gemm_shortened_sequence_is_rejected(self):
        profile, arrays = self.sequence()
        arrays["x"] = arrays["x"][:3]
        with self.assertRaisesRegex(ValueError, "shape"):
            self.check(profile, arrays)

    def test_forged_reported_error_is_rejected_with_valid_raw_digest(self):
        for name in ("max_abs_error", "max_fraction_of_error_bound"):
            profile, arrays = self.linear()
            arrays["actual"][0, 0] += arrays["error_bound"][0, 0] / 2
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "numerical summary"):
                self.check(profile, arrays, summary={name: 0})

    def test_derived_metric_summaries_must_match_verified_cases_and_steps(self):
        profile, _ = self.linear()
        profile["work_units"] = 32
        profile["metric"] = "max_local_cycles"
        case = dict(
            metrics=dict(max_local_cycles=10, mean_local_cycles=9, pe_cycle_spread=1),
            max_fraction_of_error_bound=0.5,
        )
        metrics = dict(
            case["metrics"],
            max_error_fraction=0.5,
            work_units_per_cycle=3.2,
            pe_cycles=160,
            experiment_wall_seconds=2.5,
        )
        run = dict(profile=profile, cases=[case], metrics=metrics, steps=[dict(wall_seconds=2.5)])
        check_summary(run)
        for name in metrics:
            run["metrics"] = dict(metrics, **{name: metrics[name] + 1})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "metric summary"):
                check_summary(run)


if __name__ == "__main__":
    unittest.main()
