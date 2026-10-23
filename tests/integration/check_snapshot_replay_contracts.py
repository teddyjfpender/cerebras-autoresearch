"""Full-core auditor rejects corrupt words and invalid receipt selections."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))

import numpy as np

from tools.replay_snapshots import arrays_contract, audit, equal_bits, unpack


class SnapshotReplayTests(unittest.TestCase):
    def test_changed_last_word_signed_zero_or_precision_rejected(self):
        original = np.zeros((4, 4, 4, 16), np.float16)
        for kind in ("last", "signed_zero", "precision", "shape"):
            actual = original.copy()
            if kind == "last":
                actual.reshape(-1)[-1] = 1
            elif kind == "signed_zero":
                actual.view(np.uint16).reshape(-1)[-1] = 0x8000
            elif kind == "precision":
                actual = actual.astype(np.float32)
            else:
                actual = actual[:-1]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                equal_bits(actual, original)

    def test_all_fp16_and_fp32_output_frames_reconstructed(self):
        for dtype in (np.float16, np.float32):
            values = (np.arange(4 * 16 * 32).reshape(4, 16, 32) / 17).astype(dtype)
            values.reshape(-1)[0] = -0.0
            packed = values.reshape(4, 4, 4, 4, 8).transpose(1, 3, 0, 4, 2).copy()
            equal_bits(unpack(packed.view(np.float16), 4, 4, 8, 4, dtype), values)

    def test_empty_wrong_schema_and_duplicate_receipts_rejected(self):
        pointer = dict(path="runs/not-a-receipt.json", sha256="0" * 64)
        for manifest in (
            dict(schema_version=1, receipts=[]),
            dict(schema_version=0, receipts=[pointer]),
            dict(schema_version=1, receipts=[pointer, pointer]),
        ):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                audit(manifest)

    def test_complete_contract_includes_both_timer_and_all_weight_frames(self):
        profile = dict(
            adapter="reusable_gemm", dimensions=dict(M=16, K=80, N=272, T=4), pe_grid=[4, 4]
        )
        _, _, _, _, expected, _, _ = arrays_contract(profile)
        self.assertEqual(expected["time_buf_f32"], 6)
        self.assertEqual(expected["ALL_OUT"], 4 * 4 * 68)
        for frame in range(4):
            self.assertEqual(expected[f"W_HISTORY_{frame}"], 20 * 68)


if __name__ == "__main__":
    unittest.main()
