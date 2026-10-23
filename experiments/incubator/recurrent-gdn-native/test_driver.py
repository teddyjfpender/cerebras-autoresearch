"""SDK-side transport and lifecycle tests without launching a simulator."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
import recurrent_gdn_native as driver  # noqa: E402
import recurrent_gdn_native_oracle as oracle  # noqa: E402
from recurrent_gdn_native_layout import profile  # noqa: E402


class RecurrentDriverTests(unittest.TestCase):
    def test_full_mixed_precision_capture_preserves_every_word(self):
        for t in (4, 8):
            grid = profile(t)
            k, v = 128, grid.tile_v
            data = grid.inputs(oracle.inputs(t, 42, "signed"))
            shapes, counts = driver.capture_contract(k, v, t, data)
            captured, expected = {}, {}
            for name, (shape, dtype) in shapes.items():
                size = grid.pes * int(np.prod(shape))
                dtype = np.dtype(dtype)
                if dtype == np.float32:
                    bits = np.arange(size, dtype=np.uint32) + 0x3F800000
                    bits[:4] = [0, 0x80000000, 0x7FC01234, 0x7F800000]
                    value = bits.view(np.float32)
                else:
                    value = np.arange(size, dtype=np.uint16)
                    value[:4] = [0, 0x8000, 0x7FC0, 0x7F80]
                expected[name] = value.reshape(grid.height, grid.width, *shape)
                captured[name] = value.view(np.uint16).view(np.float16)
                self.assertEqual(counts[name], int(np.prod(shape)) * dtype.itemsize // 2)
            result = driver.decode_capture(captured, shapes, grid)
            for name, value in result.items():
                self.assertEqual(value.dtype, expected[name].dtype)
                self.assertEqual(value.tobytes(), expected[name].tobytes())

    def test_timer_transport_and_48_bit_wrap(self):
        from common import timing

        start, end = 2**48 - 4, 7
        words = np.array(
            [
                [
                    start & 65535,
                    (start >> 16) & 65535,
                    start >> 32,
                    end & 65535,
                    (end >> 16) & 65535,
                    end >> 32,
                ]
            ]
            * 48,
            np.uint32,
        )
        self.assertEqual(timing(driver.pack_timer(words, profile(4)), 1)["max_local_cycles"], 11)
        words[0, 0] = 65536
        with self.assertRaises(ValueError):
            driver.pack_timer(words, profile(4))

    def run_main(self, folder, t=4):
        path = Path(folder)
        (path / "out.json").write_text(json.dumps(dict(params=dict(K=128, V=8, T=t, W=8, H=6))))
        before = Path.cwd()
        try:
            with patch.object(
                sys, "argv", ["gdn.py", "--name", str(path), "--case", "signed", "--seed", "7"]
            ):
                driver.main()
        finally:
            os.chdir(before)

    def test_shutdown_fault_rejects_before_core_or_pass(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(driver, "get_platform"),
            patch.object(driver, "SdkRuntime") as runtime,
            patch.object(driver, "read_core_grid_arrays") as capture,
            patch.object(driver, "save_case") as save,
        ):
            runtime.return_value.stop.side_effect = RuntimeError("shutdown fault")
            with self.assertRaisesRegex(RuntimeError, "shutdown fault"):
                self.run_main(folder)
            capture.assert_not_called()
            save.assert_not_called()
            self.assertEqual(runtime.return_value.memcpy_d2h.call_args.args[4:7], (8, 6, 6))

    def test_stale_core_and_changed_tokens_reject_before_execution(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(driver, "SdkRuntime") as runtime:
            (Path(folder) / "out.core").touch()
            with self.assertRaisesRegex(RuntimeError, "stale core"):
                self.run_main(folder)
            runtime.assert_not_called()
        with tempfile.TemporaryDirectory() as folder, patch.object(driver, "SdkRuntime") as runtime:
            with self.assertRaises(ValueError):
                self.run_main(folder, t=3)
            runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
