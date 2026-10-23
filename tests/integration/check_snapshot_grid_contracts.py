"""Rectangular/core-sharing support preserves exact words and complete footprints."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
import numpy as np
import snapshot_grid as grid


class GridSnapshotTests(unittest.TestCase):
    def fixture(self, folder, coordinates):
        program = Path(folder) / "program.elf"
        program.touch()
        core = MagicMock()
        core.memory.get_flags.return_value = 2
        core.half_bits.side_effect = lambda x, y, address, size: np.array([x, y], np.uint16)
        elf = MagicMock()
        elf.get_flags.return_value = 2
        elf.iter_coordinates.return_value = coordinates
        return core, elf

    def test_one_elf_can_cover_multiple_physical_compute_pes(self):
        with tempfile.TemporaryDirectory() as folder:
            core, elf = self.fixture(folder, [(4, 1), (5, 1)])
            with (
                patch.object(grid, "CoreSnapshot", return_value=core),
                patch.object(grid, "ELFMemory", return_value=elf),
                patch.object(grid, "program_symbols", return_value={"A": (32, 4)}),
            ):
                arrays = grid.read_core_grid_arrays("fixture.core", folder, 2, 1, {"A": 2})
            self.assertEqual(arrays["A"].dtype, np.uint16)
            np.testing.assert_array_equal(arrays["A"], [4, 1, 5, 1])
            self.assertEqual(core.half_bits.call_count, 2)

    def test_duplicate_missing_empty_and_outside_footprints_reject(self):
        for coordinates in ([(4, 1), (4, 1)], [(4, 1)], [], [(4, 1), (6, 1)]):
            with self.subTest(coordinates=coordinates), tempfile.TemporaryDirectory() as folder:
                core, elf = self.fixture(folder, coordinates)
                with (
                    patch.object(grid, "CoreSnapshot", return_value=core),
                    patch.object(grid, "ELFMemory", return_value=elf),
                    patch.object(grid, "program_symbols", return_value={"A": (32, 4)}),
                ):
                    with self.assertRaises(ValueError):
                        grid.read_core_grid_arrays("fixture.core", folder, 2, 1, {"A": 2})

    def test_architecture_mismatch_rejects_before_read(self):
        with tempfile.TemporaryDirectory() as folder:
            core, elf = self.fixture(folder, [(4, 1), (5, 1)])
            elf.get_flags.return_value = 3
            with (
                patch.object(grid, "CoreSnapshot", return_value=core),
                patch.object(grid, "ELFMemory", return_value=elf),
                patch.object(grid, "program_symbols", return_value={"A": (32, 4)}),
            ):
                with self.assertRaisesRegex(ValueError, "architectures"):
                    grid.read_core_grid_arrays("fixture.core", folder, 2, 1, {"A": 2})
            core.half_bits.assert_not_called()

    def test_empty_and_invalid_contracts_reject(self):
        for width, height, expected in (
            (0, 1, {"A": 2}),
            (1, 0, {"A": 2}),
            (1, 1, {}),
            (1, 1, {"A": 0}),
        ):
            with (
                self.subTest(width=width, height=height, expected=expected),
                self.assertRaises(ValueError),
            ):
                grid.read_core_grid_arrays("fixture.core", "missing", width, height, expected)


if __name__ == "__main__":
    unittest.main()
