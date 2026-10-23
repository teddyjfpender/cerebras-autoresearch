"""SDK snapshot transport gates; no performance qualification is implied."""

import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
from snapshot import CoreSnapshot, program_symbols  # noqa: E402


def core_fixture(width=2, height=3, pe_bytes=32):
    identity = b"\x7fELF\x02\x01\x01" + bytes(9)
    header = struct.pack("<16sHHIQQQIHHHHHH", identity, 4, 248, 1, 0, 64, 0, 2, 64, 56, 2, 64, 0, 0)
    size = width * height * pe_bytes
    note_segment = struct.pack("<IIQQQQQQ", 4, 0, 176, 0, 0, 32, 0, 0)
    note = struct.pack("<III", 9, 8, 1) + b"Cerebras\0\0\0\0" + struct.pack("<II", width, height)
    segment = struct.pack("<IIQQQQQQ", 1, 7, 208, 0, height << 20 | width, size, pe_bytes, 0)
    words = np.arange(size // 2, dtype=np.uint16) ^ np.uint16(0x8000)
    return header + note_segment + segment + note + words.tobytes()


class SnapshotTests(unittest.TestCase):
    def snapshot(self, data):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "capture.core"
        path.write_bytes(data)
        return CoreSnapshot(path)

    def test_every_pe_and_last_word_preserve_raw_half_bits(self):
        memory = self.snapshot(core_fixture())
        for y in range(3):
            for x in range(2):
                actual = memory.half_bits(x, y, 0, 32)
                start = (y * 2 + x) * 16
                expected = np.arange(start, start + 16, dtype=np.uint16) ^ np.uint16(0x8000)
                np.testing.assert_array_equal(actual, expected)
        self.assertEqual(memory.half_bits(0, 0, 0, 2)[0], 0x8000)
        self.assertEqual(memory.half_bits(1, 2, 30, 2)[0], 0x805F)

    def test_truncated_sram_is_rejected(self):
        with self.assertRaises((ValueError, RuntimeError)):
            self.snapshot(core_fixture()[:-2])

    def test_incorrect_grid_extent_is_rejected(self):
        data = bytearray(core_fixture())
        struct.pack_into("<Q", data, 64 + 56 + 32, 4 << 20 | 2)
        with self.assertRaises((ValueError, RuntimeError)):
            self.snapshot(data)

    def test_unaligned_or_outside_pe_arrays_are_rejected(self):
        memory = self.snapshot(core_fixture())
        for arguments in ((2, 0, 0, 2), (0, 3, 0, 2), (0, 0, 1, 2), (0, 0, 0, 3), (0, 0, 30, 4)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                memory.half_bits(*arguments)

    def symbols(self, entries, expected):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "program.elf"
        path.touch()
        with patch("snapshot.ELFFile") as elf:
            table = elf.return_value.get_section_by_name.return_value
            table.iter_symbols.return_value = [
                SimpleNamespace(
                    name=name,
                    entry=SimpleNamespace(
                        st_info=SimpleNamespace(type="STT_OBJECT"), st_value=address, st_size=size
                    ),
                )
                for name, address, size in entries
            ]
            return program_symbols(path, expected)

    def test_exact_compiler_alias_and_size_are_required(self):
        entries = [("$$csl_base_address$$7$$W_HISTORY_0", 128, 2720)]
        self.assertEqual(self.symbols(entries, {"W_HISTORY_0": 1360}), {"W_HISTORY_0": (128, 2720)})
        with self.assertRaises(ValueError):
            self.symbols(entries, {"W_HISTORY_0": 1361})
        with self.assertRaises(ValueError):
            self.symbols(entries, {"W_HISTORY_1": 1360})

    def test_conflicting_addresses_cannot_silently_select_a_frame(self):
        entries = [("W_HISTORY_0", 128, 32), ("$$csl_base_address$$7$$W_HISTORY_0", 160, 32)]
        with self.assertRaises(ValueError):
            self.symbols(entries, {"W_HISTORY_0": 16})


if __name__ == "__main__":
    unittest.main()
