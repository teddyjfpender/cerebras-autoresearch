"""Read complete simulated SRAM from an SDK ELF core after a clean stop.

Use the SDK's ELF memory reader and PE coordinate metadata. Require a complete
SRAM segment and exact-sized compiled symbols; reject incompatible layouts
rather than substituting bytes from an unloaded or neighboring region.
"""

import re
from pathlib import Path

import numpy as np
from cerebras.sdk.debug.debug_util import ELFMemory
from elftools.elf.elffile import ELFFile


class CoreSnapshot:
    def __init__(self, path):
        self.memory = ELFMemory(str(path))
        self.width, self.height = self.memory.get_fabric_dimensions()
        with Path(path).open("rb") as stream:
            elf = ELFFile(stream)
            if elf.elfclass != 64 or not elf.little_endian or elf.header.e_type != "ET_CORE":
                raise ValueError("Expected a little-endian ELF64 SDK core")
            segments = [
                s.header
                for s in elf.iter_segments()
                if s.header.p_type == "PT_LOAD" and s.header.p_vaddr == 0
            ]
        if len(segments) != 1:
            raise ValueError("SDK core must contain exactly one full SRAM segment")
        segment = segments[0]
        self.pe_bytes = segment.p_memsz
        size = segment.p_filesz
        if (
            self.width < 1
            or self.height < 1
            or self.pe_bytes < 1
            or size != self.width * self.height * self.pe_bytes
            or segment.p_offset + size > Path(path).stat().st_size
        ):
            raise ValueError("Truncated or incompatible SDK SRAM segment")

    def half_bits(self, x, y, address, size):
        if (
            not 0 <= x < self.width
            or not 0 <= y < self.height
            or address < 0
            or size < 1
            or address % 2
            or size % 2
            or address + size > self.pe_bytes
        ):
            raise ValueError("Snapshot array escapes its physical PE SRAM")
        data = self.memory.read_data(x, y, address, size)
        if len(data) != size or any(value is None for value in data):
            raise ValueError("Snapshot does not contain the complete array")
        return np.frombuffer(bytes(data), dtype="<u2").copy()


def program_symbols(path, expected):
    """Resolve exact-sized source globals, never pointer values or nearby bytes."""
    result = {}
    with Path(path).open("rb") as stream:
        elf = ELFFile(stream)
        table = elf.get_section_by_name(".symtab")
        if table is None:
            raise ValueError("Compiled ELF has no symbol table")
        for symbol in table.iter_symbols():
            alias = re.fullmatch(r"\$\$csl_base_address\$\$\d+\$\$(\w+)", symbol.name)
            name = alias[1] if alias else symbol.name
            if name not in expected or symbol.entry.st_info.type != "STT_OBJECT":
                continue
            location = (symbol.entry.st_value, symbol.entry.st_size)
            if (name in result and result[name] != location) or location[1] != 2 * expected[name]:
                raise ValueError(f"Ambiguous or incorrectly sized snapshot symbol: {name}")
            result[name] = location
    if result.keys() != expected.keys():
        raise ValueError("Compiled ELF is missing a required full snapshot array")
    return result


def read_core_arrays(core, program_dir, p, expected):
    """Return all requested FP16 words for every compute PE, in PE-major order."""
    snapshot = CoreSnapshot(core)
    arrays = {name: np.empty((p, p, count), np.uint16) for name, count in expected.items()}
    seen = set()
    for path in sorted(Path(program_dir).glob("*.elf")):
        program = ELFMemory(str(path))
        if program.get_flags() != snapshot.memory.get_flags():
            raise ValueError("Snapshot and compiled ELF target different architectures")
        coordinates = list(program.iter_coordinates())
        if len(coordinates) != 1:
            raise ValueError("Expected one physical compute PE per compiled ELF")
        x, y = map(int, coordinates[0])
        logical = (x - 4, y - 1)
        if not all(0 <= v < p for v in logical) or logical in seen:
            raise ValueError("Compiled ELF snapshot footprint differs from the profile")
        seen.add(logical)
        for name, (address, size) in program_symbols(path, expected).items():
            arrays[name][logical[1], logical[0]] = snapshot.half_bits(x, y, address, size)
    if seen != {(x, y) for x in range(p) for y in range(p)}:
        raise ValueError("Snapshot is missing a compute PE")
    return {name: value.ravel().view(np.float16) for name, value in arrays.items()}
