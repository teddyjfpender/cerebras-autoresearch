"""Read complete rectangular compute grids as raw words, including shared ELFs."""

from pathlib import Path

import numpy as np
from snapshot import CoreSnapshot, ELFMemory, program_symbols


def read_core_grid_arrays(core, program_dir, width, height, expected):
    if width < 1 or height < 1 or not expected or any(count < 1 for count in expected.values()):
        raise ValueError("Invalid complete compute-grid contract")
    snapshot = CoreSnapshot(core)
    arrays = {name: np.empty((height, width, count), np.uint16) for name, count in expected.items()}
    seen = set()
    for path in sorted(Path(program_dir).glob("*.elf")):
        program = ELFMemory(str(path))
        if program.get_flags() != snapshot.memory.get_flags():
            raise ValueError("Snapshot and compiled ELF target different architectures")
        coordinates = list(program.iter_coordinates())
        if not coordinates:
            raise ValueError("Compiled program has no physical coordinates")
        symbols = program_symbols(path, expected)
        for x, y in coordinates:
            x, y = int(x), int(y)
            logical = x - 4, y - 1
            if not (0 <= logical[0] < width and 0 <= logical[1] < height) or logical in seen:
                raise ValueError("Compiled snapshot footprint differs from the fixed compute grid")
            seen.add(logical)
            for name, (address, size) in symbols.items():
                arrays[name][logical[1], logical[0]] = snapshot.half_bits(x, y, address, size)
    if seen != {(x, y) for x in range(width) for y in range(height)}:
        raise ValueError("Snapshot is missing a compute PE")
    return {name: value.ravel() for name, value in arrays.items()}
