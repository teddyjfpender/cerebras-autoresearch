"""Record SDK-reported program-object memory without assuming ELF loads add."""

import json
import re
import subprocess
import sys
from pathlib import Path

out = Path(sys.argv[1])
observations = []
for path in sorted((out / "bin").glob("*.elf")):
    text = subprocess.check_output(["/cbcore/bin/cs-readelf", "--msize", str(path)], text=True)
    (out / (path.stem + ".memory.txt")).write_text(text)
    for x, y, size in re.findall(r"\((\d+),\s*(\d+)\):\s*(\d+) bytes", text):
        observations.append({"object": path.name, "x": int(x), "y": int(y), "bytes": int(size)})
if not observations:
    raise RuntimeError("No program-object PE memory observations")
(out / "memory.json").write_text(
    json.dumps(
        {
            "max_program_object_bytes_per_pe": max(v["bytes"] for v in observations),
            "scope": "bin/*.elf program objects; not union of all ELF loads or physical area",
            "observations": observations,
        },
        indent=2,
    )
    + "\n"
)
