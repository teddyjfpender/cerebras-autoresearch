"""Unscored program-object memory checks on already compiled full-head ELFs."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PATHS = {
    "baseline": "runs/v2/20261023T160509Z-recurrent-gdn-full.k128-v128-h16-t2-a6dce575/work/out",
    "full-gdn-bulk-extended-dsr": "runs/maintenance/full-bulk-extended-compile-v4/out",
    "full-gdn-bulk-vector-norm": "runs/maintenance/full-bulk-vector-norm16-compile/out",
}


def main():
    output = ROOT / "results/reports/2026-10-23/recurrent-gdn-full/compiled-memory-preflight.json"
    if output.exists():
        raise ValueError("Historical reports are immutable")
    records = {}
    for name, relative in PATHS.items():
        observations = []
        files = {}
        for path in sorted((ROOT / relative / "bin").glob("*.elf")):
            text = subprocess.check_output(
                ["/cbcore/bin/cs-readelf", "--msize", str(path)], text=True
            )
            files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            observations.extend(
                dict(x=int(x), y=int(y), bytes=int(size))
                for x, y, size in re.findall(r"\((\d+),\s*(\d+)\):\s*(\d+) bytes", text)
            )
        if {(v["x"], v["y"]) for v in observations} != {
            (x + 4, y + 1) for x in range(32) for y in range(24)
        }:
            raise ValueError("Incomplete compiled compute grid")
        records[name] = dict(
            max_program_object_bytes_per_pe=max(v["bytes"] for v in observations),
            elf_sha256=files,
            observations=observations,
        )
    baseline = records["baseline"]["max_program_object_bytes_per_pe"]
    for record in records.values():
        record["ratio_to_baseline"] = record["max_program_object_bytes_per_pe"] / baseline
    result = dict(
        schema_version=1,
        status="compiled_memory_observed",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        records=records,
        sdk_device_execution=False,
        performance_measurement=False,
        qwen_tps_result=False,
        scope="Compiler program-object metric on complete768-PE footprints. Device correctness and cycles remain unverified; no retention decision.",
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {n: v[n] for n in ("max_program_object_bytes_per_pe", "ratio_to_baseline")}
                for k, v in records.items()
            }
        )
    )


if __name__ == "__main__":
    main()
