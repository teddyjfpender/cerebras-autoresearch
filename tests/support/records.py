"""Synthetic evidence fixtures for gate tests; these do not simulate numerical work."""

from cerebras_research.catalog import get_profile
from cerebras_research.paths import digest, identity, tree_digest, write_json
from cerebras_research.runner import CASES


def make_record(
    root, name, variant="baseline", seed=7, cycles=100, memory=100, profile_id="meshgemv.square64"
):
    folder = root / name
    work = folder / "work"
    (work / "out").mkdir(parents=True)
    (work / "kernel.csl").write_text("// " + variant)
    (work / "driver.py").write_text("# fixed oracle")
    capsule = folder / "evaluator-source"
    capsule.mkdir()
    (capsule / "evaluator.py").write_text("# synthetic evaluator fixture")
    evaluator = {"evaluator.py": digest(capsule / "evaluator.py")}
    profile = get_profile(profile_id)
    cases = []
    for case in CASES:
        raw = work / "out" / (case + ".npz")
        raw.write_bytes(f"{seed}-{case}".encode())
        data = {
            "case": case,
            "seed": seed,
            "passed": True,
            "max_fraction_of_error_bound": 0.1,
            "artifact": raw.name,
            "artifact_sha256": digest(raw),
            "metrics": {"max_local_cycles": cycles},
        }
        write_json(work / "out" / f"case-{case}.json", data)
        cases.append(data)
    mem = {"max_program_object_bytes_per_pe": memory}
    write_json(work / "out/memory.json", mem)
    harness = {"driver.py": digest(work / "driver.py")}
    data = {
        "schema_version": 2,
        "provenance": {"evaluator_files": evaluator, "evaluator_sha256": identity(evaluator)},
        "status": "validated",
        "screen_only": False,
        "profile": profile,
        "profile_sha256": identity(profile),
        "variant": variant,
        "compile_command": ["cslc", "kernel.csl", "--variant=" + variant],
        "seed": seed,
        "source_sha256": tree_digest(work),
        "harness_files": harness,
        "harness_sha256": identity(harness),
        "architecture": "wse3",
        "simulator_threads": 1,
        "sdk": {"sif_sha256": "fixed"},
        "source": {"commit": "fixed"},
        "baseline_tree_sha256": "fixed",
        "validation": {"contract": "fixed"},
        "cases": cases,
        "memory": mem,
        "metrics": {"max_local_cycles": cycles, "max_program_object_bytes_per_pe": memory},
        "artifacts": {
            str(p.relative_to(folder)): digest(p) for p in work.rglob("*") if p.is_file()
        },
    }
    path = folder / "result.json"
    write_json(path, data)
    return path
