"""Isolated SDK runs with atomic receipts, numerical evidence, and resource metrics."""

import json
import math
import re
import shutil
import time
import uuid

from .adapters import prepare
from .catalog import profile_identity
from .paths import ROOT, digest, identity, tree_digest, write_json
from .process import run_logged

CASES = ("positive", "signed", "zero", "cancellation", "one_hot", "mixed_scale")


def execute(
    profile, variant="baseline", seed=20261022, screen=False, candidate=None, deadline=None
):
    if not shutil.which("cslc") or not shutil.which("cs_python"):
        raise RuntimeError("SDK tools absent from PATH; run through tools/sdk-run")
    if variant == "candidate" and candidate is None:
        raise ValueError("--variant candidate requires --candidate NAME")
    if candidate and variant != "candidate":
        raise ValueError("Use --variant candidate with a candidate workspace")
    run_id = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-"
        + profile["id"]
        + "-"
        + uuid.uuid4().hex[:8]
    )
    folder = ROOT / "runs/v2" / run_id
    folder.mkdir(parents=True)
    work = folder / "work"
    record = {
        "schema_version": 2,
        "id": run_id,
        "status": "preparing",
        "profile": profile,
        "profile_sha256": profile_identity(profile),
        "variant": variant,
        "seed": seed,
        "screen_only": screen,
        "simulator_threads": 1,
        "architecture": "wse3",
        "sdk": json.loads((ROOT / "config/sdk-install.json").read_text()),
        "steps": [],
        "cases": [],
        "metrics": {},
        "artifacts": {},
        "qwen_tps_result": False,
    }
    receipt = folder / "result.json"
    write_json(receipt, record)

    def step(args):
        remaining = profile["timeout_seconds"]
        if deadline is not None:
            remaining = min(remaining, deadline - time.monotonic())
        if remaining <= 0:
            raise RuntimeError("Experiment budget exhausted")
        index = len(record["steps"])
        print(f"{profile['id']} [{variant}] step {index}: {' '.join(args)}", flush=True)
        data = run_logged(args, work, work / f"step-{index}.log", remaining)
        record["steps"].append(data)
        write_json(receipt, record)
        return (work / data["log"]).read_text()

    try:
        compile_args, source, baseline_hash = prepare(profile, work, variant, candidate)
        record.update(
            source=source,
            baseline_tree_sha256=baseline_hash,
            source_sha256=tree_digest(work),
            status="running",
        )
        record["harness_files"] = {
            "benchmark_" + p.name if p.name == "pipeline.py" else p.name: digest(
                work / ("benchmark_" + p.name if p.name == "pipeline.py" else p.name)
            )
            for p in (ROOT / "benchmarks/drivers").glob("*.py")
        }
        record["harness_sha256"] = identity(record["harness_files"])
        record["compile_command"] = compile_args
        write_json(receipt, record)
        step(compile_args)
        kind = profile["adapter"]
        if kind == "sdk":
            log = step(
                [
                    "cs_python",
                    "sdk_reference.py",
                    "--seed",
                    str(seed),
                    "--validation",
                    profile["validation"],
                    *profile["run"],
                ]
            )
            if not re.search(profile["success_pattern"], log):
                raise ValueError(
                    "Upstream host driver did not report its expected completion marker"
                )
            report = json.loads((work / "upstream-validation.json").read_text())
            if not report["captured_arrays"]:
                raise ValueError("Upstream driver returned no captured output arrays")
            if profile["cycle_pattern"]:
                matches = re.findall(profile["cycle_pattern"], log)
                if len(matches) != 1:
                    raise ValueError("Expected exactly one audited cycle measurement")
                cycles = float(matches[0])
                if not math.isfinite(cycles) or cycles <= 0:
                    raise ValueError("Invalid upstream timing measurement")
                record["metrics"]["upstream_cycles"] = cycles
            record["validation"] = report
            record["status"] = (
                "diagnostic_completed"
                if profile["validation"] == "diagnostic_only"
                else "reference_validated"
            )
        else:
            driver = "benchmark_pipeline.py" if kind == "pipeline" else "dense.py"
            for case in CASES[:1] if screen else CASES:
                args = ["cs_python", driver, "--name", "out", "--case", case, "--seed", str(seed)]
                if kind != "pipeline":
                    args.extend(["--kind", kind])
                log = step(args)
                if "SUCCESS!" not in log:
                    raise ValueError(f"Numerical driver failed to complete: {case}")
                data = json.loads((work / "out" / f"case-{case}.json").read_text())
                if not data["passed"] or data["seed"] != seed:
                    raise ValueError(f"Numerical validation failed: {case}")
                record["cases"].append(data)
                (work / "out/sim.log").rename(work / "out" / f"sim-{case}.log")
                write_json(receipt, record)
            for metric in ("max_local_cycles", "mean_local_cycles", "pe_cycle_spread"):
                record["metrics"][metric] = max(c["metrics"][metric] for c in record["cases"])
            record["metrics"]["max_error_fraction"] = max(
                c["max_fraction_of_error_bound"] for c in record["cases"]
            )
            record["validation"] = {
                "oracle": "FP64 on exactly quantized inputs",
                "contract": "fixed_forward_error_v2",
                "cases": [c["case"] for c in record["cases"]],
                "fresh_process_per_case": True,
            }
            record["status"] = "screen_passed" if screen else "validated"
        step(["cs_python", "memory.py", "out"])
        record["memory"] = json.loads((work / "out/memory.json").read_text())
        observed = {(row["x"], row["y"]) for row in record["memory"]["observations"]}
        width, height = profile["pe_grid"]
        expected = {(x + 4, y + 1) for x in range(width) for y in range(height)}
        if observed != expected:
            raise ValueError("Compiled program PE footprint differs from the declared profile")
        record["metrics"]["max_program_object_bytes_per_pe"] = record["memory"][
            "max_program_object_bytes_per_pe"
        ]
        primary = record["metrics"].get(profile["metric"])
        if primary:
            record["metrics"]["work_units_per_cycle"] = profile["work_units"] / primary
            record["metrics"]["pe_cycles"] = primary * math.prod(profile["pe_grid"])
        record["metrics"]["experiment_wall_seconds"] = sum(
            s["wall_seconds"] for s in record["steps"]
        )
    except BaseException as error:
        record["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        record["error"] = str(error)
        raise
    finally:
        if work.exists():
            for path in sorted(work.rglob("*")):
                if path.is_file() and path.suffix in (".npz", ".json", ".log", ".csl", ".txt"):
                    record["artifacts"][str(path.relative_to(folder))] = digest(path)
        write_json(receipt, record)
        print(receipt, flush=True)
    return receipt
