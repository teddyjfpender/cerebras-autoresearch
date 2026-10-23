"""Isolated SDK runs with atomic receipts, numerical evidence, and resource metrics."""

import json
import math
import re
import shutil
import time
import uuid

from .adapters import prepare
from .catalog import profile_identity
from .paths import ROOT, contained, digest, identity, tree_digest, write_json
from .process import run_logged
from .provenance import capture
from .sdk import execution_sdk

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
    sdk = execution_sdk(ROOT)
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
        "sdk": sdk,
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
        started = time.monotonic()
        try:
            data = run_logged(args, work, work / f"step-{index}.log", remaining)
        except BaseException as error:
            record["steps"].append(
                {
                    "command": args,
                    "wall_seconds": time.monotonic() - started,
                    "log": f"step-{index}.log",
                    "status": "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                    "error": str(error),
                }
            )
            write_json(receipt, record)
            raise
        record["steps"].append(data)
        write_json(receipt, record)
        return (work / data["log"]).read_text()

    try:
        record["provenance"], record["research"] = capture(profile, variant, candidate)
        for relative, sha in record["provenance"]["evaluator_files"].items():
            target = contained(folder / "evaluator-source", relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
            if digest(target) != sha:
                raise ValueError("Evaluator changed while capturing its source capsule")
        compile_args, source, baseline_hash = prepare(profile, work, variant, candidate)
        record.update(
            source=source,
            baseline_tree_sha256=baseline_hash,
            source_sha256=tree_digest(work),
            status="running",
        )
        record["harness_files"] = {
            str(path.relative_to(work)): digest(path) for path in sorted(work.rglob("*.py"))
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
            driver = {
                "pipeline": "benchmark_pipeline.py",
                "gated_mlp": "resident.py",
                "decoder": "resident.py",
                "reusable_gemm": "reusable_gemm.py",
            }.get(kind, "dense.py")
            for case in CASES[:1] if screen else CASES:
                args = ["cs_python", driver, "--name", "out", "--case", case, "--seed", str(seed)]
                if kind not in ("pipeline", "reusable_gemm"):
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
                "contract": profile.get("oracle_contract", "fixed_forward_error_v2"),
                "cases": [c["case"] for c in record["cases"]],
                "fresh_process_per_case": True,
            }
            record["status"] = "screen_passed" if screen else "validated"
            if kind in ("gated_mlp", "decoder"):
                record["metrics"][
                    "cycles_per_frame" if kind == "gated_mlp" else "cycles_per_token"
                ] = record["metrics"]["max_local_cycles"] / profile["dimensions"]["T"]
            if kind == "reusable_gemm":
                record["metrics"]["cycles_per_gemm"] = (
                    record["metrics"]["max_local_cycles"] / profile["dimensions"]["T"]
                )
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
        record["metrics"]["experiment_wall_seconds"] = sum(
            step["wall_seconds"] for step in record["steps"]
        )
        if work.exists():
            for path in sorted(work.rglob("*")):
                if path.is_file() and path.suffix in (
                    ".npz",
                    ".json",
                    ".log",
                    ".csl",
                    ".txt",
                    ".elf",
                    ".core",
                ):
                    record["artifacts"][str(path.relative_to(folder))] = digest(path)
        write_json(receipt, record)
        from .ledger import append_receipts

        append_receipts([receipt])
        # Console delivery must not prevent durable evidence or mask the
        # original SDK failure when a long attached connection disappears.
        try:
            print(receipt, flush=True)
        except BrokenPipeError:
            pass
    return receipt
