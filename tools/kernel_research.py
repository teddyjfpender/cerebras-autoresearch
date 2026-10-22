#!/usr/bin/env python3
"""Run reproducible CSL kernel experiments with numerical and confirmation gates."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "research/kernel_sources.json"
CASES = {"positive", "signed", "zero", "cancellation", "one_hot", "mixed_scale"}


def compare(base_path, candidate_path, minimum_speedup=1.05):
    """Compare a fully validated pair; a screen can never pass this gate."""
    base, candidate = (json.loads(Path(p).read_text()) for p in (base_path, candidate_path))
    for path, run in zip((base_path, candidate_path), (base, candidate)):
        validation = run.get("validation_result", {})
        cases = validation.get("cases", [])
        if (run.get("status") != "validated_kernel_run" or run.get("screen_only")
                or validation.get("status") != "all_cases_passed"
                or len(cases) != len(CASES) or {c["case"] for c in cases} != CASES):
            raise ValueError("Only complete six-case numerical runs can be compared")
        if any(not math.isfinite(c["max_fraction_of_error_bound"])
               or not 0 <= c["max_fraction_of_error_bound"] <= 1 for c in cases):
            raise ValueError("Numerical contract failed")
        cycles = [c["upstream_max_cycles"] for c in cases]
        if any(not math.isfinite(v) or v <= 0 for v in cycles):
            raise ValueError("Missing or invalid simulated timing")
        if run.get("upstream_max_simulated_cycles") != max(cycles):
            raise ValueError("Timing summary does not match per-case measurements")
        if validation.get("seed") != run.get("seed"):
            raise ValueError("Validation seed differs from run settings")
        cfg = f"out_4_{run['M'] // 4}_{run['N'] // 4}_2"
        for case in cases:
            artifact = Path(path).parent / cfg / case.get("artifact", "missing")
            if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != case.get("artifact_sha256"):
                raise ValueError(f"Missing or changed raw artifact: {artifact}")
    if base.get("variant") != "baseline" or candidate.get("variant") == "baseline":
        raise ValueError("Expected baseline and candidate variants")
    for key in ("P", "M", "N", "group_num", "seed", "source", "verifier_sha256", "commands"):
        if base.get(key) != candidate.get(key) or key not in base:
            raise ValueError(f"Incomparable runs: {key} differs or is absent")
    for key in ("sif_sha256", "sdk_release", "sdk_build"):
        if base["sdk_install"].get(key) != candidate["sdk_install"].get(key):
            raise ValueError(f"Incomparable SDK: {key}")
    for key in ("reference", "contract", "warmups", "repeats", "timing_status"):
        if base["validation_result"].get(key) != candidate["validation_result"].get(key):
            raise ValueError(f"Incomparable validation: {key}")
    speedup = base["upstream_max_simulated_cycles"] / candidate["upstream_max_simulated_cycles"]
    return {"baseline": str(base_path), "candidate": str(candidate_path),
            "metric": "upstream maximum simulated cycle span", "speedup": speedup,
            "cycle_reduction_percent": 100 * (1 - 1 / speedup),
            "minimum_speedup": minimum_speedup,
            "decision": "needs_fresh_confirmation" if speedup >= minimum_speedup else "rejected",
            "qwen_tps_result": False}


def challenge(variant, m, n, seed, minimum_speedup, initial_paths=()):
    """Bounded single-shape search: paired baseline, screen, full suite, fresh pair."""
    if variant == "baseline":
        raise ValueError("Challenge requires a candidate variant")
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-challenge-{os.getpid()}"
    folder = ROOT / "runs" / run_id
    folder.mkdir(parents=True)
    report = {"variant": variant, "M": m, "N": n, "seed": seed,
              "minimum_speedup": minimum_speedup, "status": "running", "runs": [],
              "qwen_tps_result": False}

    def execute(kind, run_seed, screen=False):
        path = meshgemv(kind, m, n, run_seed, screen)
        if not isinstance(path, Path):
            raise RuntimeError("SDK preflight failed")
        report["runs"].append(str(path))
        return path

    try:
        if initial_paths:
            base_path, candidate_path = map(Path, initial_paths)
            initial = json.loads(candidate_path.read_text())
            if any(initial.get(k) != v for k, v in
                   (("variant", variant), ("M", m), ("N", n), ("seed", seed))):
                raise ValueError("Initial pair does not match challenge settings")
            report["runs"].extend(str(p) for p in (base_path, candidate_path))
        else:
            base_path = execute("baseline", seed)
            screen_path = execute(variant, seed, screen=True)
            base, screen = (json.loads(p.read_text()) for p in (base_path, screen_path))
            if base["upstream_max_simulated_cycles"] / screen["upstream_max_simulated_cycles"] < minimum_speedup:
                report["status"] = "rejected_at_screen"
                return 0
            candidate_path = execute(variant, seed)
        if candidate_path:
            first = compare(base_path, candidate_path, minimum_speedup)
            report["initial_comparison"] = first
            if first["decision"] == "rejected":
                report["status"] = "rejected_after_full_validation"
            else:
                fresh_base = execute("baseline", seed + 1)
                fresh_candidate = execute(variant, seed + 1)
                confirmation = compare(fresh_base, fresh_candidate, minimum_speedup)
                for previous, fresh in ((base_path, fresh_base), (candidate_path, fresh_candidate)):
                    old, new = (json.loads(p.read_text()) for p in (previous, fresh))
                    for key in ("source_sha256", "verifier_sha256", "sdk_install"):
                        if old.get(key) != new.get(key):
                            raise ValueError(f"Source or toolchain changed during confirmation: {key}")
                report["confirmation_comparison"] = confirmation
                report["status"] = ("confirmed_kernel_improvement" if
                                    confirmation["decision"] != "rejected" else "rejected_at_confirmation")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        (folder / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(folder / "result.json", flush=True)
    return 0


def command(args, cwd, timeout=120):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True,
                            timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"{' '.join(args)} failed:\n{result.stdout}{result.stderr}")
    return result.stdout


def run_logged(args, cwd, log_path, timeout=600):
    with log_path.open("w") as log:
        process = subprocess.Popen(args, cwd=cwd, text=True, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                # A simulator shutdown fault under emulation can flood sim.log.
                # Bound disk usage as well as wall time and reject the run.
                for path in [log_path, *cwd.glob("out*/sim.log")]:
                    if path.stat().st_size > 64 * 1024 * 1024:
                        raise RuntimeError(f"Log exceeded 64 MiB; run rejected: {path}")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Command timed out; see {log_path}")
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            code = process.returncode
        except BaseException:
            if process.poll() is None:
                # Stop the whole isolated group immediately to bound log growth.
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f"Command failed ({code}); see {log_path}")
    return log_path.read_text()


def verify_source(path, spec):
    head = command(["git", "rev-parse", "HEAD"], path).strip()
    if head != spec["commit"]:
        raise RuntimeError(f"{path}: expected {spec['commit']}, found {head}; preserving checkout")
    # Verify blobs directly: Linux git status can spin while refreshing an index
    # created by macOS on the 9p mount. This also detects untracked source files.
    entries = command(["git", "ls-tree", "-rz", "HEAD"], path).split("\0")
    tracked = set()
    for entry in filter(None, entries):
        metadata, name = entry.split("\t", 1)
        mode, kind, expected = metadata.split()
        leaf = path / name
        if kind != "blob" or not leaf.exists():
            raise RuntimeError(f"Unsupported or missing source: {leaf}; preserving checkout")
        data = os.readlink(leaf).encode() if mode == "120000" else leaf.read_bytes()
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual != expected:
            raise RuntimeError(f"{leaf} changed; preserving checkout")
        tracked.add(name)
    observed = set()
    for directory, folders, files in os.walk(path):
        if Path(directory) == path:
            folders[:] = [name for name in folders if name != ".git"]
        for name in files:
            observed.add(str((Path(directory) / name).relative_to(path)))
    if observed != tracked:
        raise RuntimeError(f"{path} has untracked or missing files; preserving checkout")


def fetch():
    specs = json.loads(SOURCES.read_text())
    for name, spec in specs.items():
        path = ROOT / ".deps" / name
        if not path.exists():
            path.mkdir(parents=True)
            command(["git", "init", "-q"], path)
            command(["git", "remote", "add", "origin", spec["url"]], path)
            command(["git", "fetch", "--depth=1", "origin", spec["commit"]], path)
            command(["git", "checkout", "--detach", "FETCH_HEAD"], path)
        verify_source(path, spec)
        print(f"{name}: {spec['commit']}", flush=True)


def preflight():
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "cslc": shutil.which("cslc"), "cs_python": shutil.which("cs_python"),
            "limactl": shutil.which("limactl"), "api_key_required": False,
            "sdk_release_target": "2.10.1", "blockers": []}
    for name in ("cslc", "cs_python"):
        if not info[name]:
            info["blockers"].append(f"{name} absent from PATH")
    if info["cslc"]:
        try:
            # The SDK's cslc wrapper has no standalone --version command.
            info["compiler_help"] = command([info["cslc"], "--help"], ROOT, 30).strip()
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
            info["blockers"].append(str(error))
    info["ready_for_smoke"] = not info["blockers"]
    info["sdk_download"] = "https://www.cerebras.ai/developers/sdk-request"
    return info


def smoke():
    info = preflight()
    if not info["ready_for_smoke"]:
        print(json.dumps(info, indent=2))
        return 2
    specs = json.loads(SOURCES.read_text())
    source = ROOT / ".deps/sdk-examples"
    if not source.exists():
        raise RuntimeError("Run python3 tools/kernel_research.py fetch first")
    verify_source(source, specs["sdk-examples"])
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-sdk-smoke-{os.getpid()}"
    run_dir = ROOT / "runs" / run_id
    shutil.copytree(source / "tutorials/gemv-01-complete-program", run_dir)
    result = {"run_id": run_id, "sdk": info, "source": specs["sdk-examples"],
              "status": "running", "benchmark_result": False}
    commands = [
        ["cslc", "./layout.csl", "--arch=wse3", "--fabric-dims=8,3",
         "--fabric-offsets=4,1", "--memcpy", "--channels=1", "-o", "out"],
        ["cs_python", "run.py", "--name", "out"],
    ]
    result["commands"] = commands
    try:
        for index, args in enumerate(commands):
            log = run_logged(args, run_dir, run_dir / f"step-{index}.log")
        if "SUCCESS!" not in log:
            raise RuntimeError("Tutorial did not report its numerical correctness check as successful")
        result["status"] = "correctness_smoke_passed"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(run_dir / "result.json")
    return 0


def meshgemv(variant, m, n, seed, screen=False):
    if min(m, n) <= 0 or m % 4 or n % 4:
        raise ValueError("M and N must be positive multiples of 4")
    info = preflight()
    if not info["ready_for_smoke"]:
        print(json.dumps(info, indent=2))
        return 2
    spec = json.loads(SOURCES.read_text())["waferllm"]
    source = ROOT / ".deps/waferllm"
    verify_source(source, spec)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-meshgemv-{os.getpid()}"
    run_dir = ROOT / "runs" / run_id
    shutil.copytree(source / "MeshGEMV", run_dir)
    shutil.copy2(ROOT / "kernels/meshgemv/verify_kernel.py", run_dir / "verify_kernel.py")
    if variant == "bank-alignment":
        # Derived from the pinned Apache-2.0 WaferLLM kernel; see kernels/meshgemv/LICENSE.upstream.
        kernel = run_dir / "src/meshgemv.csl"
        original = kernel.read_text()
        changed = original.replace(
            "var W_tile: [Mt*Nt]f16 = @zeros([Mt*Nt]f16);\nvar ptr_W: [*]f16 = &W_tile;\nvar W_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Nt} -> W_tile[i] });",
            "const weight_storage_t = struct { pad: [2]f16, values: [Mt*Nt]f16, };\nvar W_storage: weight_storage_t align(16) = .{ .pad = @zeros([2]f16), .values = @zeros([Mt*Nt]f16) };\nvar ptr_W: [*]f16 = &W_storage.values;\nvar W_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &W_storage.values, .extent = @as(u16, Nt) });")
        changed = changed.replace("var res_tile: [Nt]f16 =", "var res_tile: [Nt]f16 align(16) =")
        if changed == original or "W_tile" in changed:
            raise RuntimeError("Candidate transformation did not match pinned upstream source")
        kernel.write_text(changed)
    elif variant == "unroll":
        kernel = run_dir / "src/meshgemv.csl"
        original = kernel.read_text()
        expanded = "\n".join(
            f"    @fmach(res_dest_dsr, res_src0_dsr, W_src1_dsr, X_tile[{row}]);"
            for row in range(m // 4))
        if original.count("    @map(gemv_static_step, X_dsd);") != 1:
            raise RuntimeError("Unroll transformation did not match pinned source")
        kernel.write_text(original.replace("    @map(gemv_static_step, X_dsd);", expanded))
    digest = hashlib.sha256()
    for path in sorted((run_dir / "src").rglob("*.csl")):
        digest.update(str(path.relative_to(run_dir)).encode())
        digest.update(path.read_bytes())
    cfg = f"out_4_{m // 4}_{n // 4}_2"
    result = {"run_id": run_id, "sdk": info, "source": spec,
              "sdk_install": json.loads((ROOT / "research/sdk_install.json").read_text()),
              "status": "running", "P": 4, "M": m, "N": n, "group_num": 2, "seed": seed,
              "variant": variant, "source_sha256": digest.hexdigest(),
              "screen_only": screen,
              "verifier_sha256": hashlib.sha256((run_dir / "verify_kernel.py").read_bytes()).hexdigest(),
              "extra_explicit_storage_bytes_per_PE": 4 if variant == "bank-alignment" else 0,
              "validation": ("positive-only provisional screen" if screen else
                             "six fixed cases against independent FP64 numerical oracle"),
              "qwen_tps_result": False}
    commands = [
        ["cslc", "--arch=wse3", "./src/layout.csl", "--fabric-dims=11,6",
         "--fabric-offsets=4,1", f"--params=P:4,Mt:{m // 4},Nt:{n // 4},pe_num_group:2,root_1st_phase:1,root_2nd_phase:3",
         "-o", cfg, "--memcpy", "--channels=1"],
    ]
    if screen:
        commands.append(["cs_python", "verify_kernel.py", "--name", cfg, "--seed", str(seed), "--screen"])
    else:
        # After observing shutdown faults under QEMU, require each numerical
        # case to exit cleanly in its own process. Baseline and candidates use
        # identical process boundaries.
        commands.extend(["cs_python", "verify_kernel.py", "--name", cfg, "--seed", str(seed), "--case", case]
                        for case in sorted(CASES))
    result["commands"] = commands
    try:
        for index, args in enumerate(commands):
            print(f"MeshGEMV step {index}: {' '.join(args)}", flush=True)
            log = run_logged(args, run_dir, run_dir / f"step-{index}.log")
            if index:
                if "SUCCESS!" not in log:
                    raise RuntimeError("MeshGEMV numerical check did not report success")
                simulator_log = run_dir / cfg / "sim.log"
                if simulator_log.exists():
                    simulator_log.rename(run_dir / cfg / f"sim-step-{index}.log")
        if "SUCCESS!" not in log:
            raise RuntimeError("MeshGEMV numerical check did not report success")
        if not screen:
            reports = [json.loads((run_dir / cfg / f"validation-{case}.json").read_text())
                       for case in sorted(CASES)]
            if any(r["status"] != "case_passed" for r in reports):
                raise RuntimeError("Incomplete per-case validation")
            combined = dict(reports[0], status="all_cases_passed",
                            cases=[case for report in reports for case in report["cases"]],
                            process_isolation="one fresh simulator process per numerical case")
            (run_dir / cfg / "validation.json").write_text(json.dumps(combined, indent=2) + "\n")
        result["validation_result"] = json.loads((run_dir / cfg / "validation.json").read_text())
        result["upstream_max_simulated_cycles"] = max(
            case["upstream_max_cycles"] for case in result["validation_result"]["cases"])
        result["status"] = "provisional_screen_passed" if screen else "validated_kernel_run"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(run_dir / "result.json")
    return run_dir / "result.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "preflight", "smoke", "meshgemv", "compare", "challenge"))
    parser.add_argument("results", nargs="*", help="Baseline/candidate pair for compare or challenge reuse")
    parser.add_argument("--variant", choices=("baseline", "bank-alignment", "unroll"), default="baseline")
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--minimum-speedup", type=float, default=1.05,
                        help="Kernel retention gate, default 1.05; no model TPS claim")
    args = parser.parse_args()
    if not math.isfinite(args.minimum_speedup) or args.minimum_speedup <= 1:
        parser.error("--minimum-speedup must exceed 1")
    if args.command == "compare":
        if len(args.results) != 2:
            parser.error("compare requires BASELINE CANDIDATE result paths")
        print(json.dumps(compare(*args.results, args.minimum_speedup), indent=2))
        return 0
    if args.command == "challenge":
        if len(args.results) not in (0, 2):
            parser.error("challenge accepts either no paths or a validated BASELINE CANDIDATE pair")
        return challenge(args.variant, args.M, args.N, args.seed, args.minimum_speedup, args.results)
    if args.results:
        parser.error("Unexpected result paths")
    if args.command == "fetch":
        fetch()
        return 0
    if args.command == "smoke":
        return smoke()
    if args.command == "meshgemv":
        result = meshgemv(args.variant, args.M, args.N, args.seed, args.screen)
        return result if isinstance(result, int) else 0
    info = preflight()
    print(json.dumps(info, indent=2))
    return 0 if info["ready_for_smoke"] else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, subprocess.TimeoutExpired, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
