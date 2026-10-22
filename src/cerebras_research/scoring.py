"""Reject incomparable or incomplete runs; aggregate paired suite results honestly."""

import json
import math
from pathlib import Path

from .paths import contained, digest, identity, tree_digest
from .runner import CASES


def verify_provenance(data, folder):
    provenance = data.get("provenance", {})
    if provenance:
        evaluator = provenance["evaluator_files"]
        if not evaluator or identity(evaluator) != provenance["evaluator_sha256"]:
            raise ValueError("Evaluator fingerprint changed")
        for relative, sha in evaluator.items():
            if digest(contained(folder / "evaluator-source", relative)) != sha:
                raise ValueError("Evaluator source capsule changed")


def load_validated(path):
    path = Path(path)
    data = json.loads(path.read_text())
    if (
        data.get("schema_version") != 2
        or data.get("status") != "validated"
        or data.get("screen_only")
    ):
        raise ValueError(
            "Promotion requires a full independent-oracle v2 run; screens/references/diagnostics cannot qualify"
        )
    if data["profile_sha256"] != identity(data["profile"]):
        raise ValueError("Profile fingerprint changed")
    verify_provenance(data, path.parent)
    if data["source_sha256"] != tree_digest(path.parent / "work"):
        raise ValueError("Compiled source fingerprint changed")
    if not data.get("harness_files") or data["harness_sha256"] != identity(data["harness_files"]):
        raise ValueError("Harness fingerprint missing or changed")
    for relative, expected in data["harness_files"].items():
        if digest(contained(path.parent / "work", relative)) != expected:
            raise ValueError("Trusted harness changed")
    cases = data.get("cases", [])
    if len(cases) != 6 or {c["case"] for c in cases} != set(CASES):
        raise ValueError("Missing or duplicate numerical cases")
    for case in cases:
        fraction = case["max_fraction_of_error_bound"]
        cycles = case["metrics"]["max_local_cycles"]
        if (
            not case["passed"]
            or case["seed"] != data["seed"]
            or not math.isfinite(fraction)
            or not 0 <= fraction <= 1
        ):
            raise ValueError("Numerical contract failed")
        if not math.isfinite(cycles) or cycles <= 0:
            raise ValueError("Invalid timing")
        raw = contained(path.parent / "work/out", case["artifact"])
        saved_case = json.loads(
            contained(path.parent, "work/out/case-" + case["case"] + ".json").read_text()
        )
        if saved_case != case:
            raise ValueError("Case receipt differs from measured evidence")
        if digest(raw) != case["artifact_sha256"]:
            raise ValueError("Raw numerical artifact changed")
    if data["metrics"]["max_local_cycles"] != max(c["metrics"]["max_local_cycles"] for c in cases):
        raise ValueError("Timing summary changed")
    memory = json.loads((path.parent / "work/out/memory.json").read_text())
    if (
        data["memory"] != memory
        or data["metrics"]["max_program_object_bytes_per_pe"]
        != memory["max_program_object_bytes_per_pe"]
    ):
        raise ValueError("Memory summary differs from measured evidence")
    if memory["max_program_object_bytes_per_pe"] <= 0:
        raise ValueError("Invalid program memory measurement")
    if not data.get("artifacts"):
        raise ValueError("Artifact manifest absent")
    for relative, expected in data["artifacts"].items():
        if digest(contained(path.parent, relative)) != expected:
            raise ValueError(f"Artifact changed: {relative}")
    return data


def compare(base_path, candidate_path, minimum_speedup=1.05, max_memory_growth=1.10):
    if not math.isfinite(minimum_speedup) or minimum_speedup <= 1:
        raise ValueError("Minimum speedup must exceed one")
    if not math.isfinite(max_memory_growth) or max_memory_growth < 1:
        raise ValueError("Memory growth budget must be at least one")
    base, candidate = load_validated(base_path), load_validated(candidate_path)
    if base.get("provenance", {}).get("evaluator_sha256") != candidate.get("provenance", {}).get(
        "evaluator_sha256"
    ):
        raise ValueError("Incomparable runs: evaluator fingerprint; create a matched new baseline")
    if base["variant"] != "baseline" or candidate["variant"] == "baseline":
        raise ValueError("Expected baseline and candidate")
    for key in (
        "profile_sha256",
        "profile",
        "seed",
        "architecture",
        "simulator_threads",
        "sdk",
        "source",
        "baseline_tree_sha256",
        "harness_sha256",
        "validation",
    ):
        if key not in base or base[key] != candidate.get(key):
            raise ValueError(f"Incomparable runs: {key}")
    primary = base["profile"]["metric"]
    speedup = base["metrics"][primary] / candidate["metrics"][primary]
    memory_ratio = (
        candidate["metrics"]["max_program_object_bytes_per_pe"]
        / base["metrics"]["max_program_object_bytes_per_pe"]
    )
    return {
        "profile": base["profile"]["id"],
        "metric": primary,
        "baseline": str(base_path),
        "candidate": str(candidate_path),
        "speedup": speedup,
        "memory_ratio": memory_ratio,
        "cycle_reduction_percent": 100 * (1 - 1 / speedup),
        "eligible": speedup >= minimum_speedup and memory_ratio <= max_memory_growth,
        "minimum_speedup": minimum_speedup,
        "max_memory_growth": max_memory_growth,
        "decision": "needs_fresh_confirmation"
        if speedup >= minimum_speedup and memory_ratio <= max_memory_growth
        else "rejected",
        "qwen_tps_result": False,
    }


def confirm(initial_base, initial_candidate, fresh_base, fresh_candidate, **budgets):
    initial = compare(initial_base, initial_candidate, **budgets)
    fresh = compare(fresh_base, fresh_candidate, **budgets)
    for old_path, new_path in ((initial_base, fresh_base), (initial_candidate, fresh_candidate)):
        old, new = load_validated(old_path), load_validated(new_path)
        if not old.get("provenance", {}).get("evaluator_sha256") or old.get("provenance", {}).get(
            "evaluator_sha256"
        ) != new.get("provenance", {}).get("evaluator_sha256"):
            raise ValueError(
                "Fresh confirmation requires matching evaluator provenance; rebaseline historical runs"
            )
        if old["seed"] == new["seed"]:
            raise ValueError("Fresh confirmation reused the initial seed")
        for key in (
            "profile_sha256",
            "source_sha256",
            "compile_command",
            "harness_sha256",
            "sdk",
            "simulator_threads",
            "variant",
        ):
            if old[key] != new[key]:
                raise ValueError(f"Confirmation changed {key}")
    return {
        "status": "confirmed_improvement"
        if initial["eligible"] and fresh["eligible"]
        else "rejected",
        "initial": initial,
        "confirmation": dict(
            fresh, decision="confirmation_passed" if fresh["eligible"] else "rejected"
        ),
        "scope": "one fixed benchmark profile",
        "qwen_tps_result": False,
    }


def aggregate(comparisons, expected_profiles, minimum_speedup=1.05, regression_limit=1.02):
    names = [item["profile"] for item in comparisons]
    if len(names) != len(set(names)) or set(names) != set(expected_profiles):
        raise ValueError("Suite coverage is incomplete or duplicated")
    ratios = [c["speedup"] for c in comparisons]
    if not ratios or any(not math.isfinite(v) or v <= 0 for v in ratios):
        raise ValueError("Invalid suite speedups")
    geo = math.exp(sum(math.log(v) for v in ratios) / len(ratios))
    return {
        "geometric_mean_speedup": geo,
        "worst_speedup": min(ratios),
        "profiles": names,
        "eligible": geo >= minimum_speedup
        and min(ratios) >= 1 / regression_limit
        and all(c["memory_ratio"] <= c["max_memory_growth"] for c in comparisons),
        "confirmation_required": True,
        "qwen_tps_result": False,
    }
