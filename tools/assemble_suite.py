#!/usr/bin/env python3
"""Assemble complete suite evidence from hash-bound, already validated SDK runs.

This performs no new SDK measurements. Run the writer in the same Linux
environment as the simulator/journal, after the active campaign releases its lease.
"""

import argparse
import contextlib
import fcntl
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cerebras_research import ledger
from cerebras_research.catalog import get_suite, profile_identity
from cerebras_research.paths import contained, digest, identity, write_json
from cerebras_research.scoring import load_validated


@contextlib.contextmanager
def metadata_lease():
    """Share the SDK writer lock with a reserve sized for small metadata writes."""
    folder = ROOT / "runs"
    folder.mkdir(exist_ok=True)
    with (folder / ".simulator.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another benchmark owns the simulator lease") from error
        try:
            if shutil.disk_usage(folder).free < 64 * 1024**2:
                raise RuntimeError("Less than 64 MiB free; cannot write suite metadata")
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def build_suite(suite_name, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("receipts"):
        raise ValueError("Expected schema_version 1 and nonempty receipt pointers")
    profiles = get_suite(suite_name)
    if any(profile["validation"] != "independent_six_case" for profile in profiles):
        raise ValueError("Suite assembly requires independent six-case profiles")
    expected = {profile["id"]: profile_identity(profile) for profile in profiles}
    runs, imports, implementations, common = {}, {}, {}, None
    seen = set()
    for pointer in manifest["receipts"]:
        path = contained(ROOT, pointer["path"])
        relative = str(path.relative_to(ROOT))
        if relative in seen:
            raise ValueError("Duplicate receipt pointer")
        seen.add(relative)
        if digest(path) != pointer["sha256"]:
            raise ValueError("Imported receipt changed")
        data = load_validated(path)
        name = data["profile"]["id"]
        if name not in expected or data["profile_sha256"] != expected[name]:
            raise ValueError("Imported profile does not match the current named suite contract")
        if name in runs:
            raise ValueError("Duplicate suite profile")
        context = {
            key: data[key]
            for key in (
                "seed",
                "variant",
                "architecture",
                "simulator_threads",
                "sdk",
            )
        }
        context["evaluator_sha256"] = data.get("provenance", {}).get("evaluator_sha256")
        if not context["evaluator_sha256"]:
            raise ValueError("Suite assembly requires evaluator source capsules")
        if common is None:
            common = context
        elif context != common:
            raise ValueError("Suite mixes seeds, variants, evaluator or toolchains")
        runs[name] = relative
        imports[name] = {"path": relative, "sha256": pointer["sha256"]}
        implementations[name] = {
            "candidate": data.get("research", {}).get("candidate", ""),
            "variant": data["variant"],
            "source_sha256": data["source_sha256"],
            "baseline_tree_sha256": data["baseline_tree_sha256"],
            "harness_sha256": data["harness_sha256"],
        }
    if set(runs) != set(expected):
        raise ValueError("Suite coverage is incomplete")
    return {
        "schema_version": 2,
        "status": "completed",
        "suite": suite_name,
        "profiles": list(expected),
        "suite_contract_sha256": identity(profiles),
        **common,
        "screen_only": False,
        "runs": {name: runs[name] for name in expected},
        "failures": {},
        "run_imports": imports,
        "implementations": implementations,
        "executed_attempt_count": 0,
        "imported_run_count": len(runs),
        "execution_mode": "assembled_existing_full_runs",
        "assembly_tool_sha256": digest(Path(__file__)),
        "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": digest(manifest_path)},
        "scope": "exact suite coverage from separate profile runs; not a combined kernel launch",
        "qwen_tps_result": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run the writer through tools/sdk-run in the Linux journal environment")
    try:
        with metadata_lease():
            report = build_suite(args.suite, contained(ROOT, args.manifest))
            folder = (
                ROOT
                / "runs/v2"
                / (
                    time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                    + "-suite-reuse-"
                    + uuid.uuid4().hex[:8]
                )
            )
            folder.mkdir(parents=True, exist_ok=False)
            receipt = folder / "result.json"
            write_json(receipt, report)
            ledger.append_receipts([receipt])
        print(
            json.dumps(
                {
                    "receipt": str(receipt.relative_to(ROOT)),
                    "imported_runs": report["imported_run_count"],
                    "executed_attempts": 0,
                    "retention": "requires matched suite comparison and fresh confirmation",
                }
            )
        )
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
