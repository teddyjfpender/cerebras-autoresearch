#!/usr/bin/env python3
"""Export an immutable, artifact-audited terminal campaign report.

This replays receipt integrity and promotion decisions. It does not recompute
numerical arrays; use replay_numerics.py in SDK NumPy for that separate audit.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cerebras_research.ledger import receipt_row
from cerebras_research.paths import contained, digest, identity, write_json

LIVE = {"running", "preparing"}


def build_report(plan_path, state_path):
    plan, state = (json.loads(path.read_text()) for path in (plan_path, state_path))
    if state["status"] in LIVE:
        raise ValueError("Campaign is still running; use --status for progress")
    if identity(plan) != state["plan_sha256"]:
        raise ValueError("Campaign plan differs from its frozen identity")
    pointers = {}

    def add(relative, sha=None):
        path = contained(ROOT, relative)
        actual = digest(path)
        if sha and sha != actual:
            raise ValueError("Campaign receipt changed")
        pointers[str(path.relative_to(ROOT))] = actual

    for pointer in state["runs"].values():
        add(pointer["path"], pointer["sha256"])
    for attempt in state.get("attempts", []):
        add(attempt["path"], attempt["sha256"])
    for cell in state["cells"].values():
        for key in ("baseline", "receipt", "confirmation"):
            if cell.get(key):
                add(cell[key])
        for screen in cell.get("screens", []):
            add(screen["receipt"])
    for decision in state["decisions"].values():
        add(decision["receipt"], decision.get("sha256"))
    records = []
    for relative, sha in sorted(pointers.items()):
        path = contained(ROOT, relative)
        data = json.loads(path.read_text())
        if data.get("status") in LIVE:
            raise ValueError("A terminal checkpoint includes a live attempt")
        row = receipt_row(path, "live")
        if row is None:
            raise ValueError("Campaign includes an unsupported receipt")
        evaluator = data.get("provenance", {}).get("evaluator_sha256")
        if evaluator and evaluator != state["evaluator_sha256"]:
            raise ValueError("Campaign includes a different evaluator track")
        records.append(
            dict(
                receipt=relative,
                receipt_sha256=sha,
                kind=row["kind"],
                evidence_audit=row["evidence_audit"],
                status=data.get("status") or data.get("decision") or row["outcome"],
                **{
                    key: data[key]
                    for key in (
                        "decision",
                        "profile",
                        "variant",
                        "seed",
                        "screen_only",
                        "metrics",
                        "memory",
                        "validation",
                        "cases",
                        "research",
                        "provenance",
                        "sdk",
                        "profile_sha256",
                        "baseline_tree_sha256",
                        "source_sha256",
                        "harness_sha256",
                        "compile_command",
                        "steps",
                        "error",
                        "inputs",
                        "budgets",
                        "initial",
                        "confirmation",
                        "eligible",
                        "speedup",
                        "memory_ratio",
                        "metric",
                        "cycle_reduction_percent",
                    )
                    if key in data
                },
            )
        )
    return dict(
        schema_version=1,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        campaign_status=state["status"],
        plan=dict(
            path=str(plan_path.relative_to(ROOT)), sha256=digest(plan_path), identity=identity(plan)
        ),
        checkpoint=dict(path=str(state_path.relative_to(ROOT)), sha256=digest(state_path)),
        evaluator_sha256=state["evaluator_sha256"],
        audit_tool_sha256=digest(Path(__file__)),
        audit_evaluator_sha256=identity(
            {
                str(path.relative_to(ROOT)): digest(path)
                for path in sorted((ROOT / "src/cerebras_research").glob("*.py"))
                + sorted((ROOT / "benchmarks/drivers").glob("*.py"))
            }
        ),
        candidate_identities=state.get("candidate_identities", {}),
        imports=dict(
            initial_baselines={
                cell["profile"]: cell["baseline_receipt"]
                for cell in plan["cells"]
                if cell.get("baseline_receipt")
            },
            fresh_baselines=state.get("imports", {}),
        ),
        executed_attempt_count=len(state.get("attempts", [])),
        requested_profiles=[cell["profile"] for cell in plan["cells"]],
        cells=state["cells"],
        records=records,
        record_count=len(records),
        validated_case_count=sum(
            len(r.get("cases", [])) for r in records if r["status"] == "validated"
        ),
        artifact_integrity_replayed=True,
        numerical_arrays_recomputed=False,
        qwen_tps_result=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("state", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse to overwrite an existing campaign report")
    report = build_report(args.plan.resolve(), args.state.resolve())
    write_json(args.output, report)
    print(
        json.dumps(
            dict(
                status=report["campaign_status"],
                records=report["record_count"],
                output=str(args.output),
            )
        )
    )


if __name__ == "__main__":
    main()
