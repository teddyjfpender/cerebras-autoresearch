#!/usr/bin/env python3
"""Run a finite candidate campaign with shared baselines and resumable evidence.

Run inside the SDK environment. The plan and evaluator are frozen per state
file; numerical/timing drivers and promotion gates come from the core runner.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cerebras_research.catalog import get_profile, profile_identity
from cerebras_research.cli import save_decision
from cerebras_research.paths import contained, digest, identity, tree_digest, write_json
from cerebras_research.process import simulator_lease
from cerebras_research.runner import execute
from cerebras_research.scoring import compare, confirm, load_validated


def evaluator_identity():
    paths = sorted((ROOT / "src/cerebras_research").glob("*.py")) + sorted(
        (ROOT / "benchmarks/drivers").glob("*.py")
    )
    return identity({str(path.relative_to(ROOT)): digest(path) for path in paths})


def campaign(plan_path, state_path, max_seconds, retry_failed=False):
    plan = json.loads(plan_path.read_text())
    if plan.get("schema_version") != 1 or not plan.get("cells"):
        raise ValueError("A campaign needs schema_version 1 and nonempty cells")
    candidates = {}
    for cell in plan["cells"]:
        for candidate in cell["candidates"]:
            if candidate.get("candidate"):
                path = contained(ROOT / "experiments/candidates", candidate["candidate"])
                candidates[candidate["candidate"]] = {
                    "source": tree_digest(path),
                    "manifest": digest(path / "candidate.json"),
                }
    frozen = {
        "plan_sha256": identity(plan),
        "evaluator_sha256": evaluator_identity(),
        "candidate_identities": candidates,
    }
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            **frozen,
            "status": "running",
            "runs": {},
            "attempts": [],
            "decisions": {},
            "cells": {},
        }
    )
    if any(state.get(key) != value for key, value in frozen.items()):
        raise ValueError("Plan, evaluator, or candidate changed; create a new campaign state")
    budgets = argparse.Namespace(
        minimum_speedup=plan.get("minimum_speedup", 1.05),
        max_memory_growth=plan.get("max_memory_growth", 1.10),
    )
    for value, lower, strict in (
        (budgets.minimum_speedup, 1, True),
        (budgets.max_memory_growth, 1, False),
    ):
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < lower
            or (strict and value == lower)
        ):
            raise ValueError("Invalid campaign cycle/memory gate")
    for pointer in state["runs"].values():
        if digest(contained(ROOT, pointer["path"])) != pointer["sha256"]:
            raise ValueError("Cached receipt changed")
    deadline = time.monotonic() + max_seconds

    def save():
        write_json(state_path, state)

    def data(path):
        return json.loads(Path(path).read_text())

    def remember(key, path):
        pointer = {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
        state["runs"][key] = pointer
        state["attempts"].append(dict(pointer, operation=key, outcome=data(path)["status"]))
        save()

    def run(profile, spec, seed, screen=False):
        if spec.get("candidate"):
            candidate = contained(ROOT / "experiments/candidates", spec["candidate"])
            actual = {
                "source": tree_digest(candidate),
                "manifest": digest(candidate / "candidate.json"),
            }
            if actual != frozen["candidate_identities"][spec["candidate"]]:
                raise ValueError("Candidate changed during campaign; create a new state")
        name = spec.get("candidate", spec.get("variant", "baseline"))
        key = f"{profile['id']}/{name}/{seed}/{'screen' if screen else 'full'}"
        existing = state["runs"].get(key)
        if existing:
            path = contained(ROOT, existing["path"])
            if digest(path) != existing["sha256"]:
                raise ValueError("Cached receipt changed")
            outcome = data(path)["status"]
            if outcome != "cancelled" and (outcome != "failed" or not retry_failed):
                return path
        if time.monotonic() >= deadline:
            raise TimeoutError("Campaign budget exhausted")
        if evaluator_identity() != frozen["evaluator_sha256"]:
            raise ValueError("Evaluator changed during campaign; stop and rebaseline")
        before = set((ROOT / "runs/v2").glob("*/result.json"))
        try:
            path = execute(
                profile,
                spec.get("variant", "candidate"),
                seed,
                screen,
                contained(ROOT / "experiments/candidates", spec["candidate"])
                if spec.get("candidate")
                else None,
                deadline,
            )
        except KeyboardInterrupt:
            created = set((ROOT / "runs/v2").glob("*/result.json")) - before
            if len(created) == 1:
                remember(key, created.pop())
            raise
        except (RuntimeError, ValueError, OSError) as error:
            created = set((ROOT / "runs/v2").glob("*/result.json")) - before
            if len(created) != 1:
                raise
            path = created.pop()
            remember(key, path)
            print(f"ATTEMPT FAILED: {error}", flush=True)
            return path
        remember(key, path)
        return path

    def decision(key, result, kind, inputs):
        if key not in state["decisions"]:
            state["decisions"][key] = save_decision(result, kind, inputs, budgets)
            save()
        return state["decisions"][key]

    seed = plan["seed"]
    state["status"] = "running"
    state.pop("error", None)
    save()
    try:
        for cell in plan["cells"]:
            profile = get_profile(cell["profile"])
            if profile["validation"] != "independent_six_case":
                raise ValueError("Campaign promotion requires an independent six-case profile")
            previous = state["cells"].get(cell["profile"])
            if previous and (not retry_failed or previous["status"] == "confirmed_improvement"):
                continue
            baseline_spec = {"variant": "baseline"}
            imported = cell.get("baseline_receipt")
            base = contained(ROOT, imported) if imported else run(profile, baseline_spec, seed)
            if data(base)["status"] != "validated":
                state["cells"][profile["id"]] = {
                    "status": "baseline_failed",
                    "receipt": str(base.relative_to(ROOT)),
                }
                save()
                continue
            baseline = load_validated(base)
            if (
                baseline["profile_sha256"] != profile_identity(profile)
                or baseline["seed"] != seed
                or baseline["variant"] != "baseline"
                or baseline.get("provenance", {}).get("evaluator_sha256")
                != frozen["evaluator_sha256"]
            ):
                raise ValueError("Imported baseline is not a matched campaign baseline")
            eligible, screens = [], []
            for spec in cell["candidates"]:
                path = run(profile, spec, seed, True)
                screen = data(path)
                entry = {
                    "candidate": spec,
                    "receipt": str(path.relative_to(ROOT)),
                    "outcome": screen["status"],
                }
                if screen["status"] == "screen_passed":
                    ratio = (
                        baseline["metrics"][profile["metric"]]
                        / screen["metrics"][profile["metric"]]
                    )
                    memory = (
                        screen["metrics"]["max_program_object_bytes_per_pe"]
                        / baseline["metrics"]["max_program_object_bytes_per_pe"]
                    )
                    entry.update(
                        speedup=ratio,
                        memory_ratio=memory,
                        eligible=ratio >= budgets.minimum_speedup
                        and memory <= budgets.max_memory_growth,
                    )
                    if entry["eligible"]:
                        eligible.append((screen["metrics"][profile["metric"]], spec))
                screens.append(entry)
                print("SCREEN " + json.dumps(entry), flush=True)
            outcome = {
                "status": "no_confirmed_gain",
                "baseline": str(base.relative_to(ROOT)),
                "screens": screens,
            }
            for _, spec in sorted(eligible, key=lambda item: item[0]):
                name = spec.get("candidate", spec.get("variant"))
                candidate = run(profile, spec, seed)
                if data(candidate)["status"] != "validated":
                    continue
                initial = compare(
                    base, candidate, budgets.minimum_speedup, budgets.max_memory_growth
                )
                decision(
                    profile["id"] + "/" + name + "/initial",
                    initial,
                    "comparison",
                    {"baseline": base, "candidate": candidate},
                )
                if not initial["eligible"]:
                    continue
                fresh_base = run(profile, baseline_spec, seed + 1)
                fresh_candidate = run(profile, spec, seed + 1)
                if any(
                    data(path)["status"] != "validated" for path in (fresh_base, fresh_candidate)
                ):
                    continue
                result = confirm(
                    base,
                    candidate,
                    fresh_base,
                    fresh_candidate,
                    minimum_speedup=budgets.minimum_speedup,
                    max_memory_growth=budgets.max_memory_growth,
                )
                result = decision(
                    profile["id"] + "/" + name + "/confirmation",
                    result,
                    "confirmation",
                    dict(
                        zip(
                            (
                                "initial_baseline",
                                "initial_candidate",
                                "fresh_baseline",
                                "fresh_candidate",
                            ),
                            (base, candidate, fresh_base, fresh_candidate),
                        )
                    ),
                )
                if result["status"] == "confirmed_improvement":
                    outcome.update(
                        status="confirmed_improvement",
                        selected=spec,
                        confirmation=result["receipt"],
                    )
                    break
            state["cells"][profile["id"]] = outcome
            save()
            print("CELL " + json.dumps(outcome), flush=True)
        state["status"] = (
            "partial_failure"
            if any(cell["status"] == "baseline_failed" for cell in state["cells"].values())
            else "completed"
        )
    except BaseException as error:
        state["status"] = (
            "cancelled"
            if isinstance(error, KeyboardInterrupt)
            else "budget_exhausted"
            if isinstance(error, TimeoutError)
            else "failed"
        )
        state["error"] = str(error)
        state.setdefault("errors", []).append({"status": state["status"], "error": str(error)})
        raise
    finally:
        save()
        print(state_path, flush=True)
    return state


def status(plan_path, state_path):
    """Read atomic checkpoints without claiming a process is still alive."""
    plan = json.loads(plan_path.read_text())
    state = json.loads(state_path.read_text())
    cells = []
    for cell in plan["cells"]:
        result = state["cells"].get(cell["profile"], {})
        cells.append(
            {
                "profile": cell["profile"],
                "status": result.get("status", "pending"),
                "selected": result.get("selected"),
                "confirmation": result.get("confirmation"),
            }
        )
    return {
        "status": state["status"],
        "completed_runs": len(state["runs"]),
        "attempts": len(state["attempts"]),
        "cells": cells,
        "error": state.get("error"),
        "scope": "checkpoint summary, not a process-liveness check",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Read the checkpoint without SDK tools or a simulator lease",
    )
    args = parser.parse_args()
    if not 0 < args.max_seconds < float("inf"):
        parser.error("--max-seconds must be finite and positive")
    try:
        if args.status:
            print(json.dumps(status(args.plan, args.state), indent=2))
            return 0
        with simulator_lease():
            result = campaign(args.plan, args.state, args.max_seconds, args.retry_failed)
        return 0 if result["status"] == "completed" else 1
    except KeyboardInterrupt:
        print("Cancelled; campaign state and run receipts preserved", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
