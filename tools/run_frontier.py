#!/usr/bin/env python3
"""Qualify every correct speed/memory tradeoff with full initial/fresh runs.

Unlike the screen-and-select campaign, this runner does not discard a candidate
because of relative memory growth or cycle gain. Default promotion decisions
are still journaled separately. Run inside the SDK environment, with a finite
budget and an exclusive simulator lease.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from cerebras_research.catalog import get_profile, profile_identity
from cerebras_research.cli import save_decision
from cerebras_research.paths import contained, digest, tree_digest, write_json
from cerebras_research.process import simulator_lease
from cerebras_research.runner import execute
from cerebras_research.scoring import confirm, load_validated
from tools.run_campaign import (
    evaluator_identity,
    install_signal_handlers,
    worker_identity,
    worker_liveness,
)


def campaign(plan_path, state_path, max_seconds, retry_failed=False):
    plan = json.loads(plan_path.read_text())
    names = plan.get("candidates", [])
    cells = plan.get("profiles", [])
    if (
        plan.get("schema_version") != 1
        or plan.get("kind") != "full_fresh_frontier_pilot"
        or not names
        or not cells
        or len(set(names)) != len(names)
        or len({c["profile"] for c in cells}) != len(cells)
        or type(plan.get("seed")) is not int
        or plan["seed"] < 0
    ):
        raise ValueError("Expected distinct candidates/profiles and a nonnegative paired seed")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("A positive finite run budget is required")
    frozen = {
        "plan_sha256": digest(plan_path),
        "evaluator_sha256": evaluator_identity(),
        "candidate_identities": {
            name: {
                "source": tree_digest(contained(ROOT / "experiments/candidates", name)),
                "manifest": digest(
                    contained(ROOT / "experiments/candidates", name) / "candidate.json"
                ),
            }
            for name in names
        },
    }
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else dict(
            schema_version=1,
            **frozen,
            runs={},
            imports={},
            attempts=[],
            cells={},
            points=[],
        )
    )
    if any(state.get(k) != v for k, v in frozen.items()):
        raise ValueError("Plan, evaluator or candidate changed; use a new campaign state")
    for pointer in state["runs"].values():
        if digest(contained(ROOT, pointer["path"])) != pointer["sha256"]:
            raise ValueError("Cached receipt changed")
    deadline = time.monotonic() + max_seconds

    def save():
        write_json(state_path, state)

    def remember(key, path, imported=False):
        pointer = {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
        state["runs"][key] = pointer
        outcome = json.loads(path.read_text())["status"]
        if imported:
            state["imports"][key] = pointer
        else:
            state["attempts"].append(dict(pointer, operation=key, outcome=outcome))
        save()
        return pointer if outcome == "validated" else None

    def run(profile, name, seed, imported=None):
        if time.monotonic() >= deadline:
            raise TimeoutError("Frontier campaign budget exhausted")
        if evaluator_identity() != frozen["evaluator_sha256"]:
            raise ValueError("Evaluator changed during campaign")
        key = f"{profile['id']}/{name}/{seed}"
        if key in state["runs"]:
            pointer = state["runs"][key]
            data = json.loads(contained(ROOT, pointer["path"]).read_text())
            if data["status"] == "validated":
                load_validated(contained(ROOT, pointer["path"]))
                return pointer
            if data["status"] != "cancelled" and not retry_failed:
                return None
        if imported:
            path = contained(ROOT, imported["path"])
            if digest(path) != imported["sha256"]:
                raise ValueError("Imported baseline receipt changed")
            data = load_validated(path)
            if (
                data["variant"] != "baseline"
                or data["seed"] != seed
                or data["profile_sha256"] != profile_identity(profile)
                or data.get("provenance", {}).get("evaluator_sha256") != frozen["evaluator_sha256"]
            ):
                raise ValueError("Imported baseline is not matched")
            return remember(key, path, imported=True)
        candidate = None if name == "baseline" else contained(ROOT / "experiments/candidates", name)
        if candidate and frozen["candidate_identities"][name] != {
            "source": tree_digest(candidate),
            "manifest": digest(candidate / "candidate.json"),
        }:
            raise ValueError("Candidate changed during campaign")
        state["current_operation"] = key
        save()
        before = set((ROOT / "runs/v2").glob("*/result.json"))
        try:
            path = execute(
                profile, "candidate" if candidate else "baseline", seed, False, candidate, deadline
            )
        except BaseException as error:
            created = set((ROOT / "runs/v2").glob("*/result.json")) - before
            if len(created) != 1:
                raise
            path = created.pop()
            outcome = json.loads(path.read_text())["status"]
            remember(key, path)
            if not isinstance(error, (RuntimeError, ValueError, OSError)) or outcome not in (
                "failed",
                "cancelled",
            ):
                raise
            return None
        return remember(key, path)

    state.update(status="running", worker=worker_identity())
    state.pop("error", None)
    save()
    budgets = argparse.Namespace(minimum_speedup=1.05, max_memory_growth=1.10)
    try:
        for cell in cells:
            profile = get_profile(cell["profile"])
            if profile["validation"] != "independent_six_case":
                raise ValueError("Frontier qualification requires an independent six-case profile")
            bases = [
                run(profile, "baseline", plan["seed"] + offset, cell.get(field))
                for offset, field in enumerate(("initial_baseline", "fresh_baseline"))
            ]
            if any(p is None for p in bases):
                raise RuntimeError("Baseline incomplete; inspect its preserved receipt")
            for name in names:
                key = profile["id"] + "/" + name
                prior = state["cells"].get(key)
                if prior and (prior["status"] == "fresh_paired_qualified" or not retry_failed):
                    continue
                first = run(profile, name, plan["seed"])
                fresh = run(profile, name, plan["seed"] + 1) if first else None
                if not first or not fresh:
                    state["cells"][key] = {"status": "qualification_failed"}
                    save()
                    continue
                arms = dict(
                    zip(
                        (
                            "initial_baseline",
                            "initial_candidate",
                            "fresh_baseline",
                            "fresh_candidate",
                        ),
                        (bases[0], first, bases[1], fresh),
                    )
                )
                paths = {a: contained(ROOT, p["path"]) for a, p in arms.items()}
                commands = [load_validated(path)["compile_command"] for path in paths.values()]
                if any(command != commands[0] for command in commands[1:]):
                    raise ValueError("Frontier point changed compiler arguments")
                decision = confirm(*paths.values())
                saved = save_decision(decision, "confirmation", paths, budgets)
                state["points"].append(dict(label=name, **arms))
                state["cells"][key] = {
                    "status": "fresh_paired_qualified",
                    "default_policy_status": saved["status"],
                    "default_policy_receipt": saved["receipt"],
                }
                save()
        state["status"] = (
            "completed"
            if all(c["status"] == "fresh_paired_qualified" for c in state["cells"].values())
            else "completed_with_failures"
        )
    except BaseException as error:
        state.update(
            status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
        )
        raise
    finally:
        save()
    return state


def main():
    install_signal_handlers()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--export-prefix", type=Path)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        state = json.loads(args.state.read_text())
        print(
            json.dumps(
                {
                    "saved_status": state["status"],
                    "worker_liveness": worker_liveness(state.get("worker")),
                    "plan_matches": digest(args.plan) == state["plan_sha256"],
                    "qualified_points": len(state["points"]),
                    "current_operation": state.get("current_operation"),
                }
            )
        )
        return
    if sys.platform != "linux":
        parser.error("Run the writer through tools/sdk-run in the Linux journal environment")
    if args.max_seconds is None or args.max_seconds <= 0:
        parser.error("A positive --max-seconds budget is required")
    with simulator_lease():
        state = campaign(args.plan, args.state, args.max_seconds, args.retry_failed)
    # Publish immutable input snapshots; never point a report at a changing checkpoint.
    for suffix, data in (
        (".pairs.json", {"schema_version": 1, "points": state["points"]}),
        (
            ".full.json",
            {
                "schema_version": 1,
                "receipts": [
                    p
                    for p in state["runs"].values()
                    if json.loads(contained(ROOT, p["path"]).read_text())["status"] == "validated"
                ],
            },
        ),
    ):
        output = (args.export_prefix or args.state).with_suffix(suffix)
        if output.exists() and json.loads(output.read_text()) != data:
            raise ValueError("Existing manifest differs; choose a new --export-prefix")
        if not output.exists():
            write_json(output, data)
    print(json.dumps({"status": state["status"], "qualified_points": len(state["points"])}))


if __name__ == "__main__":
    main()
