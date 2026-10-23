"""Repository CLI for catalogs, bounded experiments, and evidence-based promotion."""

import argparse
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

from . import ledger, legacy
from .adapters import candidate_init
from .catalog import get_profile, get_suite, load_catalog
from .paths import ROOT, contained, digest, write_json
from .process import simulator_lease
from .runner import execute
from .scoring import aggregate, compare, confirm


def dump(value):
    print(json.dumps(value, indent=2, allow_nan=False))


def candidate_path(name):
    return contained(ROOT / "experiments/candidates", name) if name else None


def save_decision(report, kind, inputs, args):
    """Persist a reviewable decision before appending its journal event."""
    folder = (
        ROOT
        / "runs/v2"
        / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + kind + "-" + uuid.uuid4().hex[:8])
    )
    report.update(
        schema_version=2,
        record_type=kind,
        inputs={
            key: str(contained(ROOT, value).relative_to(ROOT)) for key, value in inputs.items()
        },
        budgets={
            key: getattr(args, key)
            for key in ("minimum_speedup", "max_memory_growth", "regression_limit")
            if hasattr(args, key) and (key != "regression_limit" or "suite" in kind)
        },
    )
    path = folder / "result.json"
    write_json(path, report)
    ledger.append_receipts([path])
    return dict(report, receipt=str(path.relative_to(ROOT)))


def challenge(profile, args, deadline):
    if profile["validation"] != "independent_six_case" or args.variant == "baseline":
        raise ValueError("Challenge requires a candidate and an independent six-case oracle")
    folder = (
        ROOT
        / "runs/v2"
        / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-challenge-" + uuid.uuid4().hex[:8])
    )
    report = {
        "schema_version": 2,
        "status": "running",
        "profile": profile["id"],
        "runs": [],
        "qwen_tps_result": False,
    }

    def run(variant, seed, screen=False):
        result = execute(
            profile,
            variant,
            seed,
            screen,
            candidate_path(args.candidate) if variant == "candidate" else None,
            deadline,
        )
        report["runs"].append(str(result.relative_to(ROOT)))
        write_json(folder / "result.json", report)
        return result

    try:
        base = run("baseline", args.seed)
        screen = run(args.variant, args.seed, True)
        b, s = (json.loads(p.read_text()) for p in (base, screen))
        if b["metrics"][profile["metric"]] / s["metrics"][profile["metric"]] < args.minimum_speedup:
            report["status"] = "rejected_at_screen"
        else:
            cand = run(args.variant, args.seed)
            budgets = {
                "minimum_speedup": args.minimum_speedup,
                "max_memory_growth": args.max_memory_growth,
            }
            report["initial"] = compare(base, cand, **budgets)
            if not report["initial"]["eligible"]:
                report["status"] = "rejected"
            else:
                fresh_base = run("baseline", args.seed + 1)
                fresh_cand = run(args.variant, args.seed + 1)
                report.update(confirm(base, cand, fresh_base, fresh_cand, **budgets))
    except BaseException as error:
        report["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = str(error)
        raise
    finally:
        write_json(folder / "result.json", report)
        print(folder / "result.json", flush=True)
        ledger.append_receipts([folder / "result.json"])
    return report


def suite_run(profiles, args, deadline):
    unsupported = [p["id"] for p in profiles if args.variant not in p["variants"]]
    if unsupported:
        raise ValueError(f"Variant unsupported by suite members: {unsupported}")
    folder = (
        ROOT
        / "runs/v2"
        / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-suite-" + uuid.uuid4().hex[:8])
    )
    report = {
        "schema_version": 2,
        "status": "running",
        "profiles": [p["id"] for p in profiles],
        "seed": args.seed,
        "variant": args.variant,
        "screen_only": args.screen,
        "runs": {},
        "failures": {},
        "qwen_tps_result": False,
    }
    try:
        for profile in profiles:
            if time.monotonic() >= deadline:
                report["status"] = "budget_exhausted"
                break
            try:
                path = execute(
                    profile,
                    args.variant,
                    args.seed,
                    args.screen,
                    candidate_path(args.candidate),
                    deadline,
                )
                report["runs"][profile["id"]] = str(path.relative_to(ROOT))
            except (RuntimeError, ValueError, OSError) as error:
                report["failures"][profile["id"]] = str(error)
            write_json(folder / "result.json", report)
        else:
            report["status"] = "partial_failure" if report["failures"] else "completed"
    except BaseException as error:
        report["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = str(error)
        raise
    finally:
        write_json(folder / "result.json", report)
        print(folder / "result.json", flush=True)
        ledger.append_receipts([folder / "result.json"])
    return report


def compare_suites(base_path, cand_path, args):
    base, candidate = (json.loads(Path(p).read_text()) for p in (base_path, cand_path))
    if base["status"] != "completed" or candidate["status"] != "completed":
        raise ValueError("Suite comparison requires completed coverage")
    if base["profiles"] != candidate["profiles"]:
        raise ValueError("Suite profile order/coverage changed")
    if any(set(receipt["runs"]) != set(receipt["profiles"]) for receipt in (base, candidate)):
        raise ValueError("Suite run coverage is incomplete")
    pairs = [
        compare(
            contained(ROOT, base["runs"][key]),
            contained(ROOT, candidate["runs"][key]),
            args.minimum_speedup,
            args.max_memory_growth,
        )
        for key in base["profiles"]
    ]
    return dict(
        aggregate(pairs, base["profiles"], args.minimum_speedup, args.regression_limit),
        comparisons=pairs,
    )


def confirm_suites(initial_base, initial_candidate, fresh_base, fresh_candidate, args):
    """Confirm a complete suite, including source identity and every cell's new seed."""
    initial = compare_suites(initial_base, initial_candidate, args)
    fresh = compare_suites(fresh_base, fresh_candidate, args)
    if initial["profiles"] != fresh["profiles"]:
        raise ValueError("Fresh suite coverage changed")
    receipts = [
        json.loads(Path(p).read_text())
        for p in (initial_base, initial_candidate, fresh_base, fresh_candidate)
    ]
    evidence = {}
    for name in initial["profiles"]:
        paths = [contained(ROOT, receipt["runs"][name]) for receipt in receipts]
        cell = confirm(
            *paths, minimum_speedup=args.minimum_speedup, max_memory_growth=args.max_memory_growth
        )
        # Per-cell 5% gain is not required for a suite mean; fresh identity,
        # numerical validity, memory, and worst-cell regression guards are.
        evidence[name] = {
            "fresh_identity_verified": True,
            "initial_speedup": cell["initial"]["speedup"],
            "confirmation_speedup": cell["confirmation"]["speedup"],
        }
    return {
        "status": "confirmed_suite_improvement"
        if initial["eligible"] and fresh["eligible"]
        else "rejected",
        "initial": initial,
        "confirmation": fresh,
        "cell_evidence": evidence,
        "scope": "exact complete suite coverage",
        "qwen_tps_result": False,
    }


def export_report(output):
    records = []
    for path in sorted((ROOT / "runs/v2").glob("*/result.json")):
        data = json.loads(path.read_text())
        records.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": digest(path),
                **{
                    k: data[k]
                    for k in (
                        "status",
                        "id",
                        "variant",
                        "seed",
                        "screen_only",
                        "profile_sha256",
                        "source_sha256",
                        "harness_sha256",
                        "metrics",
                        "error",
                    )
                    if k in data
                },
                "profile_id": data.get("profile", {}).get("id")
                if isinstance(data.get("profile"), dict)
                else data.get("profile"),
                "scope": "local simulator evidence",
            }
        )
    write_json(output, {"schema_version": 2, "qwen_tps_result": False, "records": records})
    return {"output": str(output), "records": len(records)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="Show profiles or suites without requiring SDK")
    listing.add_argument("--json", action="store_true")
    listing.add_argument("--suite")
    commands.add_parser("validate", help="Validate catalog, suite membership, and local JSON")
    commands.add_parser("fetch", help="Fetch and verify pinned upstream sources")
    doctor = commands.add_parser("doctor", help="Check source/toolchain availability")
    doctor.add_argument(
        "--verify-sdk",
        action="store_true",
        help="Hash the installed SIF; can take minutes under emulation",
    )
    init = commands.add_parser(
        "candidate-init", help="Create an editable CSL-only candidate workspace"
    )
    init.add_argument("profile")
    init.add_argument("name")
    init.add_argument("--start-variant", default="baseline")
    for name in ("run", "suite", "challenge"):
        cmd = commands.add_parser(name)
        cmd.add_argument("target", help="Profile ID; suite name for 'suite'")
        cmd.add_argument("--variant", default="baseline")
        cmd.add_argument("--candidate")
        cmd.add_argument("--seed", type=int, default=20261022)
        cmd.add_argument("--max-seconds", type=float, default=3600)
        if name != "challenge":
            cmd.add_argument("--screen", action="store_true")
        else:
            cmd.add_argument("--minimum-speedup", type=float, default=1.05)
            cmd.add_argument("--max-memory-growth", type=float, default=1.10)
    for name in ("compare", "compare-suite"):
        cmd = commands.add_parser(name)
        cmd.add_argument("baseline")
        cmd.add_argument("candidate")
        cmd.add_argument("--minimum-speedup", type=float, default=1.05)
        cmd.add_argument("--max-memory-growth", type=float, default=1.10)
        cmd.add_argument("--regression-limit", type=float, default=1.02)
    confirm_command = commands.add_parser("confirm-suite", help="Validate fresh suite confirmation")
    for name in ("initial_baseline", "initial_candidate", "fresh_baseline", "fresh_candidate"):
        confirm_command.add_argument(name)
    confirm_command.add_argument("--minimum-speedup", type=float, default=1.05)
    confirm_command.add_argument("--max-memory-growth", type=float, default=1.10)
    confirm_command.add_argument("--regression-limit", type=float, default=1.02)
    single_confirm = commands.add_parser(
        "confirm", help="Replay and record a fresh matched profile pair"
    )
    for name in ("initial_baseline", "initial_candidate", "fresh_baseline", "fresh_candidate"):
        single_confirm.add_argument(name)
    single_confirm.add_argument("--minimum-speedup", type=float, default=1.05)
    single_confirm.add_argument("--max-memory-growth", type=float, default=1.10)
    journal = commands.add_parser("ledger", help="Append, verify, and inspect the research TSV")
    journal.add_argument("action", choices=("sync", "verify", "board"))
    journal.add_argument("--artifacts", action="store_true")
    journal.add_argument("--base-ref", help="Require exact TSV prefix from this Git commit")
    exp = commands.add_parser(
        "export", help="Export compact receipts; preserve raw evidence in runs/"
    )
    exp.add_argument("output", type=Path)
    commands.add_parser("storage", help="Report generated artifact size; does not delete evidence")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        for key, lower, strict in (
            ("max_seconds", 0, True),
            ("minimum_speedup", 1, True),
            ("max_memory_growth", 1, False),
            ("regression_limit", 1, False),
        ):
            if hasattr(args, key):
                value = getattr(args, key)
                if not math.isfinite(value) or value < lower or (strict and value == lower):
                    raise ValueError(f"Invalid --{key.replace('_', '-')}")
        if args.command == "list":
            profiles = get_suite(args.suite) if args.suite else list(load_catalog().values())
            if args.json:
                dump(profiles)
            else:
                for item in profiles:
                    print(
                        f"{item['id']:38s} {item['family']:18s} {item['precision']:13s} {item['validation']:22s} {item['metric'] or 'correctness only'}"
                    )
            return 0
        if args.command == "validate":
            profiles = load_catalog()
            for path in (ROOT / "config").rglob("*.json"):
                json.loads(path.read_text())
            dump(
                {
                    "profiles": len(profiles),
                    "suites": len(list((ROOT / "benchmarks/suites").glob("*.json"))),
                    "status": "valid",
                }
            )
            return 0
        if args.command == "fetch":
            legacy.fetch()
            return 0
        if args.command == "doctor":
            info = legacy.preflight()
            if args.verify_sdk:
                manifest = json.loads((ROOT / "config/sdk-install.json").read_text())
                image = (
                    Path(os.environ.get("CEREBRAS_SDK_DIR", ROOT / manifest["local_sdk_directory"]))
                    / manifest["sif"]
                )
                info["sdk_digest_verified"] = digest(image) == manifest["sif_sha256"]
                if not info["sdk_digest_verified"]:
                    raise ValueError("SDK SIF digest mismatch")
            dump(info)
            return 0 if info["ready_for_smoke"] else 2
        if args.command == "candidate-init":
            print(candidate_init(get_profile(args.profile), args.name, args.start_variant))
            return 0
        if args.command == "compare":
            result = compare(
                args.baseline, args.candidate, args.minimum_speedup, args.max_memory_growth
            )
            dump(
                save_decision(
                    result,
                    "comparison",
                    {"baseline": args.baseline, "candidate": args.candidate},
                    args,
                )
            )
            return 0
        if args.command == "compare-suite":
            dump(
                save_decision(
                    compare_suites(args.baseline, args.candidate, args),
                    "suite_comparison",
                    {"baseline": args.baseline, "candidate": args.candidate},
                    args,
                )
            )
            return 0
        if args.command in ("confirm", "confirm-suite"):
            paths = (
                args.initial_baseline,
                args.initial_candidate,
                args.fresh_baseline,
                args.fresh_candidate,
            )
            if args.command == "confirm-suite":
                result = confirm_suites(
                    args.initial_baseline,
                    args.initial_candidate,
                    args.fresh_baseline,
                    args.fresh_candidate,
                    args,
                )
                kind = "suite_confirmation"
            else:
                result = confirm(
                    *paths,
                    minimum_speedup=args.minimum_speedup,
                    max_memory_growth=args.max_memory_growth,
                )
                kind = "confirmation"
            dump(
                save_decision(
                    result,
                    kind,
                    dict(
                        zip(
                            (
                                "initial_baseline",
                                "initial_candidate",
                                "fresh_baseline",
                                "fresh_candidate",
                            ),
                            paths,
                        )
                    ),
                    args,
                )
            )
            return 0
        if args.command == "ledger":
            if args.action == "sync":
                dump(
                    ledger.append_receipts(
                        sorted((ROOT / "runs/v2").glob("*/result.json")), "historical_import"
                    )
                )
            elif args.action == "verify":
                dump(ledger.verify(args.artifacts, args.base_ref))
            else:
                with ledger.lock():
                    result = ledger.board(ledger.read_rows())
                    write_json(ROOT / "results/leaderboard.json", result)
                dump(result)
            return 0
        if args.command == "export":
            dump(export_report(args.output))
            return 0
        if args.command == "storage":
            dump(
                {
                    "generated_bytes": sum(
                        p.stat().st_size for p in (ROOT / "runs").rglob("*") if p.is_file()
                    ),
                    "policy": "Raw evidence retained; archive before manual deletion",
                }
            )
            return 0
        if args.max_seconds <= 0:
            raise ValueError("--max-seconds must be positive")
        deadline = time.monotonic() + args.max_seconds
        with simulator_lease():
            if args.command == "run":
                execute(
                    get_profile(args.target),
                    args.variant,
                    args.seed,
                    args.screen,
                    candidate_path(args.candidate),
                    deadline,
                )
                return 0
            if args.command == "suite":
                return (
                    0
                    if suite_run(get_suite(args.target), args, deadline)["status"] == "completed"
                    else 1
                )
            return (
                0
                if challenge(get_profile(args.target), args, deadline)["status"] != "failed"
                else 1
            )
    except KeyboardInterrupt:
        print("Cancelled; receipt and partial evidence preserved", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, TypeError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
