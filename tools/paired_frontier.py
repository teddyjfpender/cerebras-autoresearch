#!/usr/bin/env python3
"""Export a fresh-paired cycle/memory frontier independently of retention policy.

Inputs are four-arm, digest-bound full-run manifests. This derived report does
not promote candidates or write journal rows. Ordinary runs/decisions retain
their existing journal and scoring semantics.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cerebras_research.paths import contained, digest, identity, write_json
from cerebras_research.scoring import confirm, load_validated

ARMS = ("initial_baseline", "initial_candidate", "fresh_baseline", "fresh_candidate")


def mark_frontier(points):
    """Keep incomparable tracks separate; exact cost ties both remain on the front."""
    return [
        dict(
            point,
            pareto_optimal=not any(
                other["track_sha256"] == point["track_sha256"]
                and other["cycles"] <= point["cycles"]
                and other["memory_bytes_per_pe"] <= point["memory_bytes_per_pe"]
                and (
                    other["cycles"] < point["cycles"]
                    or other["memory_bytes_per_pe"] < point["memory_bytes_per_pe"]
                )
                for other in points
            ),
        )
        for point in points
    ]


def build_report(manifest):
    if manifest.get("schema_version") != 1 or not manifest.get("points"):
        raise ValueError("Expected nonempty schema-version-1 points manifest")
    points, baselines, seen = [], {}, set()
    for item in manifest["points"]:
        paths, runs = {}, {}
        for arm in ARMS:
            pointer = item[arm]
            path = contained(ROOT, pointer["path"])
            if digest(path) != pointer["sha256"]:
                raise ValueError("Receipt digest changed")
            paths[arm], runs[arm] = path, load_validated(path)
        key = tuple(paths.values())
        if key in seen:
            raise ValueError("Duplicate four-arm point")
        seen.add(key)
        # This performs all matched-input, provenance and fresh-source checks.
        # A policy rejection does not invalidate a correct, fresh-paired point.
        policy = confirm(*(paths[arm] for arm in ARMS))
        base, candidate = runs["initial_baseline"], runs["initial_candidate"]
        if any(runs[arm]["compile_command"] != base["compile_command"] for arm in ARMS):
            raise ValueError("Frontier point changed compiler arguments")
        metric = base["profile"]["metric"]
        track = {
            "profile_sha256": base["profile_sha256"],
            "baseline_source_sha256": base["source_sha256"],
            "baseline_tree_sha256": base["baseline_tree_sha256"],
            "harness_sha256": base["harness_sha256"],
            "sdk": base["sdk"],
            "evaluator_sha256": base["provenance"]["evaluator_sha256"],
            "architecture": base["architecture"],
            "simulator_threads": base["simulator_threads"],
            "compile_command": base["compile_command"],
            "paired_seeds": [base["seed"], runs["fresh_baseline"]["seed"]],
        }
        track_id = identity(track)

        def costs(first, fresh):
            result = {
                "cycles": max(runs[a]["metrics"][metric] for a in (first, fresh)),
                "memory_bytes_per_pe": max(
                    runs[a]["metrics"]["max_program_object_bytes_per_pe"] for a in (first, fresh)
                ),
                "per_seed": [
                    {
                        "seed": runs[a]["seed"],
                        "cycles": runs[a]["metrics"][metric],
                        "memory_bytes_per_pe": runs[a]["metrics"][
                            "max_program_object_bytes_per_pe"
                        ],
                    }
                    for a in (first, fresh)
                ],
                "max_fraction_of_error_bound": max(
                    case["max_fraction_of_error_bound"]
                    for arm in (first, fresh)
                    for case in runs[arm]["cases"]
                ),
            }
            work = base["profile"].get("work_units")
            if work is not None:
                result["work_units_per_cycle"] = work / result["cycles"]
                result["work_unit"] = base["profile"]["work_unit"]
            return result

        baseline = costs("initial_baseline", "fresh_baseline")
        if track_id in baselines and baselines[track_id]["per_seed"] != baseline["per_seed"]:
            raise ValueError("Frontier points have different baseline measurements")
        baselines[track_id] = dict(
            profile=base["profile"]["id"],
            track_sha256=track_id,
            track=track,
            profile_contract=base["profile"],
            candidate="baseline",
            qualification="full_six_case_two_seed_baseline",
            metric=metric,
            conservative_speedup=1.0,
            memory_ratio=1.0,
            **baseline,
        )
        points.append(
            dict(
                profile=base["profile"]["id"],
                track_sha256=track_id,
                candidate=item.get("label", candidate.get("candidate", "candidate")),
                source_sha256=candidate["source_sha256"],
                qualification="matched_full_six_case_fresh_pair",
                metric=metric,
                conservative_speedup=min(policy[a]["speedup"] for a in ("initial", "confirmation")),
                memory_ratio=max(policy[a]["memory_ratio"] for a in ("initial", "confirmation")),
                default_policy_eligible=policy["status"] == "confirmed_improvement",
                default_policy=policy,
                receipts={arm: item[arm] for arm in ARMS},
                **costs("initial_candidate", "fresh_candidate"),
            )
        )
    return {
        "schema_version": 1,
        "status": "fresh_paired_frontier_report",
        "scope": "Fixed-contract simulator cycles and program-object bytes per PE; no hardware/model TPS.",
        "qualification": "Full six-case runs at both paired seeds, artifact/source checks and matched evaluator. Independent raw-array replay is a separate report.",
        "dominance": "Within the same track: no worse on both conservative cycles and memory, strictly better on at least one. No scalar weighting or memory-growth eligibility cutoff.",
        "points": mark_frontier([*baselines.values(), *points]),
        "qwen_tps_result": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing evidence report")
    report = build_report(json.loads(args.manifest.read_text()))
    report["manifest_sha256"] = digest(args.manifest)
    report["reporter_sha256"] = digest(__file__)
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "points": len(report["points"])}))


if __name__ == "__main__":
    main()
