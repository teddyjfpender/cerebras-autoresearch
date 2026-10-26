#!/usr/bin/env python3
"""Describe raw-array differences for matched full runs; never grants retention.

Run inside SDK NumPy. Manifest: {schema_version:1,pairs:[{baseline:{path,sha256},
candidate:{path,sha256}}]}. Explicitly name every requested array with --keys;
--bf16-keys identifies raw uint16 BF16 words rather than ordinary integer data.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from cerebras_research.paths import contained, digest, write_json
from cerebras_research.scoring import load_validated


def array_difference(a, b, bf16=False):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("Paired array shape/precision differs")
    av, bv = (np.ascontiguousarray(v) for v in (a, b))
    byte_a, byte_b = (v.reshape(-1).view(np.uint8).reshape(-1, v.dtype.itemsize) for v in (av, bv))
    changed = int(np.any(byte_a != byte_b, axis=1).sum())
    if bf16:
        if av.dtype != np.uint16:
            raise ValueError("BF16 bit arrays must use uint16 storage")
        av, bv = ((v.astype(np.uint32) << 16).view(np.float32) for v in (av, bv))
    if not np.isfinite(av).all() or not np.isfinite(bv).all():
        raise ValueError("Nonfinite numeric evidence")
    return dict(
        dtype=str(a.dtype),
        interpretation="BF16 raw words" if bf16 else "stored numeric dtype",
        shape=list(a.shape),
        element_count=a.size,
        changed_elements_by_bits=changed,
        identical_bits=changed == 0,
        max_absolute_numeric_difference=float(
            np.max(np.abs(av.astype(np.float64) - bv.astype(np.float64)))
        ),
    )


def audit(manifest, keys, bf16_keys):
    if manifest.get("schema_version") != 1 or not manifest.get("pairs"):
        raise ValueError("Expected nonempty schema-version-1 pair manifest")
    if not keys or len(keys) != len(set(keys)) or not set(bf16_keys) <= set(keys):
        raise ValueError("Require distinct explicit array keys and a BF16 subset")
    records = []
    seen = set()
    for pair in manifest["pairs"]:
        paths, runs = [], []
        for arm in ("baseline", "candidate"):
            pointer = pair[arm]
            path = contained(ROOT, pointer["path"])
            if digest(path) != pointer["sha256"]:
                raise ValueError("Receipt changed")
            paths.append(path)
            runs.append(load_validated(path))
        if tuple(paths) in seen or paths[0] == paths[1]:
            raise ValueError("Duplicate or self pair")
        seen.add(tuple(paths))
        before, after = runs
        for name in (
            "profile",
            "profile_sha256",
            "seed",
            "sdk",
            "architecture",
            "simulator_threads",
            "compile_command",
            "harness_sha256",
            "baseline_tree_sha256",
            "validation",
        ):
            if before[name] != after[name]:
                raise ValueError(f"Unmatched pair: {name}")
        if (
            before["variant"] != "baseline"
            or before["provenance"]["evaluator_sha256"] != after["provenance"]["evaluator_sha256"]
        ):
            raise ValueError("Require matching evaluator and a baseline first arm")
        cases = []
        for left, right in zip(before["cases"], after["cases"]):
            if left["case"] != right["case"]:
                raise ValueError("Numerical families differ")
            artifacts = [
                path.parent / "work/out" / case["artifact"]
                for path, case in zip(paths, (left, right))
            ]
            for path, case in zip(artifacts, (left, right)):
                if digest(path) != case["artifact_sha256"]:
                    raise ValueError("Raw artifact changed")
            with (
                np.load(artifacts[0], allow_pickle=False) as a,
                np.load(artifacts[1], allow_pickle=False) as b,
            ):
                differences = {
                    key: array_difference(a[key], b[key], key in bf16_keys) for key in keys
                }
            cases.append(
                dict(
                    case=left["case"],
                    arrays=differences,
                    artifacts_sha256=[left["artifact_sha256"], right["artifact_sha256"]],
                )
            )
        if len(cases) != 6 or len(before["cases"]) != len(after["cases"]):
            raise ValueError("Require complete paired six-case coverage")
        records.append(
            dict(
                profile=before["profile"]["id"],
                seed=before["seed"],
                receipts=pair,
                evaluator_sha256=before["provenance"]["evaluator_sha256"],
                cases=cases,
            )
        )
    return dict(
        schema_version=1,
        status="matched_raw_arrays_compared",
        case_pairs=sum(len(r["cases"]) for r in records),
        tool_sha256=digest(Path(__file__)),
        records=records,
        scope="Descriptive bit/numeric differences on these paired arrays; not a new oracle, promotion or equivalent-work proof.",
        qwen_tps_result=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--bf16-keys", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refuse to overwrite an array report")
    report = audit(json.loads(args.manifest.read_text()), args.keys, args.bf16_keys)
    write_json(args.output, report)
    print(
        json.dumps(
            dict(status=report["status"], case_pairs=report["case_pairs"], output=str(args.output))
        )
    )


if __name__ == "__main__":
    main()
