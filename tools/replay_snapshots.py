#!/usr/bin/env python3
"""Corroborate resident outputs, immutable inputs, boundary state and timers against SDK cores.

Run through tools/sdk-run cs_python. This launches no device and grants no score.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))

import numpy as np
from snapshot import read_core_arrays
from snapshot_grid import read_core_grid_arrays

from cerebras_research.paths import contained, digest, write_json
from cerebras_research.scoring import load_validated


def equal_bits(actual, saved):
    if (
        actual.shape != saved.shape
        or actual.dtype != saved.dtype
        or actual.tobytes() != saved.tobytes()
    ):
        raise ValueError("Saved evidence disagrees with complete simulator SRAM")


def unpack(raw, p, mt, nt, t, dtype):
    return (
        raw.view(dtype).reshape(p, p, t, nt, mt).transpose(2, 0, 4, 1, 3).reshape(t, p * mt, p * nt)
    )


def arrays_contract(profile):
    dims = profile["dimensions"]
    p, t, mt = profile["pe_grid"][0], dims["T"], dims["M"] // 4
    if profile["adapter"] == "reusable_gemm":
        kt, nt = dims["K"] // p, dims["N"] // p
        expected = dict(ALL_OUT=t * mt * nt, X_FRAMES=t * mt * kt, W_CANON=kt * nt)
        canonical = dict(X_FRAMES="canonical_x_after", W_CANON="canonical_w_after")
        outputs = dict(ALL_OUT=("actual", nt, np.float16, np.float64))
        weight_count = kt * nt
    elif profile["adapter"] == "resident_swiglu":
        ht, ft = dims["H"] // p, dims["F"] // p
        expected = dict(X_FRAMES=t * mt * ht, WG_CANON=ht * ft, WU_CANON=ht * ft, WD_CANON=ht * ft)
        canonical = {name: "canonical_" + name for name in expected}
        outputs = {
            f"{k.upper()}_HISTORY": (
                "stage_" + k,
                ft if k in "guv" else ht,
                np.float32 if k == "y" else np.float16,
                np.float32 if k == "y" else np.float16,
            )
            for k in "guvdy"
        }
        expected.update(
            {
                name: t * mt * n * (2 if dtype == np.float32 else 1)
                for name, (_, n, dtype, _) in outputs.items()
            }
        )
        weight_count = ht * ft
    elif profile["adapter"] == "resident_swiglu_bf16":
        ht, ft = dims["H"] // p, dims["F"] // p
        expected = dict(X_FRAMES=t * mt * ht, WG_CANON=ht * ft, WU_CANON=ht * ft, WD_CANON=ht * ft)
        canonical = {name: "canonical_" + name for name in expected}
        outputs = {
            f"{k.upper()}_HISTORY": ("bits_" + k, ft if k in "guv" else ht, np.uint16, np.uint16)
            for k in "guvdy"
        }
        expected.update({name: t * mt * n for name, (_, n, _, _) in outputs.items()})
        weight_count = ht * ft
    else:
        raise ValueError("Unsupported full resident snapshot contract")
    expected.update({f"W_HISTORY_{i}": weight_count for i in range(t)})
    expected["time_buf_f32"] = 6
    return p, t, mt, weight_count, expected, canonical, outputs


def audit_recurrent_case(data, path, case):
    from recurrent_gdn import capture_contract, decode_capture, output_shapes, pack_timer

    dims = data["profile"]["dimensions"]
    k, v = dims["K"], dims["V"]
    core = path.parent / f"work/out/case-{case['case']}.core"
    if data["artifacts"].get(str(core.relative_to(path.parent))) != digest(core):
        raise ValueError("Simulator core is missing or changed")
    artifact = path.parent / "work/out" / case["artifact"]
    if digest(artifact) != case["artifact_sha256"]:
        raise ValueError("Saved recurrent artifact changed")
    with np.load(artifact, allow_pickle=False) as raw:
        names = ("Q_CANON", "K_CANON", "V_CANON", "G_CANON", "BETA_CANON", "S0")
        inputs = {name: raw["input_" + name] for name in names}
        shapes, expected = capture_contract(k, v, inputs)
        captured = decode_capture(
            read_core_grid_arrays(core, path.parent / "work/out/bin", 4, 4, expected), shapes
        )
        for name in names:
            equal_bits(captured[name], raw["canonical_" + name])
            equal_bits(captured[name], inputs[name])
        for name in output_shapes(k, v):
            equal_bits(captured[name], raw["FINAL_STATE" if name == "STATE" else name])
        equal_bits(captured["times"], raw["timer_core_u16"])
        equal_bits(captured["times"], raw["timer_memcpy_u16"])
        equal_bits(pack_timer(captured["times"]), raw["timer_u32"])
    return dict(
        case=case["case"],
        complete_output_state_and_timer_match=True,
        core_sha256=digest(core),
        half_words_checked=16 * sum(expected.values()),
    )


def audit(manifest):
    pointers = manifest.get("receipts", [])
    if manifest.get("schema_version") != 1 or not pointers:
        raise ValueError("Expected a nonempty schema-version-1 receipt manifest")
    paths = [contained(ROOT, item["path"]) for item in pointers]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate audit receipt")
    records = []
    for pointer, path in zip(pointers, paths):
        if digest(path) != pointer["sha256"]:
            raise ValueError("Selected receipt changed")
        data = load_validated(path)
        if data["harness_files"].get("snapshot.py") != digest(
            ROOT / "benchmarks/drivers/snapshot.py"
        ):
            raise ValueError("Snapshot helper differs from the run's captured source")
        if data["profile"]["adapter"] == "recurrent_gdn":
            if data["harness_files"].get("snapshot_grid.py") != digest(
                ROOT / "benchmarks/drivers/snapshot_grid.py"
            ):
                raise ValueError("Grid snapshot helper differs from the captured source")
            cases = [audit_recurrent_case(data, path, case) for case in data["cases"]]
            records.append(
                dict(
                    receipt=pointer,
                    profile=data["profile"]["id"],
                    source_sha256=data["source_sha256"],
                    harness_sha256=data["harness_sha256"],
                    evaluator_sha256=data["provenance"]["evaluator_sha256"],
                    cases=cases,
                )
            )
            continue
        p, t, mt, wc, expected, canonical, outputs = arrays_contract(data["profile"])
        cases = []
        for case in data["cases"]:
            core = path.parent / f"work/out/case-{case['case']}.core"
            relative = str(core.relative_to(path.parent))
            if data["artifacts"].get(relative) != digest(core):
                raise ValueError("Simulator core is missing or changed")
            captured = read_core_arrays(core, path.parent / "work/out/bin", p, expected)
            if data["profile"]["adapter"] == "resident_swiglu_bf16":
                captured = {name: value.view(np.uint16) for name, value in captured.items()}
            history = np.stack(
                [captured[f"W_HISTORY_{i}"].reshape(p, p, wc) for i in range(t)], axis=2
            )
            artifact = path.parent / "work/out" / case["artifact"]
            if digest(artifact) != case["artifact_sha256"]:
                raise ValueError("Saved numerical artifact changed")
            with np.load(artifact, allow_pickle=False) as raw:
                for symbol, saved in canonical.items():
                    equal_bits(captured[symbol], raw[saved])
                equal_bits(history, raw["working_w_history"])
                for symbol, (saved, n, dtype, stored_dtype) in outputs.items():
                    equal_bits(
                        unpack(captured[symbol], p, mt, n, t, dtype).astype(stored_dtype),
                        raw[saved],
                    )
                timer = captured["time_buf_f32"].view(np.uint16).view(np.uint32).ravel()
                equal_bits(timer, raw["timer_u32"].ravel())
            cases.append(
                dict(
                    case=case["case"],
                    complete_output_state_and_timer_match=True,
                    core_sha256=digest(core),
                    half_words_checked=p * p * sum(expected.values()),
                )
            )
        records.append(
            dict(
                receipt=pointer,
                profile=data["profile"]["id"],
                source_sha256=data["source_sha256"],
                harness_sha256=data["harness_sha256"],
                evaluator_sha256=data["provenance"]["evaluator_sha256"],
                cases=cases,
            )
        )
    return dict(
        schema_version=1,
        status="core_output_state_and_timer_replayed",
        case_count=sum(len(r["cases"]) for r in records),
        auditor_sha256=digest(Path(__file__)),
        helper_sha256=digest(ROOT / "benchmarks/drivers/snapshot.py"),
        records=records,
        qwen_tps_result=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse to overwrite a snapshot report")
    report = audit(json.loads(args.manifest.read_text()))
    report["manifest_sha256"] = digest(args.manifest)
    write_json(args.output, report)
    print(
        json.dumps(
            dict(status=report["status"], cases=report["case_count"], output=str(args.output))
        )
    )


if __name__ == "__main__":
    main()
