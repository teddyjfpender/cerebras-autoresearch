#!/usr/bin/env python3
"""Recompute accepted independent-oracle evidence inside SDK NumPy.

The manifest lists full validated receipt paths and their SHA-256 digests.
This read-only audit does not run a device program or promote an experiment.
Run: tools/sdk-run cs_python tools/replay_numerics.py --manifest FILE --output FILE
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))

import numpy as np
from common import timing
from resident_oracle import evidence as resident_evidence
from resident_swiglu_oracle import evidence as swiglu_evidence
from reusable_gemm_oracle import evidence as sequence_evidence

from cerebras_research.paths import contained, digest, write_json
from cerebras_research.scoring import load_validated


def recompute(profile, raw):
    kind, dims = profile["adapter"], profile["dimensions"]
    strict = True
    if kind in ("meshgemv", "meshgemm"):
        if raw["x"].dtype != np.float16 or raw["w"].dtype != np.float16:
            raise ValueError("Linear inputs are not quantized FP16")
        x, w = raw["x"].astype(np.float64), raw["w"].astype(np.float64)
        p = profile["pe_grid"][0]
        gemv = kind == "meshgemv"
        k = dims["M"] if gemv else dims["K"]
        m = 1 if gemv else dims["M"]
        if x.shape != (m, k) or w.shape != (k, dims["N"]):
            raise ValueError("Linear input shape disagrees with the profile")
        expected = x @ w
        if raw["actual"].shape != ((p, dims["N"]) if gemv else expected.shape):
            raise ValueError("Incomplete linear output")
        length, u = (k // p if gemv else k) + p, 2.0**-11
        gamma = length * u / (1 - length * u)
        bound = gamma * (np.abs(x) @ np.abs(w)) + (k + p) * 2.0**-24
    elif kind == "reusable_gemm":
        if profile.get("oracle_contract") != "resident_gemm_sequence_fp16_v1":
            raise ValueError("Unsupported resident GEMM oracle version")
        if raw["x"].shape != (dims["T"], dims["M"], dims["K"]) or raw["w"].shape != (
            dims["K"],
            dims["N"],
        ):
            raise ValueError("Sequence input shape disagrees with the profile")
        arrays, _, bound = sequence_evidence(
            profile["pe_grid"][0],
            raw["x"],
            raw["w"],
            raw["actual"],
            raw["working_w_history"],
            raw["canonical_x_after"],
            raw["canonical_w_after"],
        )
        np.testing.assert_array_equal(
            raw["expected_working_w_history"], arrays["expected_working_w_history"]
        )
        expected = arrays["expected"]
        strict = bool(arrays["exact_state_match"])
    elif kind == "resident_swiglu":
        if profile.get("oracle_contract") != "resident_swiglu_sequence_fp16_fp32_v1":
            raise ValueError("Unsupported resident SwiGLU oracle version")
        t, m, h, f = (dims[n] for n in ("T", "M", "H", "F"))
        if any(
            raw[n].shape != shape
            for n, shape in dict(x=(t, m, h), wg=(h, f), wu=(h, f), wd=(f, h)).items()
        ):
            raise ValueError("Resident SwiGLU inputs disagree with profile")
        arrays, _, bound = swiglu_evidence(
            profile["pe_grid"][0],
            *(raw[n] for n in ("x", "wg", "wu", "wd")),
            {k: raw["stage_" + k] for k in "guvdy"},
            {k: raw["canonical_" + k] for k in ("X_FRAMES", "WG_CANON", "WU_CANON", "WD_CANON")},
            raw["working_w_history"],
        )
        np.testing.assert_array_equal(raw["actual"], arrays["actual"])
        np.testing.assert_array_equal(
            raw["expected_working_w_history"], arrays["expected_working_w_history"]
        )
        expected = arrays["expected"]
        strict = bool(arrays["exact_state_match"])
    elif kind == "pipeline":
        x, w = raw["x"].astype(np.float64), raw["w"].astype(np.float64)
        if x.shape != (5, dims["K"]) or w.shape != (dims["K"], dims["N"]):
            raise ValueError("Chain input shape disagrees with the profile")
        if any(raw[name].dtype != np.float32 for name in ("x", "w", "bias", "residual")):
            raise ValueError("Chain inputs are not quantized FP32")
        expected = np.maximum(x @ w + raw["bias"], 0) + raw["residual"]
        if raw["actual"].shape != expected.shape:
            raise ValueError("Incomplete chain output")
        length, u = 2 * dims["K"] + 8, 2.0**-24
        gamma = length * u / (1 - length * u)
        bound = (
            gamma * (np.abs(x) @ np.abs(w) + np.abs(raw["bias"]) + np.abs(raw["residual"]))
            + 32 * 2.0**-149
        )
    elif kind in ("gated_mlp", "decoder"):
        if profile.get("oracle_contract") != "resident_transformer_fp32_v1":
            raise ValueError("Unsupported resident oracle version")
        data = {name[6:]: raw[name] for name in raw.files if name.startswith("input_")}
        names = ("Y", "HISTORY", "CACHE") if kind == "decoder" else ("Y",)
        actual = {name: raw["actual_" + name] for name in names}
        h, f, t = dims["H"], dims["F"], dims["T"]
        shapes = dict(X=(t, h), SCALE1=(h,), WG=(f, h), WU=(f, h), WD=(h, f))
        output_shapes = dict(Y=(t, h))
        if kind == "decoder":
            c = dims["C"]
            shapes.update(SCALE2=(h,), CACHE=(2, c, h))
            shapes.update({name: (h, h) for name in ("WQ", "WK", "WV", "WO")})
            output_shapes.update(HISTORY=(t, 2, c, h), CACHE=(2, c, h))
        if set(data) != set(shapes) or any(
            data[name].shape != shape for name, shape in shapes.items()
        ):
            raise ValueError("Resident input shape disagrees with the profile")
        if any(actual[name].shape != shape for name, shape in output_shapes.items()):
            raise ValueError("Resident output shape disagrees with the profile")
        if any(value.dtype != np.float32 for value in (*data.values(), *actual.values())):
            raise ValueError("Resident data are not FP32")
        arrays, _, bound = resident_evidence(kind, dims, data, actual)
        np.testing.assert_array_equal(raw["actual"], arrays["actual"])
        expected = arrays["expected"]
        strict = bool(arrays["exact_state_match"])
    else:
        raise ValueError(f"Unsupported independent adapter: {kind}")
    return expected, bound, strict


def check_case(profile, case, artifact):
    if digest(artifact) != case["artifact_sha256"]:
        raise ValueError("Raw artifact digest changed")
    with np.load(artifact, allow_pickle=False) as raw:
        expected, bound, strict = recompute(profile, raw)
        np.testing.assert_allclose(raw["expected"], expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(raw["error_bound"], bound, rtol=2e-12, atol=0)
        error = np.abs(raw["actual"] - expected)
        if not np.isfinite(raw["actual"]).all() or not np.all(error <= bound) or not strict:
            raise ValueError("Recomputed numerical/state contract failed")
        if "exact_state_match" in raw and bool(raw["exact_state_match"]) != strict:
            raise ValueError("Saved exact-state result disagrees with replay")
        saved_error = np.abs(raw["actual"] - raw["expected"])
        if case["max_abs_error"] != float(saved_error.max()) or case[
            "max_fraction_of_error_bound"
        ] != float((saved_error / raw["error_bound"]).max()):
            raise ValueError("Saved numerical summary disagrees with its arrays")
        metrics = timing(raw["timer_u32"], profile["timing_protocol"]["repeats"])
        if metrics != case["metrics"]:
            raise ValueError("Recomputed timestamps disagree with the case receipt")
        return {
            "case": case["case"],
            "max_error_fraction_recomputed": float((error / bound).max()),
            "max_abs_error_recomputed": float(error.max()),
            "exact_state_match": strict,
            "metrics": metrics,
            "artifact_sha256": case["artifact_sha256"],
        }


def check_summary(data):
    profile, metrics, cases = data["profile"], data["metrics"], data["cases"]
    expected = {
        name: max(case["metrics"][name] for case in cases)
        for name in ("max_local_cycles", "mean_local_cycles", "pe_cycle_spread")
    }
    primary = expected["max_local_cycles"]
    expected["max_error_fraction"] = max(case["max_fraction_of_error_bound"] for case in cases)
    expected["pe_cycles"] = primary * math.prod(profile["pe_grid"])
    if "work_units" in profile:
        expected["work_units_per_cycle"] = profile["work_units"] / primary
    unit = {
        "gated_mlp": "cycles_per_frame",
        "decoder": "cycles_per_token",
        "reusable_gemm": "cycles_per_gemm",
        "resident_swiglu": "cycles_per_frame",
    }.get(profile["adapter"])
    if unit:
        expected[unit] = primary / profile["dimensions"]["T"]
    if profile["adapter"] == "resident_swiglu":
        expected["amortized_cycles_per_batch_row"] = primary / (
            profile["dimensions"]["T"] * profile["dimensions"]["M"]
        )
    if data.get("steps"):
        expected["experiment_wall_seconds"] = sum(step["wall_seconds"] for step in data["steps"])
    if any(metrics[name] != value for name, value in expected.items() if name in metrics):
        raise ValueError("Run metric summary disagrees with its case/step evidence")


def audit(manifest):
    records = []
    paths = [contained(ROOT, p["path"]) for p in manifest["receipts"]]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate audit receipt")
    for pointer in manifest["receipts"]:
        path = contained(ROOT, pointer["path"])
        if digest(path) != pointer["sha256"]:
            raise ValueError("Selected receipt changed")
        data = load_validated(path)
        cases = [
            check_case(data["profile"], case, path.parent / "work/out" / case["artifact"])
            for case in data["cases"]
        ]
        check_summary(data)
        records.append(
            {
                "receipt": pointer["path"],
                "receipt_sha256": pointer["sha256"],
                "profile": data["profile"]["id"],
                "seed": data["seed"],
                "variant": data["variant"],
                "evaluator_sha256": data["provenance"]["evaluator_sha256"],
                "profile_sha256": data["profile_sha256"],
                "source_sha256": data["source_sha256"],
                "harness_sha256": data["harness_sha256"],
                "sdk_sif_sha256": data["sdk"]["sif_sha256"],
                "metrics": data["metrics"],
                "cases": cases,
            }
        )
    return {
        "schema_version": 1,
        "status": "numerically_replayed",
        "audit_tool_sha256": digest(Path(__file__)),
        "oracle_helpers_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                ROOT / "benchmarks/drivers/common.py",
                ROOT / "benchmarks/drivers/resident_oracle.py",
                ROOT / "benchmarks/drivers/reusable_gemm_oracle.py",
                ROOT / "benchmarks/drivers/resident_swiglu_oracle.py",
            )
        },
        "records": records,
        "case_count": sum(len(r["cases"]) for r in records),
        "qwen_tps_result": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("receipts"):
        parser.error("Manifest needs schema_version 1 and a nonempty receipt list")
    if args.output.exists():
        parser.error("Refuse to overwrite an existing audit report")
    result = audit(manifest)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "runs": len(result["records"]),
                "cases": result["case_count"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
