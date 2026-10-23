"""Unscored FP32 bulk arithmetic preflight using actual qualified SDK prefixes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
sys.path.insert(0, str(ROOT / "src"))
from recurrent_gdn_native_oracle import evidence, inputs  # noqa: E402
from resident_swiglu_bf16_oracle import decode, encode  # noqa: E402

from cerebras_research.scoring import load_validated  # noqa: E402


def reduced(products):
    # Products[head,K,V], four independent partial sums in ascending K groups.
    partial = np.zeros((3, 4, 128), np.float32)
    grouped = products.reshape(3, 32, 4, 128)
    for group in range(32):
        partial = (partial + grouped[:, group]).astype(np.float32)
    a = (partial[:, 0] + partial[:, 1]).astype(np.float32)
    b = (partial[:, 2] + partial[:, 3]).astype(np.float32)
    return (a + b).astype(np.float32)


def simulate(data, prefixes, tokens):
    state = data["S0"].copy()
    states = []
    deltas = []
    outputs = []
    for frame in range(tokens):
        q = prefixes["Q_HISTORY"][frame]
        key = prefixes["K_HISTORY"][frame]
        a = prefixes["A_HISTORY"][frame]
        state = (state * a[:, None, None]).astype(np.float32)
        memory = reduced((state * key[None, :, None]).astype(np.float32))
        delta = (
            (decode(data["V_CANON"])[frame] - memory) * data["BETA_CANON"][frame, :, None]
        ).astype(np.float32)
        product = (key[None, :, None] * delta[:, None, :]).astype(np.float32)
        state = (state + product).astype(np.float32)
        output = reduced((state * q[None, :, None]).astype(np.float32))
        states.append(state.copy())
        deltas.append(delta.copy())
        outputs.append(encode(output))
    return dict(
        **prefixes,
        STATE_HISTORY=np.stack(states),
        DELTA_HISTORY=np.stack(deltas),
        Y_HISTORY=np.stack(outputs),
        FINAL_STATE=state.copy(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Historical reports are immutable")
    records = []
    receipts = [
        "runs/v2/20261023T142107Z-recurrent-gdn-native.k128-v128-t4-43a81f78/result.json",
        "runs/v2/20261023T142715Z-recurrent-gdn-native.k128-v128-t8-c3bf1dbf/result.json",
    ]
    for relative in receipts:
        path = ROOT / relative
        run = load_validated(path)
        tokens = run["profile"]["dimensions"]["T"]
        for case in run["cases"]:
            rawpath = path.parent / "work/out" / case["artifact"]
            if hashlib.sha256(rawpath.read_bytes()).hexdigest() != case["artifact_sha256"]:
                raise ValueError("SDK fixture changed")
            data = inputs(tokens, run["seed"], case["case"])
            with np.load(rawpath, allow_pickle=False) as raw:
                prefix = {n: raw[n].copy() for n in ("Q_HISTORY", "K_HISTORY", "A_HISTORY")}
            actual = simulate(data, prefix, tokens)
            arrays, error, bound = evidence(tokens, data, actual, data)
            if not arrays["exact_state_match"] or not np.all(error <= bound):
                raise AssertionError("Original fixed state/precision envelope failed")
            records.append(
                dict(
                    profile=run["profile"]["id"],
                    case=case["case"],
                    source_artifact_sha256=case["artifact_sha256"],
                    max_error_fraction=float((error / bound).max()),
                    passed=True,
                )
            )
    report = dict(
        schema_version=1,
        status="fixed_bulk_fp32_arithmetic_preflight_passed",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        records=records,
        sdk_device_execution=False,
        performance_measurement=False,
        qwen_tps_result=False,
        scope="FP32 MUL/ADD and four-partial arithmetic, using actual verified normalized/decay prefixes. mem4d traversal, compiler behavior, aliasing, complete timed work and device speed require SDK runs.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status=report["status"],
                cases=len(records),
                max_error_fraction=max(r["max_error_fraction"] for r in records),
            )
        )
    )


if __name__ == "__main__":
    main()
