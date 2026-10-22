"""SDK host driver: test quantized FP16 inputs against an FP64 GEMV oracle."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

import host_common as hc


def inputs(m, n, seed):
    rng = np.random.default_rng(seed)
    yield "positive", rng.random((1, m)), rng.random((m, n))
    yield "signed", rng.uniform(-1, 1, (1, m)), rng.uniform(-1, 1, (m, n))
    yield "zero", np.zeros((1, m)), rng.uniform(-1, 1, (m, n))
    signs = np.where(np.arange(m) % 2, -1.0, 1.0)
    yield "cancellation", np.ones((1, m)), np.broadcast_to(signs[:, None], (m, n)).copy()
    x = np.zeros((1, m))
    x[0, m // 3] = 1
    yield "one_hot", x, rng.uniform(-1, 1, (m, n))
    scale = np.where(np.arange(m) % 2, 16.0, 1 / 16.0)
    yield "mixed_scale", rng.uniform(-1, 1, (1, m)) * scale, rng.uniform(-1, 1, (m, n)) / scale[:, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--screen", action="store_true", help="Positive input only; cannot confirm a candidate")
    parser.add_argument("--case", choices=("positive", "signed", "zero", "cancellation", "one_hot", "mixed_scale"),
                        help="Run one numerical case in a fresh simulator process")
    args = parser.parse_args()
    out = Path(args.name).resolve()
    params = json.loads((out / "out.json").read_text())["params"]
    p, mt, nt = (int(params[key]) for key in ("P", "Mt", "Nt"))
    m, n = p * mt, p * nt
    os.chdir(out)
    runner = SdkRuntime(str(out), simfab_numthreads=1, suppress_simfab_trace=True)
    io16, io32, order = MemcpyDataType.MEMCPY_16BIT, MemcpyDataType.MEMCPY_32BIT, MemcpyOrder.ROW_MAJOR
    results = []
    runner.load()
    runner.run()
    try:
        sx, sw, sr = (runner.get_id(key) for key in ("X", "W", "res"))
        st, sf = (runner.get_id(key) for key in ("time_memcpy", "time_ref"))
        for case, x, w in inputs(m, n, args.seed):
            if args.screen and case != "positive":
                continue
            if args.case and case != args.case:
                continue
            x, w = x.astype(np.float16), w.astype(np.float16)
            x_data = np.tile(x.reshape(p, mt), (1, p))
            runner.memcpy_h2d(sx, input_array_to_u32(x_data.ravel(), 1, 1), 0, 0, p, p, mt,
                              streaming=False, data_type=io16, order=order, nonblock=False)
            runner.memcpy_h2d(sw, input_array_to_u32(hc.tile_W(w, p, mt, nt), 1, 1), 0, 0, p, p, mt * nt,
                              streaming=False, data_type=io16, order=order, nonblock=False)
            runner.launch("init_task", nonblock=False)
            runner.launch("meshgemv_host", np.int16(1), np.int16(10), nonblock=False)
            raw = np.zeros(p * n, dtype=np.uint32)
            runner.memcpy_d2h(raw, sr, 0, 0, p, p, nt, streaming=False,
                              data_type=io16, order=order, nonblock=False)
            actual = memcpy_view(raw, np.dtype(np.float16)).reshape(p, n).astype(np.float64)
            expected = x.astype(np.float64) @ w.astype(np.float64)
            # At most mt local FP16 FMAs and p reduction additions on each path.
            # Standard forward-error bound, plus a conservative subnormal floor.
            u = 2.0 ** -11
            gamma = ((mt + p) * u) / (1 - (mt + p) * u)
            bound = gamma * (np.abs(x.astype(np.float64)) @ np.abs(w.astype(np.float64)))
            bound += (m + p) * 2.0 ** -24
            error = np.abs(actual - expected)
            if not np.isfinite(actual).all() or not np.all(error <= bound):
                raise AssertionError(f"{case}: output violates fixed FP16 forward-error bound")
            timer, reference = np.zeros(p * p * 3, np.float32), np.zeros(p * p * 2, np.float32)
            runner.memcpy_d2h(timer, st, 0, 0, p, p, 3, streaming=False,
                              data_type=io32, order=order, nonblock=False)
            runner.memcpy_d2h(reference, sf, 0, 0, p, p, 2, streaming=False,
                              data_type=io32, order=order, nonblock=False)
            mean, maximum = hc.decode_timing(timer.reshape(p, p, 3), reference.reshape(p, p, 2), p, 10)
            artifact = out / f"case-{case}.npz"
            np.savez_compressed(artifact, x=x, w=w, actual=actual, expected=expected,
                                timer_u32=timer.view(np.uint32), reference_u32=reference.view(np.uint32))
            results.append({"case": case, "max_abs_error": float(error.max()),
                            "max_fraction_of_error_bound": float((error / bound).max()),
                            "upstream_mean_cycles": float(mean), "upstream_max_cycles": float(maximum),
                            "artifact": artifact.name, "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()})
            print(f"PASS {case}: cycles={maximum}, max_abs_error={error.max()}", flush=True)
    finally:
        runner.stop()
    report = {"status": "case_passed" if args.case else ("screen_passed" if args.screen else "all_cases_passed"), "reference": "FP64 GEMV on exactly quantized FP16 inputs",
              "contract": "gamma_(Mt+P) * sum(abs(x*w)) + (M+P)*2^-24",
              "seed": args.seed, "warmups": 1, "repeats": 10, "cases": results,
              "simulator_threads": 1,
              "timing_status": "upstream timestamp decoding; hardware calibration pending"}
    report_name = f"validation-{args.case}.json" if args.case else "validation.json"
    (out / report_name).write_text(json.dumps(report, indent=2) + "\n")
    print("SUCCESS!", flush=True)


if __name__ == "__main__":
    main()
