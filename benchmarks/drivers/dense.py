"""Independent full-output oracle for pinned WaferLLM GEMV and GEMM kernels."""

import argparse
import json
import os
from pathlib import Path

import host_common as hc
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder, SdkRuntime
from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view
from common import CASES, matrix_inputs, save_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--kind", choices=("meshgemv", "meshgemm"), required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    out = Path(args.name).resolve()
    params = json.loads((out / "out.json").read_text())["params"]
    p, mt, nt = (int(params[k]) for k in ("P", "Mt", "Nt"))
    gemv = args.kind == "meshgemv"
    kt = mt if gemv else int(params["Kt"])
    m, k, n = (1 if gemv else p * mt), p * kt, p * nt
    x, w = matrix_inputs(m, k, n, args.seed, args.case, np.float16)
    if gemv:
        x_data = np.tile(x.reshape(p, mt), (1, p)).ravel()
        w_data = hc.tile_W(w, p, mt, nt)
        x_count, w_count, y_count = mt, mt * nt, nt
    else:
        indices = np.zeros((p, p), dtype=int)
        for row in range(p):
            for col in range(p):
                if row == 0:
                    indices[row, col] = col
                elif row == 1:
                    _, indices[row, col] = hc.assignId(indices[0, col], p)
                elif (row - 1) % 2 == 0:
                    _, indices[row, col] = hc.assignId(indices[row - 2, col], p)
                else:
                    indices[row, col], _ = hc.assignId(indices[row - 2, col], p)
        shuffled = np.zeros_like(w)
        for row in range(p):
            for col in range(p):
                origin = indices[row, col]
                shuffled[row * kt : (row + 1) * kt, col * nt : (col + 1) * nt] = w[
                    origin * kt : (origin + 1) * kt, col * nt : (col + 1) * nt
                ]
        x_data, w_data = hc.tile_X(x, p, mt, kt), hc.tile_W(shuffled, p, kt, nt)
        x_count, w_count, y_count = mt * kt, kt * nt, mt * nt
    os.chdir(out)
    runner = SdkRuntime(str(out), simfab_numthreads=1, suppress_simfab_trace=True)
    io16, io32, order = (
        MemcpyDataType.MEMCPY_16BIT,
        MemcpyDataType.MEMCPY_32BIT,
        MemcpyOrder.ROW_MAJOR,
    )
    runner.load()
    runner.run()
    try:
        for symbol, values, count in (("X", x_data, x_count), ("W", w_data, w_count)):
            runner.memcpy_h2d(
                runner.get_id(symbol),
                input_array_to_u32(values, 1, 1),
                0,
                0,
                p,
                p,
                count,
                streaming=False,
                data_type=io16,
                order=order,
                nonblock=False,
            )
        runner.launch("init_task", nonblock=False)
        repeats, warmups = (10, 1) if gemv else (1, 0)
        runner.launch(
            "meshgemv_host" if gemv else "meshgemm_host",
            np.int16(warmups),
            np.int16(repeats),
            nonblock=False,
        )
        raw = np.zeros(p * p * y_count, np.uint32)
        runner.memcpy_d2h(
            raw,
            runner.get_id("res"),
            0,
            0,
            p,
            p,
            y_count,
            streaming=False,
            data_type=io16,
            order=order,
            nonblock=False,
        )
        values = memcpy_view(raw, np.dtype(np.float16))
        actual = (values.reshape(p, n) if gemv else hc.untile_res(values, p, mt, nt)).astype(
            np.float64
        )
        expected = x.astype(np.float64) @ w.astype(np.float64)
        # FP16 FMA/reduction paths have no more than k+p rounded operations.
        # GEMV has a shorter local accumulation path; preserve its tighter bound.
        length = (mt if gemv else k) + p
        u = 2.0**-11
        if length * u >= 1:
            raise ValueError("Shape exceeds this oracle's forward-error contract")
        gamma = length * u / (1 - length * u)
        bound = (
            gamma * (np.abs(x.astype(np.float64)) @ np.abs(w.astype(np.float64)))
            + (k + p) * 2.0**-24
        )
        timer = np.zeros(p * p * 3, np.uint32)
        runner.memcpy_d2h(
            timer,
            runner.get_id("time_memcpy"),
            0,
            0,
            p,
            p,
            3,
            streaming=False,
            data_type=io32,
            order=order,
            nonblock=False,
        )
        save_case(
            out,
            args.case,
            {"x": x, "w": w, "actual": actual, "expected": expected},
            np.abs(actual - expected),
            bound,
            timer,
            repeats,
            args.seed,
        )
    finally:
        runner.stop()
    print("SUCCESS!", flush=True)


if __name__ == "__main__":
    main()
