"""Full-chain oracle for resident matvec, bias, ReLU, and residual addition."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder, SdkRuntime
from common import CASES, matrix_inputs, save_case

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--case", choices=CASES, required=True)
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()
out = Path(args.name).resolve()
params = json.loads((out / "out.json").read_text())["params"]
k, n = int(params["K"]), int(params["N"])
x, w = matrix_inputs(5, k, n, args.seed, args.case, np.float32)
rng = np.random.default_rng(args.seed + 1)
bias, residual = rng.uniform(-1, 1, (2, n)).astype(np.float32)
os.chdir(out)
runner = SdkRuntime(str(out), simfab_numthreads=1, suppress_simfab_trace=True)
runner.load()
runner.run()
try:
    for symbol, values in (("X", x), ("W", w.T.copy()), ("B", bias), ("R", residual)):
        runner.memcpy_h2d(
            runner.get_id(symbol),
            values.ravel(),
            0,
            0,
            1,
            1,
            values.size,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )
    runner.launch("compute", nonblock=False)
    raw = np.zeros(5 * n, np.float32)
    runner.memcpy_d2h(
        raw,
        runner.get_id("Y"),
        0,
        0,
        1,
        1,
        5 * n,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    t = np.zeros(6, np.uint32)
    runner.memcpy_d2h(
        t,
        runner.get_id("times"),
        0,
        0,
        1,
        1,
        6,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    packed = np.array([t[0] | t[1] << 16, t[2] | t[3] << 16, t[4] | t[5] << 16], np.uint32)
    expected = np.maximum(x.astype(np.float64) @ w.astype(np.float64) + bias, 0) + residual
    actual = raw.astype(np.float64).reshape(5, n)
    u = 2.0**-24
    gamma = (2 * k + 8) * u / (1 - (2 * k + 8) * u)
    bound = (
        gamma
        * (
            np.abs(x.astype(np.float64)) @ np.abs(w.astype(np.float64))
            + np.abs(bias)
            + np.abs(residual)
        )
        + 32 * 2.0**-149
    )
    save_case(
        out,
        args.case,
        {
            "x": x,
            "w": w,
            "bias": bias,
            "residual": residual,
            "actual": actual,
            "expected": expected,
        },
        np.abs(actual - expected),
        bound,
        packed,
        4,
        args.seed,
    )
finally:
    runner.stop()
print("SUCCESS!", flush=True)
