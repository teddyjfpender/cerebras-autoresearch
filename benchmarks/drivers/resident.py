"""SDK runner for full gated MLP and multi-token causal decoder layer."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder, SdkRuntime
from common import CASES, save_case
from resident_oracle import evidence, inputs

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--kind", choices=("gated_mlp", "decoder"), required=True)
parser.add_argument("--case", choices=CASES, required=True)
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()
out = Path(args.name).resolve()
dims = {k: int(v) for k, v in json.loads((out / "out.json").read_text())["params"].items()}
data = inputs(args.kind, dims, args.seed, args.case)
os.chdir(out)
runner = SdkRuntime(str(out), simfab_numthreads=1, suppress_simfab_trace=True)
runner.load()
runner.run()
try:
    options = dict(
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    for name, value in data.items():
        runner.memcpy_h2d(runner.get_id(name), value.ravel(), 0, 0, 1, 1, value.size, **options)
    runner.launch("compute", nonblock=False)
    actual = {"Y": np.zeros((dims["T"], dims["H"]), np.float32)}
    if args.kind == "decoder":
        actual.update(
            HISTORY=np.zeros((dims["T"], 2, dims["C"], dims["H"]), np.float32),
            CACHE=np.zeros((2, dims["C"], dims["H"]), np.float32),
        )
    for name, value in actual.items():
        runner.memcpy_d2h(value.ravel(), runner.get_id(name), 0, 0, 1, 1, value.size, **options)
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
    arrays, error, bound = evidence(args.kind, dims, data, actual)
    save_case(out, args.case, arrays, error, bound, packed, 1, args.seed)
finally:
    runner.stop()
print("SUCCESS!", flush=True)
