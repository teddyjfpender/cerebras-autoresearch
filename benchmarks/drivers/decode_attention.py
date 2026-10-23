"""SDK execution for distributed GQA; host movement is outside device timing."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder, SdkRuntime
from common import CASES, save_case
from decode_attention_oracle import evidence, inputs

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--kind", choices=("decode_attention",), required=True)
parser.add_argument("--case", choices=CASES, required=True)
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()
out = Path(args.name).resolve()
dims = {k: int(v) for k, v in json.loads((out / "out.json").read_text())["params"].items()}
p, s, d, g, t = (dims[k] for k in ("P", "S", "D", "G", "T"))
data = inputs(dims, args.seed, args.case)
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
    for name in ("Q", "NEW_K", "NEW_V"):
        # One full resident sequence on every PE; no host traffic between tokens.
        buffer = np.tile(data[name].ravel(), p).astype(np.float32)
        runner.memcpy_h2d(runner.get_id(name), buffer, 0, 0, p, 1, data[name].size, **options)
    for name in ("K", "V"):
        buffer = data[name].transpose(0, 2, 1).copy().ravel()
        runner.memcpy_h2d(runner.get_id(name), buffer, 0, 0, p, 1, s * d, **options)
    runner.launch("compute", nonblock=False)
    actual = {"Y": np.zeros((t, g, d), np.float32)}
    runner.memcpy_d2h(actual["Y"].ravel(), runner.get_id("Y"), p - 1, 0, 1, 1, t * g * d, **options)
    for name in ("K", "V"):
        buffer = np.zeros((p, d, s), np.float32)
        runner.memcpy_d2h(buffer.ravel(), runner.get_id(name), 0, 0, p, 1, s * d, **options)
        actual[name] = buffer.transpose(0, 2, 1).copy()
    raw = np.zeros((p, 6), np.uint32)
    runner.memcpy_d2h(
        raw.ravel(),
        runner.get_id("times"),
        0,
        0,
        p,
        1,
        6,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    packed = np.stack(
        (raw[:, 0] | raw[:, 1] << 16, raw[:, 2] | raw[:, 3] << 16, raw[:, 4] | raw[:, 5] << 16),
        axis=1,
    )
    arrays, error, bound = evidence(dims, data, actual)
    # All local intervals remain evidence. Only the fixed coordinator is scored.
    save_case(out, args.case, arrays, error, bound, packed, 1, args.seed, coordinator=p - 1)
finally:
    runner.stop()
print("SUCCESS!", flush=True)
