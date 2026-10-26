"""Staged BF16-storage/FP32-accumulator complete-chain SDK driver."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import (
    MemcpyDataType,
    MemcpyOrder,
    SdkRuntime,
    SimfabConfig,
    get_platform,
)
from cerebras.sdk.sdk_utils import input_array_to_u32
from common import CASES, save_case
from oracle import evidence, inputs, packed_inputs
from snapshot import read_core_arrays


def unpack(raw, p, mt, nt, t):
    tiles = raw.view(np.uint16).reshape(p, p, t, nt, mt)
    return tiles.transpose(2, 0, 4, 1, 3).reshape(t, p * mt, p * nt).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    out = Path(args.name).resolve()
    dims = json.loads((out / "out.json").read_text())["params"]
    p, mt, ht, ft, t = (int(dims[k]) for k in ("P", "Mt", "Ht", "Ft", "T"))
    values = inputs(t, p * mt, p * ht, p * ft, args.seed, args.case)
    packed = packed_inputs(p, *values)
    os.chdir(out)
    if (out / "out.core").exists():
        raise RuntimeError("Unexpected stale core")
    platform = get_platform(config=SimfabConfig(num_threads=1, suppress_trace=True, dump_core=True))
    runner = SdkRuntime(str(out), platform)
    runner.load()
    runner.run()
    options = dict(
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    try:
        for symbol, value in packed.items():
            runner.memcpy_h2d(
                runner.get_id(symbol),
                input_array_to_u32(value.ravel(), 1, 1),
                0,
                0,
                p,
                p,
                value.size // (p * p),
                **options,
            )
        runner.launch("init_task", nonblock=False)
        runner.launch("sequence_host", nonblock=False)
        timer = np.zeros(p * p * 3, np.uint32)
        runner.memcpy_d2h(
            timer,
            runner.get_id("time_memcpy"),
            0,
            0,
            p,
            p,
            3,
            **dict(options, data_type=MemcpyDataType.MEMCPY_32BIT),
        )
    finally:
        runner.stop()
    core = out / f"case-{args.case}.core"
    if core.exists():
        raise RuntimeError("Refuse to overwrite saved simulator evidence")
    (out / "out.core").rename(core)
    expected = {k: v.size // (p * p) for k, v in packed.items()}
    expected.update({k: t * mt * ft for k in ("G_HISTORY", "U_HISTORY", "V_HISTORY")})
    expected.update(D_HISTORY=t * mt * ht, Y_HISTORY=t * mt * ht)
    expected.update({f"W_HISTORY_{i}": ht * ft for i in range(t)})
    captured = read_core_arrays(core, out / "bin", p, expected)
    stages = {
        k: unpack(captured[f"{k.upper()}_HISTORY"], p, mt, ft if k in "guv" else ht, t)
        for k in "guvdy"
    }
    history = np.stack(
        [captured[f"W_HISTORY_{i}"].view(np.uint16).reshape(p, p, ht * ft) for i in range(t)],
        axis=2,
    )
    save_case(
        out,
        args.case,
        *evidence(p, *values, stages, {k: captured[k].view(np.uint16) for k in packed}, history),
        timer,
        1,
        args.seed,
    )
    print("SUCCESS!", flush=True)


if __name__ == "__main__":
    main()
