"""Proposed SDK oracle, not yet part of the qualified challenge evaluator."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder, SdkRuntime
from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view
from common import CASES, matrix_inputs, save_case
from sequence_oracle import evidence, pack_inputs, unpack_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    out = Path(args.name).resolve()
    dims = json.loads((out / "out.json").read_text())["params"]
    p, mt, kt, nt, t = (int(dims[k]) for k in ("P", "Mt", "Kt", "Nt", "T"))
    m, k, n = p * mt, p * kt, p * nt
    x, w = matrix_inputs(t * m, k, n, args.seed, args.case, np.float16)
    frames = x.reshape(t, m, k)
    packed_x, packed_w = pack_inputs(p, mt, kt, nt, frames, w)
    os.chdir(out)
    runner = SdkRuntime(str(out), simfab_numthreads=1, suppress_simfab_trace=True)
    runner.load()
    runner.run()
    options = dict(
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    try:
        for symbol, value, count in (
            ("X_FRAMES", packed_x, t * mt * kt),
            ("W_CANON", packed_w, kt * nt),
        ):
            runner.memcpy_h2d(
                runner.get_id(symbol),
                input_array_to_u32(value.ravel(), 1, 1),
                0,
                0,
                p,
                p,
                count,
                **options,
            )
        runner.launch("init_task", nonblock=False)
        runner.launch("sequence_host", nonblock=False)
        captured = {}
        for symbol, count in (
            ("ALL_OUT", t * mt * nt),
            ("W_HISTORY", t * kt * nt),
            ("X_FRAMES", t * mt * kt),
            ("W_CANON", kt * nt),
        ):
            raw = np.zeros(p * p * count, np.uint32)
            runner.memcpy_d2h(raw, runner.get_id(symbol), 0, 0, p, p, count, **options)
            captured[symbol] = memcpy_view(raw, np.dtype(np.float16)).copy()
        actual = unpack_outputs(p, mt, nt, t, captured["ALL_OUT"])
        timer = np.zeros(p * p * 3, np.uint32)
        timer_options = dict(options, data_type=MemcpyDataType.MEMCPY_32BIT)
        runner.memcpy_d2h(timer, runner.get_id("time_memcpy"), 0, 0, p, p, 3, **timer_options)
        save_case(
            out,
            args.case,
            *evidence(
                p,
                frames,
                w,
                actual,
                captured["W_HISTORY"].reshape(p, p, t, kt * nt),
                captured["X_FRAMES"],
                captured["W_CANON"],
            ),
            timer,
            1,
            args.seed,
        )
    finally:
        runner.stop()
    print("SUCCESS!", flush=True)


if __name__ == "__main__":
    main()
