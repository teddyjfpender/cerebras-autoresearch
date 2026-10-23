"""SDK driver for the full resident GEMM sequence and canonical-state oracle."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import (
    MemcpyDataType,
    MemcpyOrder,
    SdkRuntime,
    SimfabConfig,
    get_platform,
)
from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view
from common import CASES, matrix_inputs, save_case
from reusable_gemm_oracle import evidence, pack_inputs, unpack_outputs
from snapshot import read_core_arrays


def read_symbol(runner, symbol, p, count, options, progress=False):
    """Read every word with bounded row groups after the device timer ends.

    A full-grid weight-history read stalled in the simulator. Keep each reply
    below a conservative software cap while preserving the complete PE-major
    array. This changes transport only; no output or state check is omitted.
    """
    max_words = 32768
    if p < 1 or count < 1 or p * count > max_words:
        raise ValueError("A single readback row exceeds the transport word budget")
    rows_per_read = min(p, max_words // (p * count))
    raw = np.zeros((p, p, count), np.uint32)
    identifier = runner.get_id(symbol)
    for row in range(0, p, rows_per_read):
        rows = min(rows_per_read, p - row)
        block = np.zeros(rows * p * count, np.uint32)
        started = time.monotonic()
        if progress:
            print(f"READBACK {symbol}: rows {row}:{row + rows}, elements/PE={count}", flush=True)
        runner.memcpy_d2h(block, identifier, 0, row, p, rows, count, **options)
        if progress:
            print(f"READBACK {symbol}: completed in {time.monotonic() - started:.3f}s", flush=True)
        raw[row : row + rows] = block.reshape(rows, p, count)
    return memcpy_view(raw.ravel(), np.dtype(np.float16)).copy()


def read_weight_history(runner, p, t, per_frame, options, progress=False):
    """Retrieve the same full history through aliases for each saved frame."""
    frames = [
        read_symbol(runner, f"W_HISTORY_{frame}", p, per_frame, options, progress).reshape(
            p, p, per_frame
        )
        for frame in range(t)
    ]
    return np.stack(frames, axis=2).ravel()


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
    if (out / "out.core").exists():
        raise RuntimeError("Unexpected stale simulator core before a fresh case")
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
        # A blocking fabric read fences completion on every PE. The timestamp
        # values were fixed at the end of the same whole-sequence timer.
        timer = np.zeros(p * p * 3, np.uint32)
        timer_options = dict(options, data_type=MemcpyDataType.MEMCPY_32BIT)
        runner.memcpy_d2h(timer, runner.get_id("time_memcpy"), 0, 0, p, p, 3, **timer_options)
    finally:
        # Failure here rejects the case before any PASS or numerical receipt.
        runner.stop()
    core = out / f"case-{args.case}.core"
    if core.exists():
        raise RuntimeError("Refusing to overwrite a saved simulator snapshot")
    (out / "out.core").rename(core)
    expected = {
        "ALL_OUT": t * mt * nt,
        "X_FRAMES": t * mt * kt,
        "W_CANON": kt * nt,
        **{f"W_HISTORY_{frame}": kt * nt for frame in range(t)},
    }
    started = time.monotonic()
    captured = read_core_arrays(core, out / "bin", p, expected)
    history = np.stack(
        [captured[f"W_HISTORY_{frame}"].reshape(p, p, kt * nt) for frame in range(t)], axis=2
    )
    print(
        f"SNAPSHOT: complete output/state extraction in {time.monotonic() - started:.3f}s",
        flush=True,
    )
    save_case(
        out,
        args.case,
        *evidence(
            p,
            frames,
            w,
            unpack_outputs(p, mt, nt, t, captured["ALL_OUT"]),
            history,
            captured["X_FRAMES"],
            captured["W_CANON"],
        ),
        timer,
        1,
        args.seed,
    )
    print("SUCCESS!", flush=True)


if __name__ == "__main__":
    main()
