"""Mixed-precision resident recurrent operator SDK driver with complete core readback."""

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
from recurrent_gdn_native_layout import profile
from recurrent_gdn_native_oracle import K, evidence, inputs
from snapshot_grid import read_core_grid_arrays


def output_shapes(k, v, T):
    R = 1
    return dict(
        Q_HISTORY=((T, k), np.float32),
        K_HISTORY=((T, k), np.float32),
        A_HISTORY=((T, R), np.float32),
        DELTA_HISTORY=((T, R, v), np.float32),
        STATE_HISTORY=((T, R, k, v), np.float32),
        Y_HISTORY=((T, R, v), np.uint16),
        STATE=((R, k, v), np.float32),
    )


def capture_contract(k, v, T, data):
    shapes = {name: (value.shape[2:], value.dtype) for name, value in data.items()}
    shapes.update(output_shapes(k, v, T))
    shapes["times"] = ((6,), np.uint16)
    expected = {
        name: int(np.prod(shape)) * np.dtype(dtype).itemsize // 2
        for name, (shape, dtype) in shapes.items()
    }
    return shapes, expected


def decode_capture(captured, shapes, grid):
    return {
        name: captured[name]
        .view(np.uint16)
        .view(dtype)
        .reshape(grid.height, grid.width, *shape)
        .copy()
        for name, (shape, dtype) in shapes.items()
    }


def pack_timer(words, grid):
    words = np.asarray(words, np.uint32).reshape(grid.pes, 6)
    if (words > 65535).any():
        raise ValueError("Timer transport must contain zero-extended 16-bit words")
    return np.stack([words[:, i] | (words[:, i + 1] << 16) for i in (0, 2, 4)], axis=-1).ravel()


def copy_inputs(runner, data, grid, options):
    """Conservatively bound each host transfer to32768 transport words.

    This is a software iteration bound, not a claimed SDK/hardware limit. Every
    physical PE/canonical word is transferred exactly once before the timer.
    """
    limit = 32768
    for name, value in data.items():
        if value.shape[:2] != (grid.height, grid.width):
            raise ValueError("Incomplete physical input grid")
        per_pe = value.size // grid.pes
        rows = limit // (grid.width * per_pe)
        if rows < 1:
            raise ValueError("A physical input row exceeds the fixed transport bound")
        half = value.dtype == np.uint16
        if not half and value.dtype != np.float32:
            raise ValueError("Unsupported canonical input precision")
        for first in range(0, grid.height, rows):
            count = min(rows, grid.height - first)
            values = value[first : first + count].ravel()
            payload = input_array_to_u32(values, 1, 1) if half else values
            runner.memcpy_h2d(
                runner.get_id(name),
                payload,
                0,
                first,
                grid.width,
                count,
                per_pe,
                **options,
                data_type=MemcpyDataType.MEMCPY_16BIT if half else MemcpyDataType.MEMCPY_32BIT,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    out = Path(args.name).resolve()
    dims = {k: int(v) for k, v in json.loads((out / "out.json").read_text())["params"].items()}
    if set(dims) != {"K", "V", "T", "W", "H"} or dims["K"] != K:
        raise ValueError("Wrong recurrent operator dimensions")
    k, v, T = dims["K"], dims["V"], dims["T"]
    grid = profile(T)
    if (v, dims["W"], dims["H"]) != (grid.tile_v, grid.width, grid.height):
        raise ValueError("Wrong native-width layout budget")
    logical = inputs(T, args.seed, args.case)
    data = grid.inputs(logical)
    shapes, expected = capture_contract(k, v, T, data)
    os.chdir(out)
    if (out / "out.core").exists():
        raise RuntimeError("Unexpected stale core")
    platform = get_platform(config=SimfabConfig(num_threads=1, suppress_trace=True, dump_core=True))
    runner = SdkRuntime(str(out), platform)
    runner.load()
    runner.run()
    options = dict(streaming=False, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    try:
        copy_inputs(runner, data, grid, options)
        runner.launch("sequence_host", nonblock=False)
        timer = np.zeros(grid.pes * 6, np.uint32)
        runner.memcpy_d2h(
            timer,
            runner.get_id("times"),
            0,
            0,
            grid.width,
            grid.height,
            6,
            **options,
            data_type=MemcpyDataType.MEMCPY_16BIT,
        )
    finally:
        runner.stop()
    core = out / f"case-{args.case}.core"
    if core.exists():
        raise RuntimeError("Refuse to overwrite saved simulator evidence")
    (out / "out.core").rename(core)
    captured = decode_capture(
        read_core_grid_arrays(core, out / "bin", grid.width, grid.height, expected), shapes, grid
    )
    actual, canonical, transport = grid.evidence(logical, captured)
    arrays, error, bound = evidence(T, logical, actual, canonical)
    arrays.update(transport)
    arrays["timer_memcpy_u16"] = timer.astype(np.uint16).reshape(grid.height, grid.width, 6)
    arrays["timer_core_u16"] = captured["times"]
    arrays["exact_state_match"] &= np.array_equal(arrays["timer_memcpy_u16"], captured["times"])
    save_case(out, args.case, arrays, error, bound, pack_timer(timer, grid), 1, args.seed)
    print("SUCCESS!", flush=True)


if __name__ == "__main__":
    main()
