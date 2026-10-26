"""Unscored native BF16 conversion and mixed-FMA capability test; no timer."""

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


def main():
    out = Path("out").resolve()
    xbits = np.array(
        [
            0,
            0x8000,
            0x3F80,
            0xBF80,
            0x3F81,
            0xBF81,
            0x3F7F,
            0xBF7F,
            0x0080,
            0x8080,
            0x0001,
            0x8001,
            0x4F00,
            0xCF00,
            0x4000,
            0xC000,
        ],
        np.uint16,
    )
    base = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.125,
            -0.125,
            0.001,
            -0.001,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -1.0,
            0.25,
            -0.25,
        ],
        np.float32,
    )
    testbits = np.array(
        [
            0,
            0x80000000,
            0x3F808000,
            0x3F818000,
            0xBF808000,
            0xBF818000,
            0x00010000,
            0x00008000,
            0x00018000,
            0x00808000,
            0x80808000,
            0x3F800001,
            0x3F80FFFF,
            0x3F807FFF,
            0x3F808001,
            0xBF808001,
        ],
        np.uint32,
    )
    test = testbits.view(np.float32)
    expected_bf = ((testbits + np.uint32(0x7FFF) + ((testbits >> 16) & 1)) >> 16).astype(np.uint16)
    x = (xbits.astype(np.uint32) << 16).view(np.float32)
    expected_out = (x.astype(np.float64) * 1.5 + base.astype(np.float64)).astype(np.float32)
    expected_round = (expected_bf.astype(np.uint32) << 16).view(np.float32)
    os.chdir(out)
    if (out / "out.core").exists():
        raise ValueError("Unexpected stale core")
    runner = SdkRuntime(
        str(out),
        get_platform(config=SimfabConfig(num_threads=1, suppress_trace=True, dump_core=True)),
    )
    runner.load()
    runner.run()
    options = dict(streaming=False, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    captured = {}
    try:
        for symbol, value, kind in [
            ("X", input_array_to_u32(xbits, 1, 1), MemcpyDataType.MEMCPY_16BIT),
            ("BASE", base, MemcpyDataType.MEMCPY_32BIT),
            ("TEST", test, MemcpyDataType.MEMCPY_32BIT),
        ]:
            runner.memcpy_h2d(
                runner.get_id(symbol), value, 0, 0, 1, 1, 16, data_type=kind, **options
            )
        runner.launch("compute", nonblock=False)
        for symbol in ("X", "OUT", "BF", "ROUND", "BFVEC", "ROUNDVEC"):
            target = np.zeros(16, np.uint32)
            runner.memcpy_d2h(
                target,
                runner.get_id(symbol),
                0,
                0,
                1,
                1,
                16,
                data_type=MemcpyDataType.MEMCPY_16BIT
                if symbol in ("X", "BF", "BFVEC")
                else MemcpyDataType.MEMCPY_32BIT,
                **options,
            )
            captured[symbol] = target
    finally:
        runner.stop()
    normal = ((xbits & 0x7F80) != 0) | ((xbits & 0x7FFF) == 0)
    normal_tests = ((testbits & 0x7FFFFFFF) >= 0x00800000) | ((testbits & 0x7FFFFFFF) == 0)
    fma_equal = captured["OUT"] == expected_out.view(np.uint32)
    bf_equal = captured["BF"].astype(np.uint16) == expected_bf
    round_equal = captured["ROUND"] == expected_round.view(np.uint32)
    vector_equal = (captured["BFVEC"].astype(np.uint16) == expected_bf) & (
        captured["ROUNDVEC"] == expected_round.view(np.uint32)
    )
    checks = dict(
        vector_conversion_nearest_even_exact=bool(vector_equal.all()),
        canonical_input_bits_preserved=bool(np.array_equal(captured["X"].astype(np.uint16), xbits)),
        normal_fp32_accumulator_fma_exact=bool(fma_equal[normal].all()),
        subnormal_fp32_accumulator_fma_exact=bool(fma_equal[~normal].all()),
        normal_conversion_nearest_even_exact=bool((bf_equal & round_equal)[normal_tests].all()),
        subnormal_conversion_nearest_even_exact=bool((bf_equal & round_equal)[~normal_tests].all()),
    )
    np.savez_compressed(
        out / "precision-evidence.npz",
        x_bits=xbits,
        base=base,
        test_bits=testbits,
        expected_bf_bits=expected_bf,
        expected_out_bits=expected_out.view(np.uint32),
        expected_round_bits=expected_round.view(np.uint32),
        **captured,
    )
    report = dict(
        schema_version=1,
        status="precision_probe_passed" if all(checks.values()) else "precision_probe_mismatch",
        checks=checks,
        mismatch_indices=dict(
            fma=np.flatnonzero(~fma_equal).tolist(),
            bf=np.flatnonzero(~bf_equal).tolist(),
            roundtrip=np.flatnonzero(~round_equal).tolist(),
        ),
        timed_benchmark=False,
        timing_score_eligible=False,
        qwen_tps_result=False,
    )
    (out / "precision-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not all(checks.values()):
        raise AssertionError(
            "Native precision differs from proposed exact fixture; evidence preserved"
        )


if __name__ == "__main__":
    main()
