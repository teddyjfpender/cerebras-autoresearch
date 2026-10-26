"""Unscored FP32 exp emulation with the original fixed 16*u32 error gate.

This does not establish device correctness; full SDK baseline qualification is
required. It checks separate FP32 multiply/add Horner over 1,000,001 domain
points, boundary neighbors and every canonical decay input in both paired seeds.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
from common import CASES  # noqa: E402
from recurrent_gdn_oracle import inputs  # noqa: E402


def exp32(g):
    r = np.asarray(g, np.float32).copy()
    scale = np.ones_like(r)
    low = g < np.float32(-1.3862943611198906)
    middle = (~low) & (g < np.float32(-0.6931471805599453))
    r[low] = g[low] + np.float32(1.3862943611198906)
    scale[low] = np.float32(0.25)
    r[middle] = g[middle] + np.float32(0.6931471805599453)
    scale[middle] = np.float32(0.5)
    p = np.full_like(r, np.float32(1 / 362880))
    for c in (1 / 40320, 1 / 5040, 1 / 720, 1 / 120, 1 / 24, 1 / 6, 0.5, 1.0, 1.0):
        p = (p * r).astype(np.float32) + np.float32(c)
    return (p * scale).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Historical reports are immutable")
    dense = np.linspace(-2, 0, 1_000_001, dtype=np.float32)
    boundaries = []
    for c in (np.float32(-np.log(2)), np.float32(-2 * np.log(2)), np.float32(0)):
        for direction in (np.float32(-np.inf), np.float32(np.inf)):
            x = c
            for _ in range(8):
                boundaries.append(x)
                x = np.nextafter(x, direction)
    canonical = [
        inputs(k, 16, seed, case)["G_CANON"].ravel()
        for k in (16, 32)
        for seed in (20261035, 20261036)
        for case in CASES
    ]
    values = np.concatenate([dense, np.array(boundaries, np.float32), *canonical])
    values = values[(values >= -2) & (values <= 0)]
    expected = np.exp(values.astype(np.float64))
    actual = exp32(values)
    bound = 16 * 2.0**-24 * np.abs(expected) + 16 * 2.0**-149
    fraction = np.abs(actual.astype(np.float64) - expected) / bound
    if not np.all(fraction <= 1):
        raise AssertionError("Fixed decay error envelope exceeded")
    script = Path(__file__)
    output = dict(
        schema_version=1,
        status="fixed_exp_envelope_passed_in_fp32_host_emulation",
        points=int(values.size),
        domain=[-2, 0],
        degree=9,
        range_reduction="binary scaling; residual in [-ln2,0]",
        max_error_fraction=float(fraction.max()),
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        sdk_device_execution=False,
        qwen_tps_result=False,
        performance_measurement=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
