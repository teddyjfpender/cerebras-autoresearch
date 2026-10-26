"""BF16 exhaustive host preflight for a bounded SiLU tail; no device score."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
from resident_swiglu_bf16_oracle import decode, silu  # noqa: E402


def main():
    raw = np.arange(65536, dtype=np.uint16)
    raw = raw[(raw & 0x7F80) != 0x7F80]
    gates = decode(raw)
    gates = gates[np.abs(gates) <= 8]
    reference = silu(gates)
    primary = json.loads((ROOT / "runs/maintenance/bf16-silu-primitive.json").read_text())
    main_coefficients = next(r for r in primary["results"] if r["degree"] == 6)["coefficients"]
    square = (gates * gates).astype(np.float32)
    p = np.full_like(gates, main_coefficients[-1])
    for coefficient in main_coefficients[-2::-1]:
        p = (p * square + np.float32(coefficient)).astype(np.float32)
    main_value = (np.float32(0.5) * gates + square * p).astype(np.float32)
    x = (-np.abs(gates) / np.float32(16)).astype(np.float32)
    results = []
    for degree in range(3, 9):
        coefficients = []
        factorial = 1
        for i in range(degree + 1):
            if i:
                factorial *= i
            coefficients.append(float(np.float32(1 / factorial)))
        p = np.full_like(gates, coefficients[-1])
        for coefficient in coefficients[-2::-1]:
            p = (p * x + np.float32(coefficient)).astype(np.float32)
        for _ in range(4):
            p = (p * p).astype(np.float32)
        tail = np.where(gates >= 0, gates / (np.float32(1) + p), gates * p / (np.float32(1) + p))
        actual = np.where(np.abs(gates) <= 4, main_value, tail).astype(np.float32)
        error = np.abs(actual.astype(np.float64) - reference)
        relative = np.divide(
            error, np.abs(reference), out=np.zeros_like(error), where=reference != 0
        )
        e = float(relative.max())
        envelope = (1 + e) * (1 + 2**-24) * (1 + 2**-8) - 1
        results.append(
            dict(
                degree=degree,
                coefficients=coefficients,
                max_relative_error=e,
                worst_gate=float(gates[int(relative.argmax())]),
                normal_product_relative_envelope=envelope,
                primitive_gate_passed=envelope <= 0.75 * 4 * 2**-8,
            )
        )
    output = ROOT / "runs/maintenance/bf16-silu-tail-primitive.json"
    if output.exists():
        raise ValueError("Refuse to overwrite a primitive screen")
    output.write_text(
        json.dumps(
            dict(
                schema_version=1,
                status="math_screen_only",
                bf16_gate_count=len(gates),
                range=[-8, 8],
                main_degree=6,
                tail_scale=16,
                tail_squares=4,
                prescribed_normal_product_relative_gate=0.75 * 4 * 2**-8,
                unchanged_oracle_relative_allowance=4 * 2**-8,
                results=results,
                device_execution=False,
                qwen_tps_result=False,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(output=str(output), results=results), indent=2))


if __name__ == "__main__":
    main()
