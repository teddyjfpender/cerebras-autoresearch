"""Exhaustive BF16-gate host screen, separate from device qualification."""

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
    decoded = decode(raw)
    gates = decoded[np.isfinite(decoded) & (np.abs(decoded) <= 4)].astype(np.float32)
    reference = silu(gates)
    x = np.linspace(0, 16, 10001)
    g = np.sqrt(x)
    target = np.full_like(x, 0.25)
    target[1:] = np.tanh(g[1:] / 2) / (2 * g[1:])
    # Reserve one quarter of the unchanged 4*u_b local product allowance.
    prescribed_gate = 0.75 * 4 * 2**-8
    results = []
    for degree in range(4, 11):
        fit = np.polynomial.Chebyshev.fit(x, target, degree).convert(kind=np.polynomial.Polynomial)
        coefficients = fit.coef.astype(np.float32)
        square = (gates * gates).astype(np.float32)
        p = np.full_like(gates, coefficients[-1])
        for coefficient in coefficients[-2::-1]:
            p = (p * square + coefficient).astype(np.float32)
        actual = (np.float32(0.5) * gates + square * p).astype(np.float32)
        error = np.abs(actual.astype(np.float64) - reference)
        relative = np.divide(
            error, np.abs(reference), out=np.zeros_like(error), where=reference != 0
        )
        index = int(relative.argmax())
        e = float(relative[index])
        # Normal-value relative envelope after FP32 multiply and BF16 rounding.
        # Subnormal absolute terms are handled by the fixed device oracle, not
        # claimed to be proven by this primitive approximation screen.
        normal_product_envelope = (1 + e) * (1 + 2**-24) * (1 + 2**-8) - 1
        results.append(
            dict(
                degree=degree,
                coefficients=[float(v) for v in coefficients],
                max_absolute_error=float(error.max()),
                max_relative_error=e,
                worst_gate=float(gates[index]),
                normal_product_relative_envelope=normal_product_envelope,
                primitive_gate_passed=normal_product_envelope <= prescribed_gate,
            )
        )
    output = ROOT / "runs/maintenance/bf16-silu-primitive.json"
    if output.exists():
        raise ValueError("Refuse to overwrite a primitive screen")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            dict(
                schema_version=1,
                status="math_screen_only",
                bf16_gate_count=len(gates),
                range=[-4, 4],
                prescribed_normal_product_relative_gate=prescribed_gate,
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
