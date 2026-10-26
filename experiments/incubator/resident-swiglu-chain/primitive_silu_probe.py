"""Exhaustive FP16-gate math screen for future scalar SiLU candidates; no device."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
from oracle import silu  # noqa: E402


def main():
    raw = np.arange(65536, dtype=np.uint16).view(np.float16)
    gates = raw[np.isfinite(raw) & (np.abs(raw) <= 4)].astype(np.float32)
    reference = silu(gates)
    x = np.linspace(0, 16, 10001)
    g = np.sqrt(x)
    target = np.full_like(x, 0.25)
    target[1:] = np.tanh(g[1:] / 2) / (2 * g[1:])
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
        results.append(
            dict(
                degree=degree,
                coefficients=[float(v) for v in coefficients],
                max_absolute_error=float(error.max()),
                max_relative_error=float(relative[index]),
                worst_gate=float(gates[index]),
                primitive_gate_passed=bool(relative.max() <= 0.00025),
            )
        )
    output = Path(__file__).resolve().parents[3] / "runs/maintenance/swiglu-silu-primitive.json"
    if output.exists():
        raise ValueError("Refuse to overwrite a primitive screen")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            dict(
                schema_version=1,
                status="math_screen_only",
                fp16_gate_count=len(gates),
                range=[-4, 4],
                prescribed_relative_gate=0.00025,
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
