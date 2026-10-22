"""Screen a FP32 SiLU exponent implementation against an independent FP64 reference.

A numerical primitive probe, not a CSL performance result. The end-to-end
resident-block oracle and accuracy limits stay unchanged.
"""

import json

import numpy as np

COEFFICIENTS = (1 / 40320, 1 / 5040, 1 / 720, 1 / 120, 1 / 24, 1 / 6, 0.5, 1.0, 1.0)


def approximate(g, fused):
    r = (-g * np.float32(1 / 16)).astype(np.float32)
    p = np.full_like(r, np.float32(COEFFICIENTS[0]))
    for coefficient in COEFFICIENTS[1:]:
        c = np.float32(coefficient)
        if fused:
            p = (p.astype(np.float64) * r.astype(np.float64) + float(c)).astype(np.float32)
        else:
            p = (p * r + c).astype(np.float32)
    for _ in range(4):
        p = (p * p).astype(np.float32)
    return (g / (np.float32(1) + p)).astype(np.float32), p


def main():
    rng = np.random.default_rng(20261022)
    g = np.concatenate([np.linspace(-8, 8, 20001), rng.uniform(-8, 8, 10000)]).astype(np.float32)
    expected = g.astype(np.float64) / (1 + np.exp(-g.astype(np.float64)))
    reports = []
    for fused in (False, True):
        actual, exponential = approximate(g, fused)
        error = np.abs(actual.astype(np.float64) - expected)
        bound = 2e-6 + 2e-6 * np.abs(expected)
        exp_error = np.abs(exponential.astype(np.float64) / np.exp(-g.astype(np.float64)) - 1)
        reports.append(
            dict(
                fused_horner=fused,
                points=len(g),
                max_abs_error=float(error.max()),
                max_fraction_of_primitive_bound=float((error / bound).max()),
                max_exponential_relative_error=float(exp_error.max()),
            )
        )
        if not np.isfinite(actual).all() or not np.all(error <= bound):
            raise AssertionError(json.dumps(reports[-1]))
    print(
        json.dumps(
            dict(status="primitive_probe_passed", reports=reports, csl_simulation=False), indent=2
        )
    )


if __name__ == "__main__":
    main()
