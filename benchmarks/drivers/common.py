"""Trusted SDK-side numerical and timestamp helpers. No host wall time is scored."""

import hashlib
import json
from pathlib import Path

import numpy as np

CASES = ("positive", "signed", "zero", "cancellation", "one_hot", "mixed_scale")


def matrix_inputs(m, k, n, seed, case, dtype):
    rng = np.random.default_rng(seed)
    x, w = rng.uniform(-1, 1, (m, k)), rng.uniform(-1, 1, (k, n))
    if case == "positive":
        x, w = np.abs(x), np.abs(w)
    elif case == "zero":
        x[:] = 0
    elif case == "cancellation":
        x[:] = 1
        w[:] = np.where(np.arange(k) % 2, -1, 1)[:, None]
    elif case == "one_hot":
        x[:] = 0
        x[np.arange(m), np.arange(m) % k] = 1
    elif case == "mixed_scale":
        scale = np.where(np.arange(k) % 2, 16.0, 1 / 16.0)
        x *= scale
        w /= scale[:, None]
    return x.astype(dtype), w.astype(dtype)


def timing(raw, repeats):
    words = np.asarray(raw).view(np.uint32).reshape(-1, 3).astype(np.uint64)
    start = words[:, 0] | ((words[:, 1] & 65535) << np.uint64(32))
    end = (words[:, 1] >> np.uint64(16)) | (words[:, 2] << np.uint64(16))
    elapsed = ((end - start) & np.uint64((1 << 48) - 1)).astype(np.float64) / repeats
    if not np.isfinite(elapsed).all() or elapsed.min() <= 0:
        raise AssertionError("Invalid local timestamp interval")
    return {
        "max_local_cycles": float(elapsed.max()),
        "mean_local_cycles": float(elapsed.mean()),
        "pe_cycle_spread": float(elapsed.max() - elapsed.min()),
    }


def save_case(out, case, arrays, error, bound, raw_timer, repeats, seed):
    """Always preserve evidence, including output from a failing numerical case."""
    artifact = Path(out) / f"case-{case}.npz"
    np.savez_compressed(
        artifact, **arrays, timer_u32=np.asarray(raw_timer).view(np.uint32), error_bound=bound
    )
    passed = bool(np.isfinite(arrays["actual"]).all() and np.all(error <= bound))
    report = {
        "case": case,
        "seed": seed,
        "passed": passed,
        "max_abs_error": float(error.max()),
        "max_fraction_of_error_bound": float((error / bound).max()),
        "artifact": artifact.name,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "metrics": timing(raw_timer, repeats),
    }
    (Path(out) / f"case-{case}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    if not passed:
        raise AssertionError(f"{case}: output violates fixed forward-error bound; evidence saved")
    print(f"PASS {case}: {report['metrics']}", flush=True)
    return report
