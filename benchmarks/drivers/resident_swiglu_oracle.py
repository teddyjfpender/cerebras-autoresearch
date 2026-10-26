"""Independent staged resident SwiGLU specification; no device execution."""

import numpy as np
from common import CASES, matrix_inputs
from reusable_gemm_oracle import pack_inputs


def inputs(t, m, h, f, seed, case):
    if case not in CASES or t != 4:
        raise ValueError("Unknown family or frame count")
    x, wg = matrix_inputs(t * m, h, f, seed, case, np.float16)
    _, wu = matrix_inputs(t * m, h, f, seed ^ 0x519A, case, np.float16)
    _, wd = matrix_inputs(t * m, f, h, seed ^ 0xA671, case, np.float16)
    scales = np.array([0.75, 1.0, 1.25, 1.5], np.float64)[:, None, None]
    x = (x.reshape(t, m, h).astype(np.float64) * scales).astype(np.float16)
    if case == "zero":
        x.view(np.uint16).reshape(-1)[1::2] = 0x8000
    return (
        x,
        (wg / np.sqrt(h)).astype(np.float16),
        (wu / np.sqrt(h)).astype(np.float16),
        (wd / np.sqrt(f)).astype(np.float16),
    )


def silu(x):
    x = np.asarray(x, np.float64)
    magnitude = np.exp(-np.abs(x))
    return np.where(x >= 0, x / (1 + magnitude), x * magnitude / (1 + magnitude))


def linear(a, b, p):
    k = a.shape[-1]
    length, unit = k + p, 2.0**-11
    if length * unit >= 1:
        raise ValueError("Projection exceeds the fixed FP16 accumulation bound")
    a, b = a.astype(np.float64), b.astype(np.float64)
    return a @ b, length * unit / (1 - length * unit) * (np.abs(a) @ np.abs(b)) + length * 2.0**-24


def packed_inputs(p, x, wg, wu, wd):
    t, m, h = x.shape
    f = wg.shape[1]
    if p != 4 or t != 4 or any(d % p for d in (m, h, f)) or f < h:
        raise ValueError("Unsupported shape for the staged 4x4 padded chain")
    px, pg = pack_inputs(p, m // p, h // p, f // p, x, wg)
    _, pu = pack_inputs(p, m // p, h // p, f // p, x, wu)
    dummy = np.zeros((t, m, f), np.float16)
    _, pd = pack_inputs(p, m // p, f // p, h // p, dummy, wd)
    return {"X_FRAMES": px, "WG_CANON": pg, "WU_CANON": pu, "WD_CANON": pd}


def stage_shapes(x, wg):
    t, m, h = x.shape
    f = wg.shape[1]
    return {name: (t, m, f if name in ("g", "u", "v") else h) for name in "guvdy"}


def evidence(p, x, wg, wu, wd, stages, canonical, history):
    if any(a.dtype != np.float16 or not np.isfinite(a).all() for a in (x, wg, wu, wd)):
        raise ValueError("Immutable inputs must be finite FP16")
    t, m, h = x.shape
    f = wg.shape[1]
    if wg.shape != (h, f) or wu.shape != (h, f) or wd.shape != (f, h):
        raise ValueError("Invalid projection dimensions")
    shapes = stage_shapes(x, wg)
    if set(stages) != set(shapes):
        raise ValueError("Every intermediate and output must be supplied")
    for name, shape in shapes.items():
        a = stages[name]
        if a.shape != shape or a.dtype != (np.float32 if name == "y" else np.float16):
            raise ValueError("Incomplete stage shape or wrong precision")
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite stage output")
    packed = packed_inputs(p, x, wg, wu, wd)
    if set(canonical) != set(packed):
        raise ValueError("Incomplete canonical input evidence")
    exact = True
    for name, value in packed.items():
        after = canonical[name]
        if after.dtype != np.float16 or after.shape != (value.size,):
            raise ValueError("Canonical state shape/precision mismatch")
        exact &= after.tobytes() == value.ravel().tobytes()
    expected_history = np.broadcast_to(
        packed["WD_CANON"][:, :, None], (p, p, t, (f // p) * (h // p))
    ).copy()
    if history.dtype != np.float16 or history.shape != expected_history.shape:
        raise ValueError("Incomplete final-down weight history")
    exact &= history.tobytes() == expected_history.tobytes()
    g, eg = linear(x, wg, p)
    u, eu = linear(x, wu, p)
    actual_g, actual_u = (stages[k].astype(np.float64) for k in ("g", "u"))
    local_v = silu(actual_g) * actual_u
    local_v_bound = 4 * 2.0**-11 * np.abs(local_v) + 2.0**-24
    local_d, local_d_bound = linear(stages["v"], wd, p)
    local_y = stages["d"].astype(np.float64) + x.astype(np.float64)
    local_y_bound = (
        4 * 2.0**-24 * (np.abs(stages["d"].astype(np.float64)) + np.abs(x.astype(np.float64)))
        + 2.0**-149
    )
    ideal_v = silu(g) * u
    ev = (
        1.5 * np.abs(u) * eg
        + (np.abs(g) + eg) * eu
        + 4 * 2.0**-11 * (np.abs(g) + eg) * (np.abs(u) + eu)
        + 2.0**-24
    )
    ideal_d = ideal_v @ wd.astype(np.float64)
    gamma = (f + p) * 2.0**-11 / (1 - (f + p) * 2.0**-11)
    ed = (
        ev @ np.abs(wd.astype(np.float64))
        + gamma * ((np.abs(ideal_v) + ev) @ np.abs(wd.astype(np.float64)))
        + (f + p) * 2.0**-24
    )
    ideal_y = ideal_d + x.astype(np.float64)
    ey = ed + 4 * 2.0**-24 * (np.abs(ideal_d) + ed + np.abs(x.astype(np.float64))) + 2.0**-149
    checks = [
        (stages["g"], g, eg),
        (stages["u"], u, eu),
        (stages["v"], local_v, local_v_bound),
        (stages["d"], local_d, local_d_bound),
        (stages["y"], local_y, local_y_bound),
        (stages["y"], ideal_y, ey),
    ]
    actual, expected, bound = (
        np.concatenate([triple[i].astype(np.float64).ravel() for triple in checks])
        for i in range(3)
    )
    arrays = dict(
        x=x,
        wg=wg,
        wu=wu,
        wd=wd,
        **{f"stage_{k}": a for k, a in stages.items()},
        **{f"canonical_{k}": a for k, a in canonical.items()},
        working_w_history=history,
        expected_working_w_history=expected_history,
        actual=actual,
        expected=expected,
        exact_state_match=np.array(exact),
    )
    return arrays, np.abs(actual - expected), bound
