"""Staged BF16-storage / FP32-accumulator FFN oracle; no device or score."""

import numpy as np
from common import CASES, matrix_inputs
from reusable_gemm_oracle import pack_inputs

UB, U32, QB, Q32 = 2.0**-8, 2.0**-24, 2.0**-133, 2.0**-149


def encode(values):
    values = np.asarray(values)
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("BF16 conversion requires finite FP32 samples")
    words = values.view(np.uint32).astype(np.uint64)
    result = ((words + 0x7FFF + ((words >> 16) & 1)) >> 16).astype(np.uint16)
    if np.any((result & 0x7F80) == 0x7F80):
        raise ValueError("BF16 conversion overflow")
    return result


def decode(words):
    words = np.asarray(words)
    if words.dtype != np.uint16 or np.any((words & 0x7F80) == 0x7F80):
        raise ValueError("Require finite BF16 uint16 words")
    return (words.astype(np.uint32) << 16).view(np.float32)


def inputs(t, m, h, f, seed, case):
    if t != 4 or case not in CASES:
        raise ValueError("Unknown frame count or numerical family")
    x, wg = matrix_inputs(t * m, h, f, seed, case, np.float32)
    _, wu = matrix_inputs(t * m, h, f, seed ^ 0x519A, case, np.float32)
    _, wd = matrix_inputs(t * m, f, h, seed ^ 0xA671, case, np.float32)
    scales = np.array([0.75, 1.0, 1.25, 1.5], np.float32)[:, None, None]
    x = encode((x.reshape(t, m, h) * scales).astype(np.float32))
    if case == "zero":
        x.reshape(-1)[1::2] = 0x8000
    return (
        x,
        encode((wg / np.float32(np.sqrt(h))).astype(np.float32)),
        encode((wu / np.float32(np.sqrt(h))).astype(np.float32)),
        encode((wd / np.float32(np.sqrt(f))).astype(np.float32)),
    )


def silu(x):
    x = np.asarray(x, np.float64)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, x / (1 + e), x * e / (1 + e))


def linear(a, b, p):
    a, b = decode(a).astype(np.float64), decode(b).astype(np.float64)
    length = a.shape[-1] + p
    if length * U32 >= 1:
        raise ValueError("Projection exceeds fixed FP32 accumulation bound")
    gamma = length * U32 / (1 - length * U32)
    expected = a @ b
    accumulation = gamma * (np.abs(a) @ np.abs(b)) + length * Q32
    return expected, accumulation + UB * (np.abs(expected) + accumulation) + QB


def packed_inputs(p, x, wg, wu, wd):
    t, m, h = x.shape
    f = wg.shape[1]
    if p != 4 or t != 4 or any(d % p for d in (m, h, f)) or f < h:
        raise ValueError("Unsupported BF16 chain shape/grid")
    if (m // p) * (f // p) % 2 or (h // p) * (f // p) % 2:
        raise ValueError("Half-word transport requires complete even-sized tiles")
    px, pg = pack_inputs(p, m // p, h // p, f // p, x, wg)
    _, pu = pack_inputs(p, m // p, h // p, f // p, x, wu)
    _, pd = pack_inputs(p, m // p, f // p, h // p, np.zeros((t, m, f), np.uint16), wd)
    return dict(X_FRAMES=px, WG_CANON=pg, WU_CANON=pu, WD_CANON=pd)


def evidence(p, x, wg, wu, wd, stages, canonical, history):
    for value in (x, wg, wu, wd):
        decode(value)
    t, m, h = x.shape
    f = wg.shape[1]
    if wg.shape != (h, f) or wu.shape != (h, f) or wd.shape != (f, h):
        raise ValueError("Invalid BF16 projection dimensions")
    shapes = {k: (t, m, f if k in "guv" else h) for k in "guvdy"}
    if set(stages) != set(shapes):
        raise ValueError("Missing BF16 stage evidence")
    actual = {}
    for name, shape in shapes.items():
        if stages[name].shape != shape:
            raise ValueError("Incomplete BF16 stage shape")
        actual[name] = decode(stages[name]).astype(np.float64)
    packed = packed_inputs(p, x, wg, wu, wd)
    if set(canonical) != set(packed):
        raise ValueError("Missing BF16 canonical evidence")
    exact = True
    for name, value in packed.items():
        after = canonical[name]
        if after.dtype != np.uint16 or after.shape != (value.size,):
            raise ValueError("BF16 canonical shape/precision mismatch")
        exact &= after.tobytes() == value.ravel().tobytes()
    expected_history = np.broadcast_to(
        packed["WD_CANON"][:, :, None], (p, p, t, (f // p) * (h // p))
    ).copy()
    if history.dtype != np.uint16 or history.shape != expected_history.shape:
        raise ValueError("Incomplete BF16 working weight history")
    exact &= history.tobytes() == expected_history.tobytes()
    x64, wd64 = decode(x).astype(np.float64), decode(wd).astype(np.float64)
    g, eg = linear(x, wg, p)
    u, eu = linear(x, wu, p)
    local_v = silu(actual["g"]) * actual["u"]
    local_v_bound = 4 * UB * np.abs(local_v) + QB
    local_d, local_d_bound = linear(stages["v"], wd, p)
    local_y = actual["d"] + x64
    eadd = 4 * U32 * (np.abs(actual["d"]) + np.abs(x64)) + Q32
    local_y_bound = eadd + UB * (np.abs(local_y) + eadd) + QB
    ideal_v = silu(g) * u
    ev = (
        1.5 * np.abs(u) * eg
        + (np.abs(g) + eg) * eu
        + 4 * UB * (np.abs(g) + eg) * (np.abs(u) + eu)
        + QB
    )
    ideal_d = ideal_v @ wd64
    ein = ev @ np.abs(wd64)
    length = f + p
    gamma = length * U32 / (1 - length * U32)
    eacc_d = gamma * ((np.abs(ideal_v) + ev) @ np.abs(wd64)) + length * Q32
    ed = ein + eacc_d + UB * (np.abs(ideal_d) + ein + eacc_d) + QB
    ideal_y = ideal_d + x64
    scale = np.abs(ideal_d) + ed + np.abs(x64)
    eadd_y = 4 * U32 * scale + Q32
    ey = ed + eadd_y + UB * (scale + eadd_y) + QB
    checks = [
        (actual["g"], g, eg),
        (actual["u"], u, eu),
        (actual["v"], local_v, local_v_bound),
        (actual["d"], local_d, local_d_bound),
        (actual["y"], local_y, local_y_bound),
        (actual["y"], ideal_y, ey),
    ]
    flattened, expected, bound = (
        np.concatenate([row[i].ravel() for row in checks]).astype(np.float64) for i in range(3)
    )
    arrays = dict(
        x_bits=x,
        wg_bits=wg,
        wu_bits=wu,
        wd_bits=wd,
        **{"bits_" + k: value for k, value in stages.items()},
        **{"canonical_" + k: value for k, value in canonical.items()},
        working_w_history=history,
        expected_working_w_history=expected_history,
        exact_state_match=np.array(exact),
        actual=flattened,
        expected=expected,
    )
    return arrays, np.abs(flattened - expected), bound
