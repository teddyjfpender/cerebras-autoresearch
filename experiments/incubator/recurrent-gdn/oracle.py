"""Independent FP64 stateful gated-delta oracle; staging only, no device score."""

import numpy as np
from common import CASES
from resident_swiglu_bf16_oracle import decode, encode

U, Q32, UB, QB = 2.0**-24, 2.0**-149, 2.0**-8, 2.0**-133
P, T, R = 4, 4, 3


def gamma(n):
    if n * U >= 1:
        raise ValueError("FP32 error bound exhausted")
    return n * U / (1 - n * U)


def inputs(k, v, seed, case):
    if k not in (16, 32) or v != 16 or case not in CASES:
        raise ValueError("Unsupported recurrent operator profile/family")
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1, 1, (P, P, T, k)).astype(np.float32)
    key = rng.uniform(-1, 1, q.shape).astype(np.float32)
    value = rng.uniform(-1, 1, (P, P, T, R, v)).astype(np.float32)
    state = rng.uniform(-0.125, 0.125, (P, P, R, k, v)).astype(np.float32)
    if case == "positive":
        q, key, value, state = map(np.abs, (q, key, value, state))
    elif case == "zero":
        q.fill(0)
        key.fill(0)
        value.fill(0)
    elif case == "cancellation":
        q[:] = np.where(np.arange(k) % 2, -1, 1)
        key[:] = np.where(np.arange(k) % 2, 1, -1)
        value[:] = np.where(np.arange(v) % 2, -0.75, 0.75)
        state[:] = np.where((np.arange(k)[:, None] + np.arange(v)) % 2, -0.125, 0.125)
    elif case == "one_hot":
        q.fill(0)
        key.fill(0)
        for frame in range(T):
            q[:, :, frame, frame % k] = 1
            key[:, :, frame, (frame + 1) % k] = 1
    elif case == "mixed_scale":
        scale = np.exp2((np.arange(k) % 5) - 2).astype(np.float32)
        q *= scale
        key /= scale
        value *= np.float32(3)
    value *= np.array([0.75, 1, 1.25, 1.5], np.float32)[None, None, :, None, None]
    g = rng.uniform(-1.5, -0.03125, (P, P, T, R)).astype(np.float32)
    beta = rng.uniform(0, 1, g.shape).astype(np.float32)
    g[:, :, 0, 0] = 0
    beta[:, :, 0, 0] = 0
    beta[:, :, 1, 0] = 1
    q, key, value = map(encode, (q, key, value))
    if case == "zero":
        for a in (q, key, value):
            a.reshape(-1)[1::2] = 0x8000
    return dict(Q_CANON=q, K_CANON=key, V_CANON=value, G_CANON=g, BETA_CANON=beta, S0=state)


def validated(data, k, v):
    if k not in (16, 32) or v != 16:
        raise ValueError("Unsupported recurrent operator dimensions")
    shapes = dict(
        Q_CANON=(P, P, T, k),
        K_CANON=(P, P, T, k),
        V_CANON=(P, P, T, R, v),
        G_CANON=(P, P, T, R),
        BETA_CANON=(P, P, T, R),
        S0=(P, P, R, k, v),
    )
    if set(data) != set(shapes):
        raise ValueError("Missing immutable input")
    result = {}
    for name, shape in shapes.items():
        a = data[name]
        if a.shape != shape:
            raise ValueError("Incomplete input shape")
        if name in ("Q_CANON", "K_CANON", "V_CANON"):
            a = decode(a)
        elif a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError("Require finite FP32 decay, beta and initial state")
        result[name] = a.astype(np.float64)
    if (result["G_CANON"] > 0).any() or (result["G_CANON"] < -2).any():
        raise ValueError("Log decay outside fixed range")
    beta = result["BETA_CANON"]
    if (beta < 0).any() or (beta > 1).any():
        raise ValueError("Beta outside fixed range")
    return result


def normalized(data, k):
    q, key = data["Q_CANON"], data["K_CANON"]
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) / np.sqrt(k)
    key = key / np.sqrt((key * key).sum(-1, keepdims=True) + 1e-6)
    return q, key, np.exp(data["G_CANON"])


def kv(key, state):
    return np.einsum("...k,...hkv->...hv", key, state)


def outer(key, delta):
    return key[..., None, :, None] * delta[..., None, :]


def propagate(previous, previous_error, key, ek, a, ea, value, beta, k):
    """Full real recurrence and operation-by-operation absolute error interval."""
    d = a[..., None, None] * previous
    ed = (
        np.abs(a[..., None, None]) * previous_error
        + ea[..., None, None] * (np.abs(previous) + previous_error)
        + U * (np.abs(a) + ea)[..., None, None] * (np.abs(previous) + previous_error)
        + Q32
    )
    memory = kv(key, d)
    em = (
        kv(np.abs(key), ed)
        + kv(ek, np.abs(d) + ed)
        + gamma(2 * k) * kv(np.abs(key) + ek, np.abs(d) + ed)
        + 2 * k * Q32
    )
    delta = beta[..., None] * (value - memory)
    e_delta = (
        np.abs(beta[..., None]) * em
        + gamma(2) * np.abs(beta[..., None]) * (np.abs(value) + np.abs(memory) + em)
        + 2 * Q32
    )
    state = d + outer(key, delta)
    e_state = (
        ed
        + outer(np.abs(key), e_delta)
        + outer(ek, np.abs(delta) + e_delta)
        + gamma(2) * (np.abs(d) + ed + outer(np.abs(key) + ek, np.abs(delta) + e_delta))
        + 2 * Q32
    )
    return state, e_state, delta, e_delta


def output_bound(q, eq, state, es, k):
    output = kv(q, state)
    error = (
        kv(np.abs(q), es)
        + kv(eq, np.abs(state) + es)
        + gamma(2 * k) * kv(np.abs(q) + eq, np.abs(state) + es)
        + 2 * k * Q32
    )
    return output, error + UB * (np.abs(output) + error) + QB


def evidence(k, v, inputs_raw, actual_raw, canonical):
    data = validated(inputs_raw, k, v)
    shapes = dict(
        Q_HISTORY=(P, P, T, k),
        K_HISTORY=(P, P, T, k),
        A_HISTORY=(P, P, T, R),
        DELTA_HISTORY=(P, P, T, R, v),
        STATE_HISTORY=(P, P, T, R, k, v),
        Y_HISTORY=(P, P, T, R, v),
        FINAL_STATE=(P, P, R, k, v),
    )
    if actual_raw.keys() != shapes.keys() or canonical.keys() != inputs_raw.keys():
        raise ValueError("Missing full intermediate/state or immutable evidence")
    actual = {}
    for name, shape in shapes.items():
        a = actual_raw[name]
        if a.shape != shape:
            raise ValueError("Incomplete state/output shape")
        if name == "Y_HISTORY":
            a = decode(a)
        elif a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError("Require finite FP32 intermediate/state history")
        actual[name] = a.astype(np.float64)
    exact = True
    for name, before in inputs_raw.items():
        after = canonical[name]
        if after.dtype != before.dtype or after.shape != before.shape:
            raise ValueError("Immutable input shape/precision changed")
        exact &= before.tobytes() == after.tobytes()
    exact &= (
        actual_raw["FINAL_STATE"].tobytes()
        == actual_raw["STATE_HISTORY"][:, :, -1].copy().tobytes()
    )
    q, key, a = normalized(data, k)
    eq = gamma(2 * k + 24) * np.abs(q) + (2 * k + 24) * Q32
    ek = gamma(2 * k + 24) * np.abs(key) + (2 * k + 24) * Q32
    ea = 16 * U * np.abs(a) + 16 * Q32
    checks = [(actual["Q_HISTORY"], q, eq), (actual["K_HISTORY"], key, ek)]
    checks.append((actual["A_HISTORY"], a, ea))
    previous = data["S0"]
    ideal_previous = previous.copy()
    ideal_error = np.zeros_like(previous)
    for frame in range(T):
        qa = actual["Q_HISTORY"][:, :, frame]
        ka = actual["K_HISTORY"][:, :, frame]
        aa = actual["A_HISTORY"][:, :, frame]
        state = actual["STATE_HISTORY"][:, :, frame]
        value, beta = data["V_CANON"][:, :, frame], data["BETA_CANON"][:, :, frame]
        local_state, es, delta, edelta = propagate(
            previous,
            np.zeros_like(previous),
            ka,
            np.zeros_like(ka),
            aa,
            np.zeros_like(aa),
            value,
            beta,
            k,
        )
        checks.append((actual["DELTA_HISTORY"][:, :, frame], delta, edelta))
        checks.append((state, local_state, es))
        local_output, eout = output_bound(qa, np.zeros_like(qa), state, np.zeros_like(state), k)
        checks.append((actual["Y_HISTORY"][:, :, frame], local_output, eout))
        ideal_previous, ideal_error, _, _ = propagate(
            ideal_previous,
            ideal_error,
            key[:, :, frame],
            ek[:, :, frame],
            a[:, :, frame],
            ea[:, :, frame],
            value,
            beta,
            k,
        )
        checks.append((state, ideal_previous, ideal_error))
        ideal_output, ideal_output_error = output_bound(
            q[:, :, frame], eq[:, :, frame], ideal_previous, ideal_error, k
        )
        checks.append((actual["Y_HISTORY"][:, :, frame], ideal_output, ideal_output_error))
        previous = state
    flattened, expected, bound = (
        np.concatenate([row[i].ravel() for row in checks]).astype(np.float64) for i in range(3)
    )
    arrays = dict(
        **{"input_" + k: v for k, v in inputs_raw.items()},
        **{"canonical_" + k: v for k, v in canonical.items()},
        **actual_raw,
        actual=flattened,
        expected=expected,
        exact_state_match=np.array(exact),
    )
    return arrays, np.abs(flattened - expected), bound
