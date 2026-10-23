"""Native-width, one key-head group specialized to the two-token full-head contract: FP64 recurrence and fixed error intervals.

Logical arrays have one key head and three value heads. Physical column sharding
is a separate bit-checked contract in recurrent_gdn_native_layout.py.
"""

import numpy as np
from common import CASES
from recurrent_gdn_oracle import Q32, U, gamma, output_bound, propagate
from resident_swiglu_bf16_oracle import decode, encode

K, V, R = 128, 128, 3
TOKENS = (2,)


def inputs(tokens, seed, case):
    k, v, T = K, V, tokens
    if tokens not in TOKENS or case not in CASES:
        raise ValueError("Unsupported recurrent operator profile/family")
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1, 1, (T, k)).astype(np.float32)
    key = rng.uniform(-1, 1, q.shape).astype(np.float32)
    value = rng.uniform(-1, 1, (T, R, v)).astype(np.float32)
    state = rng.uniform(-0.125, 0.125, (R, k, v)).astype(np.float32)
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
            q[frame, frame % k] = 1
            key[frame, (frame + 1) % k] = 1
    elif case == "mixed_scale":
        scale = np.exp2((np.arange(k) % 5) - 2).astype(np.float32)
        q *= scale
        key /= scale
        value *= np.float32(3)
    value *= np.linspace(0.75, 1.5, T, dtype=np.float32)[:, None, None]
    g = rng.uniform(-1.5, -0.03125, (T, R)).astype(np.float32)
    beta = rng.uniform(0, 1, g.shape).astype(np.float32)
    g[0, 0] = 0
    beta[0, 0] = 0
    beta[1, 0] = 1
    q, key, value = map(encode, (q, key, value))
    if case == "zero":
        for a in (q, key, value):
            a.reshape(-1)[1::2] = 0x8000
    return dict(Q_CANON=q, K_CANON=key, V_CANON=value, G_CANON=g, BETA_CANON=beta, S0=state)


def validated(data, tokens):
    k, v, T = K, V, tokens
    if tokens not in TOKENS:
        raise ValueError("Unsupported recurrent operator dimensions")
    shapes = dict(
        Q_CANON=(T, k),
        K_CANON=(T, k),
        V_CANON=(T, R, v),
        G_CANON=(T, R),
        BETA_CANON=(T, R),
        S0=(R, k, v),
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


def evidence(tokens, inputs_raw, actual_raw, canonical):
    k, v, T = K, V, tokens
    data = validated(inputs_raw, tokens)
    shapes = dict(
        Q_HISTORY=(T, k),
        K_HISTORY=(T, k),
        A_HISTORY=(T, R),
        DELTA_HISTORY=(T, R, v),
        STATE_HISTORY=(T, R, k, v),
        Y_HISTORY=(T, R, v),
        FINAL_STATE=(R, k, v),
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
    exact &= actual_raw["FINAL_STATE"].tobytes() == actual_raw["STATE_HISTORY"][-1].copy().tobytes()
    q, key, a = normalized(data, k)
    eq = gamma(2 * k + 24) * np.abs(q) + (2 * k + 24) * Q32
    ek = gamma(2 * k + 24) * np.abs(key) + (2 * k + 24) * Q32
    ea = 16 * U * np.abs(a) + 16 * Q32
    checks = [(actual["Q_HISTORY"], q, eq), (actual["K_HISTORY"], key, ek)]
    checks.append((actual["A_HISTORY"], a, ea))
    labels = ["q_normalized", "k_normalized", "decay_exp"]
    previous = data["S0"]
    ideal_previous = previous.copy()
    ideal_error = np.zeros_like(previous)
    for frame in range(T):
        qa = actual["Q_HISTORY"][frame]
        ka = actual["K_HISTORY"][frame]
        aa = actual["A_HISTORY"][frame]
        state = actual["STATE_HISTORY"][frame]
        value, beta = data["V_CANON"][frame], data["BETA_CANON"][frame]
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
        checks.append((actual["DELTA_HISTORY"][frame], delta, edelta))
        checks.append((state, local_state, es))
        local_output, eout = output_bound(qa, np.zeros_like(qa), state, np.zeros_like(state), k)
        checks.append((actual["Y_HISTORY"][frame], local_output, eout))
        ideal_previous, ideal_error, _, _ = propagate(
            ideal_previous,
            ideal_error,
            key[frame],
            ek[frame],
            a[frame],
            ea[frame],
            value,
            beta,
            k,
        )
        checks.append((state, ideal_previous, ideal_error))
        ideal_output, ideal_output_error = output_bound(
            q[frame], eq[frame], ideal_previous, ideal_error, k
        )
        checks.append((actual["Y_HISTORY"][frame], ideal_output, ideal_output_error))
        labels.extend(
            f"token{frame}.{name}"
            for name in ("delta_local", "state_local", "y_local", "state_full", "y_full")
        )
        previous = state
    flattened, expected, bound = (
        np.concatenate([row[i].ravel() for row in checks]).astype(np.float64) for i in range(3)
    )
    arrays = dict(
        **{"input_" + k: v for k, v in inputs_raw.items()},
        **{"canonical_" + k: v for k, v in canonical.items()},
        **actual_raw,
        check_labels=np.array(labels),
        check_offsets=np.cumsum([0] + [row[0].size for row in checks], dtype=np.int64),
        actual=flattened,
        expected=expected,
        exact_state_match=np.array(exact),
    )
    return arrays, np.abs(flattened - expected), bound
