"""4x4-PE resident GEMM packing and full sequence evidence, independent of CSL."""

import numpy as np

ORIGINS = np.array([[0, 1, 2, 3], [1, 3, 0, 2], [2, 0, 3, 1], [3, 2, 1, 0]])


def pack_inputs(p, mt, kt, nt, frames, weights):
    if p != 4 or frames.shape[1:] != (p * mt, p * kt) or weights.shape != (p * kt, p * nt):
        raise ValueError("Sequence input shape/grid disagrees with contract")
    t = frames.shape[0]
    packed_x = frames.reshape(t, p, mt, p, kt).transpose(1, 3, 0, 4, 2)
    weight_tiles = weights.reshape(p, kt, p, nt).transpose(0, 2, 1, 3)
    packed_w = np.empty((p, p, kt, nt), dtype=weights.dtype)
    for row in range(p):
        for col in range(p):
            packed_w[row, col] = weight_tiles[ORIGINS[row, col], col]
    return np.ascontiguousarray(packed_x.reshape(p, p, t, mt * kt)), np.ascontiguousarray(
        packed_w.reshape(p, p, kt * nt)
    )


def unpack_outputs(p, mt, nt, t, raw_values):
    tiles = raw_values.reshape(p, p, t, nt, mt)
    return tiles.transpose(2, 0, 4, 1, 3).reshape(t, p * mt, p * nt).astype(np.float64)


def evidence(p, frames, weights, actual, history, canonical_x_after, canonical_w_after):
    if frames.dtype != np.float16 or weights.dtype != np.float16:
        raise ValueError("Sequence inputs must be quantized FP16")
    t, m, k = frames.shape
    n = weights.shape[1]
    packed_x, packed_w = pack_inputs(p, m // p, k // p, n // p, frames, weights)
    history_shape = (p, p, t, (k // p) * (n // p))
    if history.shape != history_shape or actual.shape != (t, m, n):
        raise ValueError("Incomplete sequence output/state shape")
    if canonical_x_after.shape != (packed_x.size,) or canonical_w_after.shape != (packed_w.size,):
        raise ValueError("Incomplete canonical input snapshot shape")
    if any(value.dtype != np.float16 for value in (history, canonical_x_after, canonical_w_after)):
        raise ValueError("Sequence state must remain FP16")
    expected_history = np.broadcast_to(packed_w[:, :, None], history_shape).copy()
    exact = (
        np.array_equal(canonical_x_after.view(np.uint16), packed_x.ravel().view(np.uint16))
        and np.array_equal(canonical_w_after.view(np.uint16), packed_w.ravel().view(np.uint16))
        and np.array_equal(history.view(np.uint16), expected_history.view(np.uint16))
    )
    x64, w64 = frames.astype(np.float64), weights.astype(np.float64)
    expected = x64 @ w64
    length, u = k + p, 2.0**-11
    if length * u >= 1:
        raise ValueError("Shape exceeds fixed FP16 accumulation bound")
    gamma = length * u / (1 - length * u)
    bound = gamma * (np.abs(x64) @ np.abs(w64)) + length * 2.0**-24
    arrays = dict(
        x=frames,
        w=weights,
        actual=actual,
        expected=expected,
        working_w_history=history,
        expected_working_w_history=expected_history,
        canonical_x_after=canonical_x_after,
        canonical_w_after=canonical_w_after,
        exact_state_match=np.array(exact),
    )
    return arrays, np.abs(actual - expected), bound
