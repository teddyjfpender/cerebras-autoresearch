"""Independent FP64 semantics for complete resident FP32 blocks.

V1 fixed end-to-end tolerance: 5e-5 absolute + 5e-5 relative, including every
written cache value. Untouched cache slots require exact equality (a subnormal
bound avoids division by zero in the shared evidence format). This is a fixed
accuracy contract, not a claimed analytic forward-error bound.
"""

import numpy as np

CONTRACT = "resident_transformer_fp32_v1"
ATOL = RTOL = 5e-5


def inputs(kind, dims, seed, case):
    h, f, t = (dims[k] for k in ("H", "F", "T"))
    rng = np.random.default_rng(seed)
    shapes = {"X": (t, h), "SCALE1": (h,), "WG": (f, h), "WU": (f, h), "WD": (h, f)}
    if kind == "decoder":
        shapes.update(
            SCALE2=(h,),
            WQ=(h, h),
            WK=(h, h),
            WV=(h, h),
            WO=(h, h),
            CACHE=(2, dims["C"], h),
        )
    data = {}
    for name, shape in shapes.items():
        if name.startswith("SCALE"):
            value = rng.uniform(0.75, 1.25, shape)
        else:
            value = rng.uniform(-1, 1, shape)
            if name.startswith("W"):
                value /= np.sqrt(shape[-1])
            if case == "positive":
                value = np.abs(value)
        data[name] = value.astype(np.float32)
    if case == "zero":
        data["X"][:] = 0
    elif case == "cancellation":
        data["X"][:] = 1
        for name, value in data.items():
            if name.startswith("SCALE"):
                value[:] = 1
            elif name.startswith("W"):
                value[:] = np.where(np.arange(value.shape[-1]) % 2, -1, 1) / value.shape[-1]
    elif case == "one_hot":
        data["X"][:] = 0
        data["X"][np.arange(t), np.arange(t) % h] = 1
    elif case == "mixed_scale":
        data["X"] *= np.where(np.arange(h) % 2, 16, 1 / 16).astype(np.float32)
    return data


def rmsnorm(x, scale):
    return x / np.sqrt(np.mean(x * x) + 1e-5) * scale


def gated_mlp(norm, data):
    gate = data["WG"] @ norm
    up = data["WU"] @ norm
    return data["WD"] @ ((gate / (1 + np.exp(-gate))) * up)


def reference(kind, dims, data):
    """No runtime imports, device outputs, or lower precision intermediate values."""
    data = {key: value.astype(np.float64) for key, value in data.items()}
    outputs, history = [], []
    cache = data.get("CACHE")
    h, heads = dims["H"], dims.get("A", 1)
    d = h // heads
    for step, x in enumerate(data["X"]):
        norm = rmsnorm(x, data["SCALE1"])
        residual = x
        if kind == "decoder":
            q, k, v = (data[name] @ norm for name in ("WQ", "WK", "WV"))
            position = dims["L"] + step
            angle = position / (10000.0 ** (np.arange(0, d, 2) / d))
            co, si = np.cos(angle), np.sin(angle)
            for vector in (q, k):
                pairs = vector.reshape(heads, d // 2, 2)
                first, second = pairs[:, :, 0].copy(), pairs[:, :, 1].copy()
                pairs[:, :, 0] = first * co - second * si
                pairs[:, :, 1] = first * si + second * co
            cache[0, position], cache[1, position] = k, v
            keys = cache[0, : position + 1].reshape(position + 1, heads, d)
            values = cache[1, : position + 1].reshape(position + 1, heads, d)
            scores = np.einsum("hd,shd->hs", q.reshape(heads, d), keys) / np.sqrt(d)
            probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            context = np.einsum("hs,shd->hd", probabilities, values).ravel()
            residual = x + data["WO"] @ context
            norm = rmsnorm(residual, data["SCALE2"])
            history.append(cache.copy())
        outputs.append(residual + gated_mlp(norm, data))
    result = {"Y": np.array(outputs)}
    if kind == "decoder":
        result.update(HISTORY=np.array(history), CACHE=cache.copy())
    return result


def evidence(kind, dims, data, actual):
    expected = reference(kind, dims, data)
    names = tuple(expected)
    for name in names:
        if actual[name].shape != expected[name].shape:
            raise ValueError(f"Wrong complete-output shape: {name}")
    bounds = {name: ATOL + RTOL * np.abs(value) for name, value in expected.items()}
    exact_state_match = True
    if kind == "decoder":
        # Initial context and all future slots must retain their original bits.
        for step in range(dims["T"]):
            written = slice(dims["L"], dims["L"] + step + 1)
            mask = np.ones((2, dims["C"], dims["H"]), bool)
            mask[:, written] = False
            bounds["HISTORY"][step][mask] = 2.0**-149
            exact_state_match &= np.array_equal(
                actual["HISTORY"][step].astype(np.float32).view(np.uint32)[mask],
                expected["HISTORY"][step].astype(np.float32).view(np.uint32)[mask],
            )
        mask = np.ones((2, dims["C"], dims["H"]), bool)
        mask[:, dims["L"] : dims["L"] + dims["T"]] = False
        bounds["CACHE"][mask] = 2.0**-149
        exact_state_match &= np.array_equal(
            actual["CACHE"].astype(np.float32).view(np.uint32)[mask],
            expected["CACHE"].astype(np.float32).view(np.uint32)[mask],
        )
    arrays = {"input_" + name: value for name, value in data.items()}
    arrays.update({"actual_" + name: value for name, value in actual.items()})
    arrays.update({"expected_" + name: value for name, value in expected.items()})
    arrays["actual"] = np.concatenate([actual[name].ravel() for name in names]).astype(np.float64)
    arrays["expected"] = np.concatenate([expected[name].ravel() for name in names])
    arrays["exact_state_match"] = np.array(exact_state_match)
    bound = np.concatenate([bounds[name].ravel() for name in names])
    return arrays, np.abs(arrays["actual"] - arrays["expected"]), bound
