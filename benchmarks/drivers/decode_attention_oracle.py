"""Independent dense FP64 semantics for sharded, resident GQA decode attention.

Device layout and distributed reduction order are deliberately absent here.
All cache stores must preserve FP32 bits; every token's complete output is checked.
"""

import numpy as np

CONTRACT = "sharded_gqa_decode_fp32_v1"
ATOL = RTOL = 1e-4


def shapes(dims):
    p, s, d, g, t = (dims[k] for k in ("P", "S", "D", "G", "T"))
    return dict(Q=(t, g, d), NEW_K=(t, d), NEW_V=(t, d), K=(p, s, d), V=(p, s, d))


def validate_inputs(dims, data):
    if not 0 < dims["L"] <= dims["P"] * dims["S"] - dims["T"]:
        raise ValueError("Invalid attention context/capacity")
    expected = shapes(dims)
    if set(data) != set(expected) or any(
        data[k].shape != shape or data[k].dtype != np.float32 for k, shape in expected.items()
    ):
        raise ValueError("Attention input shape or FP32 precision changed")
    if any(not np.isfinite(v).all() for v in data.values()):
        raise ValueError("Attention inputs must be finite")


def inputs(dims, seed, case):
    rng = np.random.default_rng(seed)
    data = {k: rng.uniform(-1, 1, shape).astype(np.float32) for k, shape in shapes(dims).items()}
    if case == "positive":
        data = {k: np.abs(v) for k, v in data.items()}
    elif case == "zero":
        # Uniform attention still requires all causal values and cache appends.
        data["Q"][:] = 0
    elif case == "cancellation":
        data["Q"][:] = np.where(np.arange(dims["D"]) % 2, -1, 1)
        data["K"][:] = data["NEW_K"][:] = 1
        data["V"][:] = np.where(np.arange(dims["S"])[None, :, None] % 2, -1, 1)
    elif case == "one_hot":
        data["Q"][:] = 0
        for t in range(dims["T"]):
            for g in range(dims["G"]):
                data["Q"][t, g, (t + g) % dims["D"]] = 1
    elif case == "mixed_scale":
        # Includes sharply peaked softmax and dissimilar heads, not just scaled values.
        data["Q"] *= np.where(np.arange(dims["G"]) % 2, 64, 1 / 64)[None, :, None]
        data["V"] *= np.where(np.arange(dims["D"]) % 2, 16, 1 / 16)[None, None, :]
    elif case != "signed":
        raise ValueError("Unknown attention case")
    validate_inputs(dims, data)
    return data


def reference(dims, data):
    validate_inputs(dims, data)
    d = dims["D"]
    keys = data["K"].reshape(-1, d).astype(np.float64).copy()
    values = data["V"].reshape(-1, d).astype(np.float64).copy()
    outputs = []
    for step, query in enumerate(data["Q"].astype(np.float64)):
        position = dims["L"] + step
        keys[position], values[position] = data["NEW_K"][step], data["NEW_V"][step]
        scores = (query @ keys[: position + 1].T) / np.sqrt(d)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        outputs.append(weights @ values[: position + 1])
    return dict(
        Y=np.array(outputs), K=keys.reshape(data["K"].shape), V=values.reshape(data["V"].shape)
    )


def evidence(dims, data, actual):
    expected = reference(dims, data)
    if set(actual) != set(expected) or any(
        actual[k].shape != v.shape or actual[k].dtype != np.float32 for k, v in expected.items()
    ):
        raise ValueError("Incomplete attention output/state or changed precision")
    exact = all(
        np.array_equal(actual[k].view(np.uint32), expected[k].astype(np.float32).view(np.uint32))
        for k in ("K", "V")
    )
    arrays = {"input_" + k: v for k, v in data.items()}
    arrays.update({"actual_" + k: v for k, v in actual.items()})
    arrays.update({"expected_" + k: v for k, v in expected.items()})
    arrays["actual"] = np.concatenate([actual[k].ravel() for k in ("Y", "K", "V")]).astype(
        np.float64
    )
    arrays["expected"] = np.concatenate([expected[k].ravel() for k in ("Y", "K", "V")])
    arrays["exact_state_match"] = np.array(exact)
    bounds = np.concatenate(
        [
            (ATOL + RTOL * np.abs(expected["Y"])).ravel(),
            np.full(expected["K"].size + expected["V"].size, 2.0**-149),
        ]
    )
    return arrays, np.abs(arrays["actual"] - arrays["expected"]), bounds
