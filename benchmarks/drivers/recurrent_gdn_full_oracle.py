"""All sixteen native key-head groups, using fixed per-group FP64 bounds."""

import numpy as np
import recurrent_gdn_full_single_oracle as single

G, R, K, V, T = 16, 3, 128, 128, 2
NAMES = ("Q_CANON", "K_CANON", "V_CANON", "G_CANON", "BETA_CANON", "S0")
STAGES = (
    "Q_HISTORY",
    "K_HISTORY",
    "A_HISTORY",
    "DELTA_HISTORY",
    "STATE_HISTORY",
    "Y_HISTORY",
    "FINAL_STATE",
)


def inputs(tokens, seed, case):
    if tokens != T:
        raise ValueError("Full-head contract requires a two-token decode pair")
    children = np.random.SeedSequence(seed).spawn(G)
    groups = [
        single.inputs(T, int(s.generate_state(1, dtype=np.uint64)[0]), case) for s in children
    ]
    return {name: np.stack([g[name] for g in groups]) for name in NAMES}


def evidence(tokens, data, actual, canonical):
    if (
        tokens != T
        or set(data) != set(NAMES)
        or set(canonical) != set(NAMES)
        or set(actual) != set(STAGES)
    ):
        raise ValueError("Incomplete full-head contract")
    if any(a.shape[0] != G for a in (*data.values(), *actual.values(), *canonical.values())):
        raise ValueError("All sixteen key-head groups must be present")
    pieces = []
    errors = []
    bounds = []
    for group in range(G):
        arrays, error, bound = single.evidence(
            T,
            {n: a[group] for n, a in data.items()},
            {n: a[group] for n, a in actual.items()},
            {n: a[group] for n, a in canonical.items()},
        )
        pieces.append(arrays)
        errors.append(error)
        bounds.append(bound)
    labels = []
    offsets = [0]
    for group, piece in enumerate(pieces):
        labels.extend(f"group{group}.{name}" for name in piece["check_labels"])
        lengths = np.diff(piece["check_offsets"])
        offsets.extend((np.cumsum(lengths) + offsets[-1]).tolist())
    arrays = dict(
        **{"input_" + n: a for n, a in data.items()},
        **{"canonical_" + n: a for n, a in canonical.items()},
        **actual,
        actual=np.concatenate([p["actual"] for p in pieces]),
        expected=np.concatenate([p["expected"] for p in pieces]),
        check_labels=np.array(labels),
        check_offsets=np.array(offsets, np.int64),
        exact_state_match=np.array(all(bool(p["exact_state_match"]) for p in pieces)),
    )
    return arrays, np.concatenate(errors), np.concatenate(bounds)
