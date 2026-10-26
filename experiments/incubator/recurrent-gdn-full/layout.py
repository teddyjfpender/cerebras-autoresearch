"""Complete 16-group native head ownership across a fixed768 PE rectangle."""

from dataclasses import dataclass

import numpy as np
from oracle import NAMES, STAGES, G, T
from recurrent_gdn_native_layout import Grid as NativeGrid


@dataclass(frozen=True)
class PairGrid(NativeGrid):
    def __post_init__(self):
        if (self.tokens, self.tile_v, self.width, self.height) != (2, 8, 8, 6):
            raise ValueError("Fixed two-token group layout required")


class Grid:
    width, height, tokens, tile_v = 32, 24, 2, 8
    pes = 768

    def __init__(self):
        self.inner = PairGrid(2, 8, 8, 6)
        if self.pes != G * self.inner.pes:
            raise ValueError("Incomplete full-head PE budget")

    def join(self, arrays):
        a = np.concatenate([x.reshape(self.inner.pes, *x.shape[2:]) for x in arrays])
        return a.reshape(self.height, self.width, *a.shape[1:]).copy()

    def split(self, array):
        if array.shape[:2] != (self.height, self.width):
            raise ValueError("Incomplete full-head physical grid")
        a = array.reshape(G, self.inner.height, self.inner.width, *array.shape[2:])
        return [x.copy() for x in a]

    def mapped(self, array, method, *args, **kwargs):
        if array.shape[0] != G:
            raise ValueError("Missing logical key-head groups")
        return self.join([getattr(self.inner, method)(a, *args, **kwargs) for a in array])

    def key(self, array):
        return self.mapped(array, "key")

    def per_head(self, array):
        return self.mapped(array, "per_head")

    def columns(self, array, state=False):
        return self.mapped(array, "columns", state)

    def initial(self, array):
        return self.mapped(array, "initial")

    def inputs(self, data):
        pieces = [self.inner.inputs({n: a[g] for n, a in data.items()}) for g in range(G)]
        return {n: self.join([p[n] for p in pieces]) for n in NAMES}

    def evidence(self, data, captured):
        pieces = {n: self.split(a) for n, a in captured.items() if n != "times"}
        actual = []
        canonical = []
        for g in range(G):
            a, c, _ = self.inner.evidence(
                {n: x[g] for n, x in data.items()}, {n: x[g] for n, x in pieces.items()}
            )
            actual.append(a)
            canonical.append(c)
        transport = {
            "physical_" + n: captured[n].copy()
            for n in (*NAMES, "Q_HISTORY", "K_HISTORY", "A_HISTORY")
        }
        return (
            {n: np.stack([a[n] for a in actual]) for n in STAGES},
            {n: np.stack([a[n] for a in canonical]) for n in NAMES},
            transport,
        )


def profile(tokens):
    if tokens != T:
        raise ValueError("Fixed two-token full-head contract required")
    return Grid()
