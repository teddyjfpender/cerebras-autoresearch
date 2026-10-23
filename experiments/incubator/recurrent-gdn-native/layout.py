"""Lossless native-head column sharding, including checks of every replica."""

from dataclasses import dataclass

import numpy as np
from oracle import K, R, V


@dataclass(frozen=True)
class Grid:
    tokens: int
    tile_v: int
    width: int
    height: int

    def __post_init__(self):
        if (self.tokens, self.tile_v, self.width, self.height) not in ((4, 8, 8, 6), (8, 4, 12, 8)):
            raise ValueError("Only the two fixed native-width layout budgets are valid")
        if self.width * self.height != R * self.shards:
            raise ValueError("Incomplete value-head column coverage")

    @property
    def shards(self):
        return V // self.tile_v

    @property
    def pes(self):
        return self.width * self.height

    def grid(self, array):
        return array.reshape(self.height, self.width, *array.shape[1:]).copy()

    def flat(self, array):
        if array.shape[:2] != (self.height, self.width):
            raise ValueError("Wrong physical grid")
        return array.reshape(self.pes, *array.shape[2:])

    def columns(self, array, state=False):
        # Logical [T,R,(K),V] -> physical [H,W,T,1,(K),tileV].
        tail = (K,) if state else ()
        expected = (self.tokens, R, *tail, V)
        if array.shape != expected:
            raise ValueError("Incomplete logical column array")
        a = array.reshape(self.tokens, R, *tail, self.shards, self.tile_v)
        axes = (1, len(a.shape) - 2, 0, *range(2, len(a.shape) - 2), len(a.shape) - 1)
        a = a.transpose(axes).reshape(self.pes, self.tokens, 1, *tail, self.tile_v)
        return self.grid(a)

    def logical_columns(self, array, state=False):
        tail = (K,) if state else ()
        if array.shape != (self.height, self.width, self.tokens, 1, *tail, self.tile_v):
            raise ValueError("Incomplete physical column array")
        a = self.flat(array).reshape(R, self.shards, self.tokens, *tail, self.tile_v)
        axes = (2, 0, *range(3, len(a.shape) - 1), 1, len(a.shape) - 1)
        return a.transpose(axes).reshape(self.tokens, R, *tail, V).copy()

    def initial(self, array):
        if array.shape != (R, K, V):
            raise ValueError("Incomplete initial state")
        a = array.reshape(R, K, self.shards, self.tile_v).transpose(0, 2, 1, 3)
        return self.grid(a.reshape(self.pes, 1, K, self.tile_v))

    def logical_initial(self, array):
        if array.shape != (self.height, self.width, 1, K, self.tile_v):
            raise ValueError("Incomplete final/initial physical state")
        a = self.flat(array).reshape(R, self.shards, K, self.tile_v)
        return a.transpose(0, 2, 1, 3).reshape(R, K, V).copy()

    def per_head(self, array):
        if array.shape != (self.tokens, R):
            raise ValueError("Incomplete logical per-head values")
        a = np.repeat(array.T[:, None, :], self.shards, axis=1)
        return self.grid(a.reshape(self.pes, self.tokens, 1))

    def logical_head(self, array):
        if array.shape != (self.height, self.width, self.tokens, 1):
            raise ValueError("Incomplete replicated per-head values")
        a = self.flat(array).reshape(R, self.shards, self.tokens)
        if any(a[:, s].tobytes() != a[:, 0].tobytes() for s in range(1, self.shards)):
            raise ValueError("Per-head replicas disagree")
        return a[:, 0].T.copy()

    def key(self, array):
        if array.shape != (self.tokens, K):
            raise ValueError("Incomplete logical key/query")
        return self.grid(np.repeat(array[None], self.pes, axis=0))

    def logical_key(self, array):
        if array.shape != (self.height, self.width, self.tokens, K):
            raise ValueError("Incomplete replicated key/query history")
        a = self.flat(array)
        if any(row.tobytes() != a[0].tobytes() for row in a[1:]):
            raise ValueError("Key/query replicas disagree")
        return a[0].copy()

    def inputs(self, data):
        return dict(
            Q_CANON=self.key(data["Q_CANON"]),
            K_CANON=self.key(data["K_CANON"]),
            V_CANON=self.columns(data["V_CANON"]),
            G_CANON=self.per_head(data["G_CANON"]),
            BETA_CANON=self.per_head(data["BETA_CANON"]),
            S0=self.initial(data["S0"]),
        )

    def evidence(self, data, captured):
        physical = self.inputs(data)
        if not set(physical) <= captured.keys():
            raise ValueError("Missing physical canonical input evidence")
        for name, before in physical.items():
            after = captured[name]
            if (
                after.dtype != before.dtype
                or after.shape != before.shape
                or after.tobytes() != before.tobytes()
            ):
                raise ValueError("A physical canonical input replica changed: " + name)
        actual = dict(
            Q_HISTORY=self.logical_key(captured["Q_HISTORY"]),
            K_HISTORY=self.logical_key(captured["K_HISTORY"]),
            A_HISTORY=self.logical_head(captured["A_HISTORY"]),
            DELTA_HISTORY=self.logical_columns(captured["DELTA_HISTORY"]),
            STATE_HISTORY=self.logical_columns(captured["STATE_HISTORY"], state=True),
            Y_HISTORY=self.logical_columns(captured["Y_HISTORY"]),
            FINAL_STATE=self.logical_initial(captured["STATE"]),
        )
        # Preserve actual physical input and replicated-prefix evidence for a
        # separate raw-artifact replay. No tolerance is used for these checks.
        transport = {"physical_" + name: captured[name].copy() for name in physical}
        transport.update(
            {"physical_" + n: captured[n].copy() for n in ("Q_HISTORY", "K_HISTORY", "A_HISTORY")}
        )
        return actual, {n: a.copy() for n, a in data.items()}, transport


def profile(tokens):
    return (
        Grid(4, 8, 8, 6)
        if tokens == 4
        else Grid(8, 4, 12, 8)
        if tokens == 8
        else Grid(tokens, 0, 0, 0)
    )
