# Native-width recurrent head group — proposed contract

Staging only, with no device qualification or performance result. Finish the
current four-shape BF16 suite and the scaled recurrent-GDN challenge first.
This next workload keeps native K128/V128 head dimensions for one key head and
its three value heads. The complete model has additional key-head groups and
surrounding projections/convolution/output operations outside this operator.

Two proposed profiles retain the same BF16 Q/K/V/output, FP32 normalization,
log-decay/beta inputs and FP32 state contract as the scaled operator:

| Tokens | Logical state per value head | V columns per PE | Shards per value head | Compute grid |
| --- | --- | ---: | ---: | --- |
| 4 | 128 x128 FP32 | 8 | 16 | 8 x6 =48 PEs |
| 8 | 128 x128 FP32 | 4 | 32 | 12 x8 =96 PEs |

Each PE owns the entire K dimension and one V slice of one value head, so local
state updates/readouts need no cross-PE reduction. Q/K are resident replicated
inputs; beta/log-decay are replicated within each value-head shard group. Every
physical canonical copy is bit checked. Preserve every token's complete logical
FP32 state and BF16 output, normalized operands, decay, delta and final live state.
The baseline may compute repeated normalized operands; the logical operator and
all stored replicas must agree. Timing includes the initial-state copy, every
normalization/decay/state/read operation and all full snapshots.

Use the already fixed FP32/BF16 local and full-chain error propagation, with
K=128, before any device run. Each profile fixes its own PE budget; differences
in token count/area are not a speedup comparison. Fresh paired speedups compare
baseline and candidate within the same profile. Reconstruct complete logical
128 x128 states from column shards for an independent FP64 check, then compare
with the pinned upstream recurrent fallback. The native group is a model-shaped
operator, not a full layer/model or a TPS result.
