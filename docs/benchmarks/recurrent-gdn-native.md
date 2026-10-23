# Native-width recurrent gated-delta head group

One key-head group has K128 query/key vectors and three V128 value heads. This
retains native head widths while reducing the model's key-head count to one.
It is a resident recurrent operator after projections/convolution and precomputed
log-decay/beta, before gated output normalization/projection. It is not a layer,
full model, hardware TPS or hosted-inference benchmark.

| Profile | Tokens | Logical FP32 state per head | Value columns per PE | Compute grid |
| --- | ---: | --- | ---: | --- |
| `recurrent-gdn-native.k128-v128-t4` | 4 |128x128 |8 |8x6 =48 PEs |
| `recurrent-gdn-native.k128-v128-t8` | 8 |128x128 |4 |12x8 =96 PEs |

Each PE owns all128 K entries and one V-column slice of one value head. Every
memory/read dot evaluates all K terms locally, so there are no omitted network
reductions. PE linear index is y*width+x; head=index//shards and its first value
column=(index%shards)*tileV. Immutable Q/K are replicated on every PE; each head's
log-decay/beta is repeated over that head's column shards. Every physical copy
must remain bit exact, including BF16 signed zero. All normalized Q/K copies and
per-head decay copies must also agree bit for bit. Independent global-index tests
check the mapping; complete logical state/output histories are reconstructed.

The mathematical recurrence, FP32 local/full-chain error propagation and BF16
output rounding are exactly those in [the scaled contract](recurrent-gdn.md),
with K=128 and the profile's fixed token count. No oracle bounds change. Q/K/V and
Y have BF16 storage; S0, every state, normalized Q/K, decay, delta and arithmetic
are FP32. Initial state is nonzero and supplied canonically. All six deterministic
families include beta0/1, zero Q/K/V, cancellation and mixed scales.

The timer includes initial-state copy, all repeated physical normalization/exp,
every token's complete state operations, and every normalized/delta/output/full
state history store. Host transfers are outside. Full reconstructed FP32 states
and BF16 outputs, final live state, physical immutable/prefix replicas and raw
memcpy/core timers must all agree. Every case uses a fresh single-thread process
and requires clean shutdown. Program-object bytes, PE-cycle cost, imbalance and
per-check numerical errors are reported.

State arithmetic work units are3*T*(7*128*128+2*128). This excludes normalization,
exp, replicated physical work and stores from the numerator; all their costs
remain in the cycle denominator. T4/48PE and T8/96PE are different contracts;
compare each candidate only with its same-profile baseline. Cross-shape claims
require the entire two-cell suite, fresh paired seeds,1.05x speedup/1.10x memory
and worst-cell guards. The baseline is an independent FP32 vector CSL reference derived from the scaled
vector-state candidate, with new native device qualification required. It is not
Cerebras production inference code. Registration is not device qualification.

The pinned upstream PyTorch fallback has separately passed12 native-width
CPU fixtures with every token's state observed, then SDK NumPy replay of the same
raw tensors. See the incubator for SHA-bound source/reference metadata. Device
baseline qualification and optimization require their own receipts.


Host input transfers are conservatively split into complete PE-row groups of at
most32768 transport words per request. This is a software bound, not a claimed
SDK/hardware limit. Every physical canonical word and PE is covered exactly once
before the timed launch; regression tests reconstruct all input bits independently.
Readback uses a small timestamp fence, clean simulator stop and complete raw-word
core extraction, avoiding large blocking output transfers.


The registered oracle's exact source has12 passed upstream fixtures with archived
script/oracle/error-model source capsules. [CPU trace and raw tensor identities](../../results/reports/2026-10-23/recurrent-gdn-native/upstream-reference-v6.json)
and [SDK NumPy bit/numerical replay](../../results/reports/2026-10-23/recurrent-gdn-native/upstream-sdk-replay-v6.json)
are unscored supporting evidence, separate from device baseline qualification.


Both native-width profiles are now fully qualified. Normalization conversion/reuse
confirms 1.251334× geometric mean against the vector baseline, with all48 paired
cases independently replayed. [Evidence](../../results/reports/2026-10-23/recurrent-gdn-native/vector-suite-confirmed.json).


PR1-derived persistent DSR row reuse and bank-offset accumulation freshly confirm
1.991629× on T4 only:115,634→58,060 cycles,24 fully replayed case executions,
all compared output/state bits identical. T8 and full-head claims require new
complete qualification. [Evidence](../../results/reports/2026-10-23/recurrent-gdn-native/pr1-t4-confirmed.json).
