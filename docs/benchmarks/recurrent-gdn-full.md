# Full native recurrent head configuration

Registered contract proposal; no SDK device qualification yet. This workload
uses all16 key heads and48 value heads with native K128/V128 dimensions from
[pinned Qwen configuration](../../config/targets/qwen3.8-27b.json). It is a
fixed two-token resident decode pair, with BF16 Q/K/V/output and FP32 normalized
operands, decay, delta and state. Nonzero canonical initial state is supplied.

| Profile | Key/value heads | K / V | Tokens | V columns per PE | Compute grid |
| --- | --- | --- | ---: | ---: | --- |
| `recurrent-gdn-full.k128-v128-h16-t2` |16 /48 |128 /128 |2 |8 |32x24 =768 PEs |

Global row-major PE index n owns group=n//48, value head=(n//16)%3 and value
columns8*(n%16)..8*(n%16)+7. Each PE retains all128 K entries locally, so every
memory/read dot evaluates all K terms. The complete logical state is48 FP32
matrices of128x128. Full states for both tokens, normalized Q/K, decay, delta,
BF16 output and final live state are reconstructed. Every physical canonical
copy and every per-group normalized/per-head decay replica must match bitwise.
Independent global-index tests cover all768 PEs. All16 groups have independently
seeded inputs; no group is a cached duplicate of another.

The per-group equations and FP32 local/full-chain interval propagation are those
of [the native one-group contract](recurrent-gdn-native.md), specialized explicitly
to two tokens before any device run. No arithmetic bounds are relaxed. All16
groups must pass the exact same fixed checks; labels identify each group/token.
The full raw replay regenerates canonical inputs from the declared seed/family,
checks physical replicas, recomputes every state interval and matches all core
words/timestamps. Missing groups, partial histories, precision changes, altered
inputs, shutdown failures and stale cores reject the case.

The primary max_local_cycles covers initial-state copy, both complete tokens,
all physical normalization/exp/state arithmetic, all output/intermediate/full
state stores and every required term. Host transfers are outside. One thread,
fresh process per case,0 warmups,1 timed sequence,48-bit local timers and clean
shutdown are required. The baseline is independently authored FP32 vector CSL,
not Cerebras production inference code. State arithmetic work units are
16*3*2*(7*128*128+2*128); normalization/exp/stores are excluded from that numerator
but remain timed. Report program-object bytes/PE, PE cycles, imbalance and all
numerical error fractions. Retention requires1.05x speedup, memory growth<=1.10x,
all six families, matched evaluator/SDK/source capsules and a fresh paired seed.

The two-token workload bounds full-head simulator iteration cost. The separate
one-group T4/T8 profiles test deeper recurrence; their different workloads and PE
budgets do not supply full-head baselines. A one-profile full-head result claims
only this fixed contract, without cross-shape or complete-layer generality.

This is the recurrent operator after QKV/convolution and precomputed g/beta,
before gated output normalization/projection. Activations are synthetic. There
is no full decoder/model execution, hardware TPS or hosted-service result.
Broader Model Zoo and serving work is deferred while this operator is the focus.


The pinned upstream PyTorch recurrent fallback passes all six full-head fixtures
under the same fixed mathematical bounds. A separate SDK NumPy replay regenerates
all immutable input bits and checks every group/token state. This establishes
reference correspondence only, without a device speed measurement.
[Upstream report](../../results/reports/2026-10-23/recurrent-gdn-full/upstream-reference.json),
[SDK replay](../../results/reports/2026-10-23/recurrent-gdn-full/upstream-sdk-replay.json).
