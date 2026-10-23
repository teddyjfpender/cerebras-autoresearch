# Full native recurrent head configuration — proposal

No registration, SDK device execution or performance result exists yet. This
follow-up preserves native K128/V128 and all16 key-head groups/48 value heads
from the pinned Qwen3.8 configuration in config/targets/qwen3.8-27b.json. It uses
a fixed two-token decode pair with nonzero initial state, full intermediate/state
validation, BF16 Q/K/V/output and FP32 state arithmetic on768 PEs (32x24).

The two-token workload is distinct from the existing one-group T4/T8 profiles.
It keeps full-head simulator iteration cost bounded while exercising a stateful
sequence; it does not inherit their baselines. Finish native one-group validation
and freeze a candidate before any full-head device observations. New oracles,
layout budgets, source/profile/harness identities and baselines are required.

Every group owns three value heads, each V-column sharded across16 PEs with all
128 K entries local. Group/head/column ownership is derived from global row-major
PE index. All physical canonical copies, normalized operand/decay replicas, both
full logical state histories and final live state must be checked. The timer
includes initial-state copy, both complete tokens and every full snapshot.

Inputs are synthetic projected/conv Q/K/V plus precomputed g/beta. Convolution,
projections, gated output normalization, output projection, complete decoder/model,
production Cerebras kernels and hardware/hosted TPS remain outside this operator.
