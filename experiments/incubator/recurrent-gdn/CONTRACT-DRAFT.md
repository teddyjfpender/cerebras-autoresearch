# Resident recurrent gated delta rule — proposed contract v1

This is staging only. No device measurement or model claim exists yet. The
current BF16 SwiGLU evaluator remains frozen while its campaign executes.

The target configuration uses the Qwen3_5 architecture, including 16 key heads
and 48 value heads. The pinned primary reference is recorded in `reference.json`:
Transformers `torch_recurrent_gated_delta_rule` and its caller explicitly convert
to FP32, normalize Q/K, apply query scaling, decay the state, compute the delta,
update the state and read the result. This benchmark will isolate that recurrent
operator after QKV/convolution and decay/beta projections; those preceding
operators and the following gated normalization/output projection are outside
its interface. A complete model or hybrid decoder needs them implemented too.

Each of 16 fixed compute PEs owns one key head and its three value heads. Four
distinct resident tokens run in one launch from a nonzero immutable initial
state. Initial cells use K16/V16 and K32/V16 per head; these are scaled state
dimensions with the model's 1:3 key/value-head grouping. No inter-PE reduction is
required for this head layout. Precision, head count, grid, state size, token
count and complete state snapshots are fixed per profile.

Inputs are BF16 Q/K/V, FP32 log decay g, FP32 beta in [0,1], and FP32 S0. Q/K
are shared across three value heads. FP32 normalization uses epsilon 1e-6; query
is additionally divided by sqrt(K). All recurrent arithmetic/state is FP32;
the output per value head is rounded once to BF16. There is no intermediate
BF16 state rounding. The exact real-arithmetic equations of the quantized inputs
are, for each token and value head:

```
q = Q / sqrt(sum(Q*Q) + 1e-6) / sqrt(K)
k = Kinput / sqrt(sum(Kinput*Kinput) + 1e-6)
a = exp(g)
D = a * S_previous
memory = k @ D
delta = beta * (V - memory)
S = D + outer(k, delta)
O = q @ S
Y = BF16_RNE(O)
```

The timer includes loading S0 into working state, every token, normalization,
exp, every requested state/dot-product term, BF16 output conversion, and all
output/intermediate/state stores. Save Q/K normalized values, decay, delta,
every full FP32 recurrent state, and every BF16 output for every token. Require
immutable Q/K/V/g/beta/S0 bits, including signed zeros. Use one simulator thread,
fresh process per numerical family and clean shutdown before readback.

Bounds must be finalized and tested before device bringup. Let u=2^-24,
q32=2^-149, ub=2^-8, qb=2^-133, gamma(n)=n*u/(1-n*u). Require finite arrays and
the following fixed normalization/exp envelopes:

```
Eq = gamma(2*K+24)*abs(q) + (2*K+24)*q32
Ek = gamma(2*K+24)*abs(k) + (2*K+24)*q32
Ea = 16*u*abs(a) + 16*q32
```

The 1e-6 epsilon is part of the workload. For state arithmetic, propagate
elementwise absolute errors through every decay, dot product, subtraction,
beta multiplication and outer-product update using gamma(2*K) for multiply/add dot products and gamma(2) for two-operation
updates. Fused FP32 implementations fit the same bound.
Enforce local checks conditioned on the actual verified previous state and
normalized operands, plus a separate full-chain bound propagated from S0 and
ideal FP64 normalization/decay. BF16 output gets the FP32 read error plus
`ub*(abs(O)+read_error)+qb`. The final oracle must document each formula and
include cancellation, beta=0/1, decay=1, zero Q/K, nonzero S0 and all token states.
No tolerance can be selected after observing device errors.

Retention requires six numerical families, matched baselines, fixed memory and
cycle gates, fresh paired seed, and complete-suite worst-cell guards. Publish
cycles, state memory, operations/cycle, PE spread, numerical error fractions and
host experiment cost. Configuration or a host-only oracle is not execution proof.
