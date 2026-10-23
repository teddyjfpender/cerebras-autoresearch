# Resident recurrent gated delta rule — fixed contract v1

Fixed before device bringup; staged for registration. No device measurement or
model performance claim exists yet. The current BF16 SwiGLU evaluator remains
frozen while its campaign executes.

The target configuration uses the Qwen3_5 architecture, including 16 key heads
and 48 value heads. The [pinned primary reference](https://github.com/huggingface/transformers/blob/129e8dc306932532e0b9e702d1fb4d5d7771c2a4/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L424)
is recorded in `kernels/attention/recurrent-gdn/reference.json`:
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

The following bounds were fixed and host-tested before device bringup. Let u=2^-24,
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

## Explicit state/error propagation

For ideal previous S with componentwise error E, ideal normalized k with error
Ek, and ideal decay a with error Ea, all products below involving k/delta are
outer products and `k @ D` contracts the key axis. Absolute-value expressions
are componentwise and all error terms are nonnegative. Define:

```
D = a*S
ED = abs(a)*E + Ea*(abs(S)+E)
     + u*(abs(a)+Ea)*(abs(S)+E) + q32
memory = k @ D
EM = abs(k) @ ED + Ek @ (abs(D)+ED)
     + gamma(2*K)*((abs(k)+Ek) @ (abs(D)+ED)) + 2*K*q32
delta = beta*(V-memory)
Edelta = abs(beta)*EM
         + gamma(2)*abs(beta)*(abs(V)+abs(memory)+EM) + 2*q32
Snew = D + outer(k,delta)
ESnew = ED + outer(abs(k),Edelta) + outer(Ek,abs(delta)+Edelta)
        + gamma(2)*(abs(D)+ED+outer(abs(k)+Ek,abs(delta)+Edelta)) + 2*q32
O = q @ Snew
EO = abs(q) @ ESnew + Eq @ (abs(Snew)+ESnew)
     + gamma(2*K)*((abs(q)+Eq) @ (abs(Snew)+ESnew)) + 2*K*q32
EY = EO + ub*(abs(O)+EO) + qb
```

The full ideal chain starts with exact S0 (E=0), propagates ESnew through all
four tokens, and enforces every state plus BF16 output against those intervals.
Independently enforce normalized Q/K and exp against Eq/Ek/Ea. Local delta/state
checks use the actual verified previous state, K and decay with zero input
interval errors in the same formulas. Local BF16 reads use the actual saved
state and Q with zero state/input intervals. Thus a loose outer chain bound
cannot replace the local operation checks. Every bound is strictly positive.

The final live FP32 STATE must match the last saved state word-for-word. Every
canonical input and initial-state bit is immutable. The separately copied raw
16-bit timer words must match the saved core, then reconstruct positive 48-bit
elapsed clocks with wrap handling. The saved core covers all 16 compute PEs;
configuration alone or a host arithmetic fixture cannot qualify this contract.

The independent oracle has a separate transition-matrix cross-check:
`Snew = a*(I-beta*k*k^T)@S + beta*k*V^T`. This host check is algebraically
independent of the staged scalar CSL traversal. It does not constitute an
execution of the upstream PyTorch model or a full model correspondence proof.
