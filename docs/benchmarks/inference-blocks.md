# Inference block challenge

The four family suites are `prefill-ffn`, `decode-ffn`, `gated-mlp` and
`decoder`; `inference-blocks` combines all eight profiles.

Four tracks share the normal `research.py run`, `candidate-init`, `challenge`,
comparison, fresh confirmation and automatic TSV journal workflows. Inspect
`list --suite inference-blocks --json` for complete resolved profiles.

| Track | Profiles | Precision / compute budget | Timed score |
| --- | --- | --- | --- |
| Cold FFN prefill | `meshgemm.cold-prefill-ffn-up-scaled`, `meshgemm.cold-prefill-ffn-down-scaled` | FP16, 4x4 PEs | One complete cold GEMM |
| FFN decode projections | `meshgemv.ffn-up-medium`, `meshgemv.ffn-down-medium` | FP16, 4x4 PEs | Resident GEMV/reduction, one warmup and ten repeats |
| Complete gated MLP | `gated-mlp.h16-f32`, `gated-mlp.h32-f64` | FP32, one PE | Four distinct complete frames, no warmup |
| Stateful decoder layer | `decoder.h16-f32-ctx4`, `decoder.h16-f32-ctx12` | FP32, one PE, two attention heads | Four consecutive tokens including every output and cache snapshot store |

The new blocks are complete scaled computations. They are local composition
benchmarks, not a distributed decoder, a language-model sampling loop, or Qwen.
The decoder uses conventional multi-head attention, not Qwen's hybrid linear
attention. Source lives under `kernels/composed/gated-mlp/` and
`kernels/decoder/causal-layer/`; upstream sources remain pristine.

## Gated MLP semantics

For each of four distinct FP32 input vectors x, with FP32 resident weights:

```
n = RMSNorm(x, scale1, epsilon=1e-5)
g = W_gate n
u = W_up n
h = SiLU(g) * u
output = x + W_down h
```

No bias. Both projections, the nonlinear product, down projection, residual and
all four output stores are timed. RMSNorm multiplies a learned scale after
normalizing. H and F refer to hidden and intermediate widths.

## Decoder-layer semantics

For token position p = initial_context + step:

1. RMSNorm the current independent layer input using scale1.
2. Compute Q, K, V with separate dense HxH projections.
3. Rotate adjacent Q/K pairs within each head using RoPE with base 10000,
   frequency `10000**(-2*j/head_dim)` and absolute position p.
4. Write K and V to slot p in the resident cache. Initial keys are already
   rotated; initial values and unused slots are independently supplied inputs.
5. Compute stable causal softmax(Q K^T / sqrt(head_dim)) over slots 0 through p
   independently for each head. No future cache slots participate.
6. Apply the output projection and first residual.
7. RMSNorm with scale2, complete SwiGLU MLP, second residual.
8. Store the token output and a snapshot of **every** K/V cache slot.

The next token reads the cache updated by earlier tokens. Four independent
input vectors are teacher-forced layer inputs: outputs do not replace the next
input vector. This models execution of a layer across a supplied token sequence,
not autoregressive sampling by a complete language model.

Every token output, every full cache snapshot, and the final cache are collected
and checked. Existing context and slots beyond the current position must remain
bit-exact, exposing accidental writes even where final-token outputs are right.
The timer includes cache history instrumentation. It cannot be compared to a
production decoder without that instrumentation.

## Independent numerical contract

`resident_transformer_fp32_v1` uses an independent NumPy FP64 reference from the
exactly quantized FP32 input tensors. It uses NumPy dense products and einsum for
attention rather than mirroring the CSL scalar loops. All intermediates in the
reference remain FP64. The fixed componentwise end-to-end bound is:

```
abs(actual - expected) <= 5e-5 + 5e-5 * abs(expected)
```

This is a fixed accuracy requirement, **not** an analytic forward-error theorem.
Untouched cache slots additionally require identical FP32 bit patterns, including
signed zero and subnormals. A separate `exact_state_match` gate is saved in both
the raw artifact and case receipt; the numerical evidence denominator uses
2^-149 to avoid division by zero. Nonfinite outputs fail. The six families are
positive, signed, zero input, cancellation, one-hot and mixed input scales.
Weights use fan-in scaling and normalization scales span 0.75 to 1.25; the
cancellation fixture uses unit normalization scales and alternating weight
columns. Zero inputs still exercise nonzero initial-cache attention.

Raw inputs, actual and expected outputs/state, bounds and timestamps are saved
even on numerical failure. SDK-side contract tests inject early-output, early
cache, untouched-slot and final-cache defects; final-output-only tests cannot
pass those gates. Fresh seeds are required to retain an optimization.

## Metrics and workflow

Primary: `max_local_cycles` for the whole four-frame/token sequence (the maximum
across six cases). `cycles_per_frame` / `cycles_per_token` divide that value by
four; these are averages across the specified sequence, not steady-state TPS.
The ctx4 and ctx12 profiles end at context 8 and 16 respectively. They expose
context-dependent attention cost. One-PE cycle spread is necessarily zero.

Work counts include multiply/add operations in the three MLP projections and,
for decoder, Q/K/V/output projections and attention dot/value products. They
exclude norm, exp, softmax, RoPE and state copies, all of which remain timed.
Do not interpret this count as total FLOPs or utilization. Program-object memory,
PE cycles, error fraction and SDK subprocess wall time follow the shared rules.

```sh
# Qualify all eight profiles; use a finite budget suitable for the emulator.
tools/sdk-run python3 research.py suite inference-blocks --max-seconds 14400

# Create a hypothesis-bearing editable workspace and optimize one profile.
python3 research.py candidate-init decoder.h16-f32-ctx4 my-decoder
# Fill candidate.json; edit CSL computation while preserving compute/snapshot.
tools/sdk-run python3 research.py challenge decoder.h16-f32-ctx4 \
  --variant candidate --candidate my-decoder --max-seconds 7200

# Cross-shape claims require both contexts with fresh paired suite confirmation.
tools/sdk-run python3 research.py suite decoder --variant baseline --seed 101
```

The local source is the measured baseline, separate from candidate overlays.
Timer/sequence/snapshot control functions are protected. Review other source
changes for equivalent work: text guards do not formally prove semantics.


## Added context and prefill workloads

The original four-track suites remain frozen. `decoder-context-stress` adds
`decoder.h16-f32-ctx28`, a full context 28-to32 sequence with all four cache-prefix
snapshots, unchanged FP32 arithmetic and one PE. `prefill-ffn-batch64` adds
64-row versions of both cold FFN projection shapes on the same 4×4 PE budget.
These are distinct workloads requiring new baselines, not extensions of earlier
measured claims. Distributed attention has its own [GQA contract](decode-attention.md),
including a different coordinator-clock primary metric.
