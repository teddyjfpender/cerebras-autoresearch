# BF16 resident FFN precision development

Priority after the distributed FP16/FP32 chain qualifies: determine whether this
installed SDK can execute native BF16 operands with FP32 accumulation, then
implement a distinct complete FFN contract. This is closer to the model target's
storage precision than the existing FP16 accumulation profiles.

The current [SDK builtin reference](https://sdk.cerebras.ai/csl/language/builtins#fp16)
describes `--fp16-format=bf16` and `@fp16()`. Documentation alone does not qualify
this installed compiler/runtime. A bounded, exact-bit format/conversion/FMA probe
is required before registering a BF16 benchmark. Do not reuse FP16 scores.

The proposed chain retains four distinct batches, all three resident projections,
SiLU/product, residual, every intermediate and canonical/boundary weight state on
16 compute PEs. BF16 inputs/weights and G/U/V/D/Y storage use FP32 projection
accumulators and FP32 nonlinear/residual arithmetic followed by BF16 rounding.
Host evidence must preserve all raw 16-bit words and independently decode BF16;
NumPy FP16 reinterpretation is transport only, never a numeric oracle.

A fixed BF16 accumulator bound copied from the current FP16 benchmark is invalid
at F=272: `(F+P)*2^-8 >= 1`. Use the distinct FP32-accumulator contract below.
Any registration needs new baselines and precision/state/replay corruption tests.
This remains a scaled FFN without RMSNorm or hybrid attention, not a Qwen decoder.

## Installed SDK capability verified

The [native probe](../../../results/reports/2026-10-23/bf16/native-vector-precision-probe.json)
passed exact BF16 scalar/vector round-to-nearest-even conversions, round trips,
normal/subnormal operands, canonical input bits and BF16-product/FP32-accumulator
FMA. This is a16-value unscored capability fixture with no cycle measurement.
It establishes a viable precision path, not a qualified FFN or model benchmark.
The separate BF16 chain contract/oracle is now registered under
`resident-swiglu-bf16.small` and `.ffn-scaled`; device qualification is pending.
This folder preserves the original staging/probe sources. Active benchmark
sources live under `kernels/composed/resident-swiglu-bf16/`; optimization uses
`candidate-init` workspaces.
