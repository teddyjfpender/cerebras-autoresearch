# Native BF16 resident SwiGLU research

The separate BF16/FP32-accumulator contract requires fresh baselines. Qualify the
small six-case cell first, then scaled FFN. Preserve every compile, numerical or
shutdown failure. Both shapes retain four distinct resident batches and all
G/U/V/D/Y outputs and canonical/working weight state on 16 compute PEs.

Seed 20261033 is the initial arm; new candidates require full validation and
fresh paired seed 20261034 before retention. Historical FP16 scores remain on
their own evaluator and precision track. See the
[contract](../../../docs/benchmarks/resident-swiglu-bf16.md).

The first two-cell suite is confirmed at1.451350x conservative geometric mean,
worst1.404278x, with all48 raw numerical/core cases replayed. The initial vector
screens failed the unchanged memory gate; new receive-scratch and bounded-tail
candidates are separate hypotheses. The compiler-stalled partial attempt is
cancelled history, not a completed measurement. See the
[complete report](../../../results/reports/2026-10-23/resident-swiglu-bf16/initial-suite-confirmed.json).
