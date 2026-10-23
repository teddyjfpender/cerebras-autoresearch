# Native BF16 resident SwiGLU research

The separate BF16/FP32-accumulator contract requires fresh baselines. Qualify the
small six-case cell first, then scaled FFN. Preserve every compile, numerical or
shutdown failure. Both shapes retain four distinct resident batches and all
G/U/V/D/Y outputs and canonical/working weight state on 16 compute PEs.

Seed 20261033 is the initial arm; new candidates require full validation and
fresh paired seed 20261034 before retention. Historical FP16 scores remain on
their own evaluator and precision track. See the
[contract](../../../docs/benchmarks/resident-swiglu-bf16.md).
