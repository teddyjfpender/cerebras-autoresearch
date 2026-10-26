# Cold prefill lifetime refinement

Accumulate all FP16 reductions in row-major res_tile, then transpose the complete result into dead W_0_tile after its final communication callbacks. Export the named column-major res output from W_0_tile itself so no final scratch-to-output copy is needed. Input W and output res share a physical allocation in disjoint lifetime intervals under the fixed cold one-iteration contract. Preserve all outputs, workload, PE grid and timer boundaries.

Reject any outstanding communication at buffer reuse, truncated outputs, missing FP16 FMA/reduction terms, changed timer or PE budget, numerical or clean-shutdown failure, speedup below 1.05x, memory above 1.10x or failed six-case fresh paired confirmation. This destructive scratch reuse is only valid for the fixed zero-warmup single-iteration cold contracts; it is not a resident/repeated GEMM candidate.
