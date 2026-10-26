# bf16-swiglu-vector-silu6-reuse

Vectorize FP32 degree-six SiLU while reusing dead projection scratch: res_tile holds clamped G then quantized U; row_accum holds the polynomial; one extra FP32 tile holds G squared. U is saved before reuse, down accumulation clears res, all BF16 stage stores and fallback exp remain timed. Measure cycle gain versus scalar composition without exceeding memory gates.

Reject skipped/reordered projection terms, BF16 accumulation, missing stage quantization, stale logical reads, numerical/state/shutdown failures, protected-control changes, speedup below 1.05x or memory above 1.10x. Require six families, fresh matched seed and complete suite guards for cross-shape claims.
