# bf16-swiglu-projection-dispatch

Use contiguous FP32 row accumulators and hoisted BF16 weight descriptors for wide expansions, with timed FP32 packing and array-base resets on shorter projections. Preserve all mixed-FMA terms/order, BF16 stage rounding and complete state.

Reject skipped/reordered projection terms, BF16 accumulation, missing stage quantization, stale logical reads, numerical/state/shutdown failures, protected-control changes, speedup below 1.05x or memory above 1.10x. Require six families, fresh matched seed and complete suite guards for cross-shape claims.
