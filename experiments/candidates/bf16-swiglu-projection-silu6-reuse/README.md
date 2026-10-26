# bf16-swiglu-projection-silu6-reuse

Combine FP32 row/column projection dispatch, bounded FP32 degree-six SiLU and complete receive-buffer overwrite invariants. Every logical X0 word is copied and received X1/W1 words are overwritten before use; stale padding never enters arithmetic. All mixed FMA terms, stage rounding, outputs and state remain timed.

Reject skipped/reordered projection terms, BF16 accumulation, missing stage quantization, stale logical reads, numerical/state/shutdown failures, protected-control changes, speedup below 1.05x or memory above 1.10x. Require six families, fresh matched seed and complete suite guards for cross-shape claims.
