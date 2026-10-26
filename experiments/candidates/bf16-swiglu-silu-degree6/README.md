# bf16-swiglu-silu-degree6

Reduce the FP32 SiLU polynomial to degree six using the separate exhaustive BF16 gate screen, reserving one quarter of the original 4*u_b local allowance after normal product rounding. Retain stable exp outside |g|<=4, FP32 arithmetic and every BF16 stage boundary; no bounds change.

Reject skipped/reordered projection terms, BF16 accumulation, missing stage quantization, stale logical reads, numerical/state/shutdown failures, protected-control changes, speedup below 1.05x or memory above 1.10x. Require six families, fresh matched seed and complete suite guards for cross-shape claims.
