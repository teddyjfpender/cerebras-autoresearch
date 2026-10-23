# bf16-swiglu-silu-degree9

Use a degree-nine FP32 polynomial in g squared on |g|<=4 and stable SDK exp outside; preserve quantized BF16 G/U operands and V storage. The separate exhaustive BF16 host screen is preliminary; the fixed full-chain oracle remains mandatory.

Reject skipped/reordered projection terms, BF16 accumulation, missing stage quantization, stale logical reads, numerical/state/shutdown failures, protected-control changes, speedup below 1.05x or memory above 1.10x. Require six families, fresh matched seed and complete suite guards for cross-shape claims.
