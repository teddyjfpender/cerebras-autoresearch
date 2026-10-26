# Cold prefill follow-up

After eliminating the duplicate output tile, separate aligned output and weight-buffer bank addresses with two-halfword prefixes on both rotating W buffers. Preserve their original logical shapes, communication pointers, complete row-major FMAs and timed gather/copy ABI restoration. Test whether reduced memory-port contention improves both 64-row cold FFN directions over the already measured gather implementation.

Reject any outstanding communication at buffer reuse, truncated outputs, missing FP16 FMA/reduction terms, changed timer or PE budget, numerical or clean-shutdown failure, speedup below 1.05x, memory above 1.10x or failed six-case fresh paired confirmation. This destructive scratch reuse is only valid for the fixed zero-warmup single-iteration cold contracts; it is not a resident/repeated GEMM candidate.
