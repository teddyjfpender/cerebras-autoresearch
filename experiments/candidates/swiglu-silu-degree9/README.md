# swiglu-silu-degree9

Evaluate SiLU using a degree-nine FP32 polynomial in g squared for |g|<=4, retaining overflow-safe SDK exp outside. Exhaustive finite FP16 gates passed a prescribed0.00025 host relative-error screen; this is not hardware proof. Preserve FP32 activation/product, FP16 V storage and the original full-chain bound.

Reject missing/reordered arithmetic, logical reads from stale padding, any six-case intermediate/output/state or shutdown failure, protected-control changes, speedup below1.05x or memory above1.10x. Full fresh paired seeds are mandatory. Cross-cell claims require complete suite and worst-cell guards; never relax the fixed bounds.
