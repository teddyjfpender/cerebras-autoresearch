# swiglu-projection-silu-reuse

Combine projection layout/descriptor dispatch, complete receive-buffer overwrite invariants and bounded FP32 SiLU polynomial. Every projection term, distinct frame, conversion, nonlinear/residual operation, intermediate and state store remains timed and verified under the unchanged contract.

Reject missing/reordered arithmetic, logical reads from stale padding, any six-case intermediate/output/state or shutdown failure, protected-control changes, speedup below1.05x or memory above1.10x. Full fresh paired seeds are mandatory. Cross-cell claims require complete suite and worst-cell guards; never relax the fixed bounds.
