# swiglu-projection-dispatch

Port the confirmed expansion row-vector/weight-descriptor layout and contraction array-base resets into every appropriate chain projection, preserving all H/F FP16 terms/order, timed output packing and stage/frame state.

Reject missing/reordered arithmetic, logical reads from stale padding, any six-case intermediate/output/state or shutdown failure, protected-control changes, speedup below1.05x or memory above1.10x. Full fresh paired seeds are mandatory. Cross-cell claims require complete suite and worst-cell guards; never relax the fixed bounds.
