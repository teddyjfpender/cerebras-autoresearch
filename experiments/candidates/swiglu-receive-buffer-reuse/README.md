# swiglu-receive-buffer-reuse

Remove redundant X0/X1/W1 clearing: every logical X0 element is copied from the frame or computed V, and full even-sized receives overwrite X1/W1 before use. Padded words may retain prior values but never enter arithmetic. Canonical weight copies and complete state/output evidence remain fixed.

Reject missing/reordered arithmetic, logical reads from stale padding, any six-case intermediate/output/state or shutdown failure, protected-control changes, speedup below1.05x or memory above1.10x. Full fresh paired seeds are mandatory. Cross-cell claims require complete suite and worst-cell guards; never relax the fixed bounds.
