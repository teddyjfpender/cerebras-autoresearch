# Column-map descriptor base updates

Retain the reference Mt-wide column map. Preload its fixed output and X descriptor extents once per local product, then update only their base addresses for every K term instead of reloading full DSDs. Preserve auto-advancing output columns, exact FP16 K order and complete invariant-checked resident state. Use only proven compute descriptor bank1.

Reject stale descriptor state, omitted/reordered work, any six-case output/state or shutdown failure, changed protected timing/repeat/PE contract, speedup below1.05x or memory above1.10x. Retain only after a full matched fresh paired seed.
