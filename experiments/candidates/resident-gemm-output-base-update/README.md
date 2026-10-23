# Output descriptor base-only updates

Use only the already-working compute descriptor bank1: preload Nt-wide output descriptor extents once per product, then update their base addresses for each row instead of reloading complete DSDs. Keep the hoisted weight vector, exact FP16 term order, timed packing and invariant-backed frame residency. Avoid borrowing communication descriptor banks.

Reject simulator/shutdown failure, omitted arithmetic terms, changed protected timers/repetitions/PEs, six-case output/state failures, less than1.05x speedup or over1.10x memory. Require a fresh matched full six-case pair before retention.
