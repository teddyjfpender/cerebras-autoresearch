# Bulk state and vector norm

Every normalization square and sum is FP32. Sixteen independent partial sums and
a four-level FP32 tree replace the scalar128-row loop. The original fixed
normalization and propagated state bounds must accept all rounding differences.
All current-frame history scratch is overwritten before protected storage.
