# Descriptor reuse and vector normalization

Adds one reciprocal per query head and a vector output multiply. The d64-ctx128 profile is confirmed at 1.816x across six numerical cases and fresh paired seeds 20261023/20261024. Program-object memory is 43,552 versus 44,064 bytes/PE.

The FP32 contract, complete output/state checks, cache appends, PE budget,
collectives and full-sequence coordinator timer stay fixed. Qualification is
scoped to measured profiles; creation-time metadata is not a current score.
See [stage-one evidence](../../../results/reports/2026-10-23-attention-sdk-stage1.json).
