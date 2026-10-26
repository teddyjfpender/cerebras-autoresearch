# Resident recurrent gated delta-rule plans

Both K16/V16 and K32/V16 now have full six-case baselines, independent numerical
and core/timestamp replay, and a freshly confirmed vector-state candidate across
the complete two-profile suite. The conservative geometric mean is7.350146x
against the independent scalar references. Both keep four resident tokens,
16 key heads and three value heads per key head on16 PEs, BF16 Q/K/V and outputs,
and complete FP32 state history. See results/reports/2026-10-23/recurrent-gdn/.

Initial seed is 20261035; retained candidates require full validation and matched
fresh seed 20261036. This is the recurrent operator after projected/conv QKV and
precomputed decay/beta inputs; complete-model performance remains unmeasured.
