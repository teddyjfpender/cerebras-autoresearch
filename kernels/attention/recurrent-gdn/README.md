# Scalar recurrent gated delta reference

Independent implementation of the mathematical recurrence, with BF16 Q/K/V,
FP32 normalized operands/state, three value heads per key head, four tokens,
and every full state saved inside the timer. Both K16/K32 profiles passed full six-case device qualification, numerical and
core/timestamp replay. The separate vector-state candidate is freshly confirmed;
see results/reports/2026-10-23/recurrent-gdn/.
