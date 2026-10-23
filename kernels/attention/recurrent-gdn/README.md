# Scalar recurrent gated delta rule staging

Independent implementation of the mathematical recurrence, with BF16 Q/K/V,
FP32 normalized operands/state, three value heads per key head, four tokens,
and every full state saved inside the timer. No device qualification yet.
