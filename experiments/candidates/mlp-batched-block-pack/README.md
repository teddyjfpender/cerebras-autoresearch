# Blocked weight packing across independent MLP frames

Unmeasured candidate for the two resident MLP widths. B=8 and F divisible by
T*B are compile-time requirements. All inputs are resident before one timed
launch. The protected T-call loop stays unchanged; each call computes disjoint
neuron blocks across all T frames. Every call performs work and all frame
outputs complete within the original whole-sequence timer.

A single H*B FP32 scratch tile is reused for gate, up and down projection
packing. Source X, scale and weight arrays are read-only. Every frame receives
its own RMSNorm and all nonlinear, residual and projection work. SiLU uses the
SDK exp function, preserving the original math contract. Packing cost is timed.

This tests batch data reuse for independent MLP inputs. Stateful decoder token
ordering is a separate contract. No numerical/performance result is claimed
until full cases, memory checks and fresh paired confirmation pass.
