# Vector complete chain

This local CSL workspace extends vector-column accumulation with vector bias,
ReLU, residual addition and full per-frame output stores. It keeps FP32, the
existing intermediate buffer, all five distinct input frames and protected
timers/repetition controls.

Hypothesis and falsification are in `candidate.json`. The workspace is generic;
measured claims remain limited to the small profile actually validated.

See the [dogfood report](../../../docs/experiments/2026-10-22-dogfood.md).
