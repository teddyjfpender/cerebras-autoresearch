# chain-k8-unroll

Approach: `pipeline.scalar-inner-unroll`. Initial profile: `pipeline.k8-n8`.

Literal expansion of the eight reduction terms removes inner-loop indexing and branches while preserving scalar FP32 operation order.

Falsification: Reject any numerical/footprint failure, <1.05x matched cycle speedup, or >1.10x program-object memory; require a fresh-seed pair.

Shape-specific: True. Full timers and repetition controls remain fixed.
See the [dogfood campaign report](../../../docs/experiments/2026-10-22-dogfood.md)
for results; this directory alone is not evidence of a gain.

This workspace derives from the repository's local resident-chain kernel.
