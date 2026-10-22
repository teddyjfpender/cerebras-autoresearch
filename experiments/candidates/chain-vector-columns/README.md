# chain-vector-columns

Approach: `pipeline.vector-column-accumulation`. Initial profile: `pipeline.k8-n8`.

Accumulate all output channels with strided FP32 vector FMAs, then execute the complete bias/ReLU/residual epilogue for each frame. This reduces scalar addressing and loop overhead without more PEs or buffers.

Falsification: Reject any numerical/footprint failure, <1.05x matched cycle speedup, or >1.10x program-object memory; require a fresh-seed pair.

Shape-specific: False. Full timers and repetition controls remain fixed.
See the [dogfood campaign report](../../../docs/experiments/2026-10-22-dogfood.md)
for results; this directory alone is not evidence of a gain.

This workspace derives from the repository's local resident-chain kernel.
