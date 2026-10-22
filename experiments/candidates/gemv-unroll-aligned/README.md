# gemv-unroll-aligned

Approach: `meshgemv.unroll-alignment`. Initial profile: `meshgemv.square64`.

Combine literal FMA expansion with weight/result alignment to test bank-conflict reduction beyond unrolling alone.

Falsification: Reject any numerical/footprint failure, <1.05x matched cycle speedup, or >1.10x program-object memory; require a fresh-seed pair.

Shape-specific: True. Full timers and repetition controls remain fixed.
See the [dogfood campaign report](../../../docs/experiments/2026-10-22-dogfood.md)
for results; this directory alone is not evidence of a gain.

Dense sources derive from WaferLLM under Apache-2.0; attribution and license are in
`../../../third_party/NOTICE.md` and `../../../third_party/licenses/waferllm-LICENSE`.
