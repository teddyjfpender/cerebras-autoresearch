# gemm-column-unroll

Approach: `meshgemm.column-unroll`. Initial profile: `meshgemm.cold-tiny`.

Expand each four-column FMA map using the same weight scalars and DSR address progression to reduce map dispatch overhead.

Falsification: Reject any numerical/footprint failure, <1.05x matched cycle speedup, or >1.10x program-object memory; require a fresh-seed pair.

Shape-specific: True. Full timers and repetition controls remain fixed.
See the [dogfood campaign report](../../../docs/experiments/2026-10-22-dogfood.md)
for results; this directory alone is not evidence of a gain.

Dense sources derive from WaferLLM under Apache-2.0; attribution and license are in
`../../../third_party/NOTICE.md` and `../../../third_party/licenses/waferllm-LICENSE`.
