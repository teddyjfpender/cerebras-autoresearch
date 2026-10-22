# MeshGEMV

Baseline: public MeshInfra/WaferLLM, pinned in `config/sources.lock.json`.
Use `candidate-init` to obtain editable CSL with its baseline fingerprint;
keep `.deps/waferllm` pristine.

Six profiles cover square, FFN up/down aspect ratios, and larger tile footprints.
`benchmarks/drivers/dense.py` checks every replicated output row against an FP64
GEMV of quantized FP16 inputs, across six input families. The bound accounts
for local accumulation and reduction rounding. Raw output and integer timestamp
words are retained, including failing numerical output.

The built-in `unroll` expands the local FMA map while preserving arithmetic
order. `bank-alignment` changes weight/result placement, adds four explicit
padding bytes per PE, and may incur further compiler padding. Every variant
must satisfy its own measured cycle/memory gates; alignment previously failed
to improve the tested cells.

The checked-in `experiments/candidates/gemv-unroll` demonstrates a shape-specific
workspace. Its V1 result is historical; V2 uses the local elapsed-cycle contract
in `docs/benchmarks/metrics.md`. Source licenses are in `third_party/licenses/`.
