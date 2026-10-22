# Cold MeshGEMM

Pinned WaferLLM supplies the distributed CSL and topology/packing helpers.
Five profiles include square, rectangular, and scaled FFN projections.
Independent six-case validation collects the complete output of the actual
timed invocation and compares to FP64 of the quantized FP16 inputs.

These are **cold resident invocations**: inputs have already been copied into
canonical resident buffers, then the program performs one complete GEMM.
The score excludes host input loading and is not a hardware cold-cache metric.

During integration, the original warmup/repeat sequence failed the independent
output oracle. Cold invocation passed without relaxing the bound. See
`docs/research/gemm-state.md`; a valid reusable-state benchmark requires explicit
per-iteration output/state checks before measuring warmed throughput.
