# Resident matvec epilogue

The CSL in `src/` implements a complete FP32 matvec → bias → ReLU → residual
chain. `FUSED=0` stores an intermediate vector and uses a second pass;
`FUSED=1` applies the epilogue while producing each output row.

Both variants use one host launch and the same one-PE footprint. Five distinct
input frames are resident: one warmup and four timed. Every frame's output is
returned and checked, preventing a last-output-only check from hiding omitted
work. The score covers the full chain, including output stores.

The trusted driver is `benchmarks/drivers/pipeline.py`. It compares FP32 inputs
against an FP64 reference with a fixed forward-error bound. This is a small
fusion benchmark, not a transformer implementation or a hardware TPS result.
