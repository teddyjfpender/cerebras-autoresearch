# MeshGEMV experiment

Baseline: `MeshInfra/WaferLLM`, pinned by `research/kernel_sources.json`.
Upstream CSL is Apache 2.0; its license is included here. The experiment runner
copies the source into each isolated run rather than modifying the dependency.

The local host verifier uses six input families: positive random, signed random,
zeros, cancellation, one-hot, and mixed scales. It compares every replicated
output against an FP64 GEMV of the exactly quantized FP16 inputs. The fixed
forward-error contract is recorded in `validation.json` along with timings.
This tests the small kernel's FP16 arithmetic; it does not validate Qwen BF16.

The first candidate changes the resident weight/result placement to avoid
read-bank conflicts. In the original 64×64 run, the compiler placed `W_tile`
at `0x1260` and `res_tile` at `0x1460`, both at byte offset zero modulo eight.
The [SDK banking documentation](https://sdk.cerebras.ai/csl/language/appendix)
describes the offset relationship required to avoid conflicts for SIMD reads.

`--variant bank-alignment` aligns the result array and places the weights four
bytes into an aligned structure. It adds four explicit padding bytes per PE;
the compiler can also add alignment padding. PE count, dimensions, precision,
host weight symbol, and arithmetic remain fixed. Only measured results can
establish whether this placement helps. Compiler-rejected prototypes remain in
the run ledger and are not performance results.

`--variant unroll` replaces the local `@map` with an explicit sequence of
`@fmach` instructions, one for each local input. It keeps the same DSR loads,
advancing weight address, arithmetic order, reductions, and timed repetitions.
It trades instruction space for less loop overhead. Larger shapes can exhaust
instruction memory; no claim extends beyond the measured shapes.

Use `--screen` for positive inputs only. A screen is provisional and cannot
pass the comparison gate. The `challenge` command requires all six cases,
at least a 1.05× cycle speedup, and a fresh paired confirmation. This threshold
was selected during exploratory setup; it is a local kernel retention rule,
not a model TPS target. Candidates below it are rejected for that shape.

The timing decoder is currently the upstream one. Report its simulated cycle
span and preserve raw timestamps; hardware clock calibration and full Qwen
decoder integration remain separate work.
