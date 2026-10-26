# Cold prefill: reuse a dead weight buffer

Unmeasured candidate for the fixed zero-warmup, one-iteration cold GEMM profiles.
All four communication phases and FP16 FMAs execute unchanged. The row-major
accumulator uses the exported output allocation. Only after the final send and
receive completion barriers does packing overwrite W_0_tile, which is no longer
read by this cold workload. A complete timed copy restores the original
column-major output ABI. Kt >= Mt guarantees scratch capacity.

This is intentionally destructive to an input working buffer after its last use;
it is not suitable for repeated/resident GEMM without preserving canonical inputs.
Six numerical cases, matched fresh seed, memory and cycle gates remain required.

This variant uses a single mem4d descriptor to perform the complete transpose.
The complete transpose is unchanged; this variant visits columns first.

This variant gathers strided columns and writes scratch contiguously. It tests
the opposite 2D traversal orientation under the same full-transpose contract.
