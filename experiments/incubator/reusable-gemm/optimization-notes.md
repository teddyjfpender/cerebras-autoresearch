# Reusable GEMM hypotheses after qualification

Status: design notes. The resident device program must qualify before any
candidate below can establish a gain. Keep cold and instrumented sequence
scores in separate tracks.

## Separate compute and state reuse

Measure the unchanged compute with fewer redundant working-buffer resets,
the contiguous-column/map-over-K computation with all reference resets, and
their combination. The latter should not receive credit for state reuse until
all four outputs and every canonical/working-W0 snapshot pass at both seeds.
The prepared candidate creator is an operational staging helper; actual
workspaces must come from the integrated adapter's `candidate-init`.

## Reuse W1 as final transpose scratch

The current contiguous-column design allocates a separate row-major accumulator
and packs it into the fixed column-major `res_tile` ABI. A later memory-focused
candidate could accumulate row-major values directly into `res_tile`, then use
the already allocated working `W_1_tile` as transpose scratch after the final
communication/compute step has quiesced. Copy the complete packed result back
into `res_tile` before the protected output snapshot.

Required conditions:

- Require `Kt >= Mt` at compile time so W1 can hold all `Mt*Nt` output elements.
- Prove all final asynchronous sends/receives have completed before touching W1.
- Preserve canonical W0 and its exact per-frame snapshot.
- Prove each subsequent frame overwrites W1 before reading it as weights.
- Keep the full packing and copy-back work inside the timed sequence.
- Remove a duplicate accumulator clear only when the reference result clear
  covers the same entire buffer.

This trades an extra vector copy for lower live storage. It may lose cycles,
help the memory gate, or enter a measured cycle/memory tradeoff front. A passing
numerical screen alone does not establish safe state reuse or retention. Make a
new candidate after the basic mapped implementation is measured; preserve it
while testing this separate storage-lifetime hypothesis.

## Larger fused consumers

A future composed workload could let its next operator consume the row-major
accumulator directly. That would need a new end-to-end contract covering the
consumer, movement, nonlinearities, state and all outputs. The current fixed ABI
requires the transpose. Measure the complete chain before attributing a kernel
layout gain to a larger decoder or model.
