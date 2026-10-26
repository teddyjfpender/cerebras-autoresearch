# resident-gemm-pack-dsr-state

Hoist the fixed Nt-wide weight source descriptor once per K slice, reuse it across every local row with explicit accumulator descriptors, retain timed vector packing and invariant-backed frame residency. Test whether descriptor setup reduction beats compiler-generated setup without changing FP16 accumulation order.

All arithmetic, output packing, communication, canonical-input checks and full four-frame working-state snapshots remain inside the fixed contract. No timer or sequence-control edits.
