# resident-gemm-map2-state

Fuse 2 consecutive K slices into one column map with independent advancing accumulator and fixed X descriptors. Preserve exact per-output FP16 K order while reducing map-loop overhead. Shift-only DSRs are reused only after shift completion; in-flight two-hop communication DSRs 2/3 remain untouched. Combine with invariant-backed resident frame resets.

All arithmetic, output packing, communication, canonical-input checks and full four-frame working-state snapshots remain inside the fixed contract. No timer or sequence-control edits.

## Measured outcome

Compiled but failed simulator execution on all three profiles. The proposed grouped-map acceleration did not qualify.

See the [immutable campaign report](../../../results/reports/2026-10-23-resident-gemm-iteration2-campaign.json). Failed receipts remain in the TSV; this source is not retained.
