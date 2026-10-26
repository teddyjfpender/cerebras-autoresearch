# Four-term column map with explicit scalar weights

Group four consecutive FP16 K terms per column. Keep only one implicit map input and load the other three scalar weights through the current group pointer, avoiding multi-input map DSR interference. Four independently advancing output descriptors preserve K order; source DSRs1/4/5/6 are free after the shift, concurrent two-hop transfers use2/3. Residency resets retain full per-frame invariant checks.

Reject missing/reordered terms, changed fixed timer/precision/PE controls, any six-case output/state/shutdown failure, or memory above1.10x. Retain only full fresh paired evidence at1.05x or better.

## Measured outcome

Compilation failed on all three profiles because assigning an element pointer to an unknown-length pointer was rejected by this SDK. No numerical device validation occurred.

See the [immutable campaign report](../../../results/reports/2026-10-23-resident-gemm-iteration3-campaign.json). Failed receipts remain in the TSV; this source is not retained.
