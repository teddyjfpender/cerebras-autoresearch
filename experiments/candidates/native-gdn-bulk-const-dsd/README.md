# Bulk FP32 state dataflow

The repeat-DSD accumulator pattern is present in Cerebras SDK25-point stencil's
reduceBuffer. This candidate evaluates all K*V products using FP32 multiply/add,
then four partial sums. It removes scalar row-loop/descriptor setup overhead.
The unchanged local/full-chain oracle must accept every rounding difference.

| Phase | Current-frame history slot | State and evidence |
| --- | --- | --- |
| Entry | Dead; previous frames untouched | Live previous STATE, immutable S0/QKV/g/beta |
| Memory projection | All decayed-state times key products | Every K term reduced through4 partial V-vectors |
| Outer update | All key times delta products | Every FP32 STATE element updated |
| Readout | All updated-state times query products | Complete BF16 output, original conversion |
| Protected save_frame | Entire slot overwritten with STATE | All prefix/delta/output/state snapshots preserved |

All operations are blocking. No future work escapes the protected timer and no
warmup/cache output is used. R=1 is a physical-column-shard constraint; logical
value heads remain complete in the fixed native layout. No performance result
exists until an actual SDK attempt, full/fresh validation and artifact replay.

The initial mutable-descriptor version exceeded the SS register file. This new
workspace keeps the same arithmetic/lifetimes and declares immutable descriptor
templates const; the backing FP32 arrays remain mutable. No failed source is edited.
