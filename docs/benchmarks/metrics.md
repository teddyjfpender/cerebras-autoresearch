# Measurement contract

## Primary score

For independent-oracle profiles, `max_local_cycles` is the maximum per-PE
`(end_timestamp - start_timestamp) / timed_repeats`, then the maximum across
numerical cases. Decode the 48-bit counter directly from integer words, including
wraparound. Every interval uses one PE's own clock.

This includes local work and communication waits. It is not a synchronized
global makespan. Moving work between PEs can change that relationship;
communication/layout research must also audit synchronized end-to-end timing
before making a broader claim.

V1 subtracted independently captured reference clocks and included a coordinate
adjustment in its cross-PE span. That metric remains in the archive. V2 rejects
V1 receipts for promotion.

| Adapter | Region | Warmups / timed repetitions |
| --- | --- | --- |
| MeshGEMV | Resident GEMV and reduction | 1 / 10 |
| Cold MeshGEMM | One complete GEMM, including communication | 0 / 1 |
| Distributed resident SwiGLU | Four resident batches: all three GEMMs, SiLU/product, residual and every intermediate/state store | 0 / 1 sequence |
| Resident reusable GEMM | Four frames, timed resets/communication, every output and working-weight snapshot | 0 / 1 sequence |
| Resident chain | Matvec, bias, ReLU, residual, full output stores | 1 distinct frame / 4 distinct frames |
| Complete gated MLP / decoder layer | Full four-frame/token sequence and all output/state stores | 0 / 1 sequence |
| Resident recurrent gated delta rule | Four resident tokens, S0 copy, normalization, decay/delta/state/read and all full snapshots | 0 / 1 sequence |
| Native-width recurrent head group | Fixed T4/48PE or T8/96PE sequence; full state, all physical normalization/decay work and every snapshot | 0 / 1 sequence |
| Full native recurrent heads | All16 key/48 value heads, native K/V128, two complete resident tokens, every full state snapshot | 0 / 1 sequence |
| Timed SDK references | Audited upstream region | Profile-specific |
| Untimed SDK references | Numerical assertions | No cycle score |
| Host I/O diagnostics | Emulated transfer interval | No hardware bandwidth claim |

The original GEMM warmup/repeat sequence failed the independent output oracle.
Cold profiles validate the actual timed invocation. The new [resident reusable GEMM](reusable-gemm.md) profiles restore counters and
working buffers inside the timer and check every frame and weight-state boundary.
Their new oracle/sequence contract requires its own baseline qualification; the
original cold profiles and numerical bounds remain separate.

## Additional metrics

| Metric | Meaning | Use |
| --- | --- | --- |
| `mean_local_cycles` | Mean across PEs; maximum across cases | Compare with worst PE |
| `pe_cycle_spread` | Largest minus smallest local interval | Imbalance/waiting signal; not stall attribution |
| `pe_cycles` | Primary interval × compute PE count | Exposes PE budget; not energy |
| `max_program_object_bytes_per_pe` | Maximum `cs_readelf --msize` observation over `out/bin/*.elf` | Default ≤1.10× baseline |
| `work_units_per_cycle` | Declared algorithmic work / primary interval | Within-profile comparison |
| `max_error_fraction` | Error / fixed numerical bound | Must be finite and ≤1 |
| `cycles_per_frame` / `cycles_per_token` | Whole-sequence cycles divided by the profile's declared frame/token count | Fixed sequence average, not steady-state TPS |
| `cycles_per_gemm` | Whole resident GEMM sequence cycles divided by four | Instrumented sequence average for that profile |
| `experiment_wall_seconds` | Sum of measured compiler, case-run, and memory-inspection subprocess durations | SDK step cost; not WSE speed |

`experiment_wall_seconds` excludes source preparation, provenance capture,
artifact hashing, comparison/confirmation, and journal updates. It is not the
complete end-to-end latency of an autoresearch attempt. Preserve console logs
and campaign checkpoints when investigating that additional workflow overhead.

The memory metric describes program objects, not the union of every ELF load,
full-wafer SRAM, routing area, or physical area. Raw observations name every
object/PE. Compute coordinates must match the declared grid. Helper/I/O tiles
and fabric dimensions are fixed by the profile; layout-budget changes need a
distinct profile.

GEMV/GEMM work counts two arithmetic operations per multiply-add. The chain's
units include ReLU comparisons. Communication counts logical delivered elements;
I/O counts logical bytes. These rates do not measure actual link traffic, peak
utilization, clock frequency, bandwidth, or energy.

### FLOP/s and bandwidth derived from simulator cycles

The upstream [single-tile-matvec example](https://github.com/Cerebras/sdk-examples/blob/rel-sdk-2.10.0/benchmarks/single-tile-matvec/run.py)
calculates FLOP/s from an analytical operation count and timestamp cycles, using
an assumed 850 MHz clock. It also calculates bytes/s from analytical memory
access counts, and separately extrapolates a program rectangle to 750×994 PEs.
These are derived rates and a scaling model, not measured hardware counters or
evidence that a communicating workload scales to a full wafer.

For a documented operation count `F`, repetition count `R`, measured cycle
interval `C` and explicitly assumed clock `f`, report `F*R/C` FLOPs/cycle and
`F*R*f/C` estimated FLOP/s. The pinned example's FLOP/s expression omits `iters`
although its timed loop repeats the matvec; audit repetition accounting before
reusing that expression. Preserve the original example and historical results.

Current native recurrent profiles count `3*T*(7*K*V+2*V)` useful state arithmetic
operations: decay, two state reads, delta and update, with FMA counted as two
operations. Normalization, exp, replicated work and all snapshots remain timed
but are excluded from this numerator. Their recorded `work_units_per_cycle`
is therefore a useful state-arithmetic rate, not a count of all executed FLOPs.
It must not be silently relabeled as total FLOPs/cycle. Distinguish algorithmic
bytes from actual SRAM/fabric traffic when adding bandwidth estimates.

## Acceptance

Require unchanged baseline, resolved profile, precision, grid, seed, SDK
manifest, and trusted harness. All six families and artifact digests must pass.
Default gates are 1.05× speedup and ≤1.10× program-object memory. Repeat both
arms with a different seed and unchanged candidate source.
Fresh confirmation also requires identical compiler arguments for each arm.
New runs snapshot the trusted evaluator sources. Confirmation requires identical
evaluator fingerprints across both arms and seeds; archived runs without this
provenance must be rebaselined before retention under this protocol.

Suites require identical complete coverage, geometric mean speedup, and every
cell's regression guard. Default tolerance is 2% (candidate cycles ≤1.02×
baseline in each cell). A good mean cannot hide a regression or missing cell.
A suite comparison needs a fresh confirming suite pair. Do not average raw
cycle counts across workloads.

Resident nonlinear blocks use the separately versioned
[`resident_transformer_fp32_v1` accuracy contract](inference-blocks.md).
Their primary score includes all four frames/tokens, and work units count only
projection/attention arithmetic, while normalization, nonlinearities, RoPE and
cache history stores remain timed.

The distributed [resident SwiGLU chain](resident-swiglu.md) declares
`projection_arithmetic_operations = 6*T*M*H*F`. All nonlinear/residual work and
instrumentation remain timed, but are excluded from that numerator.
`amortized_cycles_per_batch_row` divides whole-sequence cycles by T*M; it is an
instrumented workload average, not the latency of one row or a TPS measurement.

Native BF16 resident FFN profiles have BF16 X/weights and G/U/V/D/Y storage,
FP32 projection accumulators and nonlinear/residual computation, with explicit
rounding at every stage boundary. They have a separate FP32-accumulator/BF16
rounding oracle and their own baselines. Whole-chain timing includes every
conversion. Do not combine their cycle scores with FP16 accumulation leagues.

For `recurrent_gdn`, max local cycles cover the complete four-token operator,
including S0 copy, normalization, exp, state arithmetic and all full snapshots.
`cycles_per_token` is the total divided by four. `state_arithmetic_operations`
counts `16*T*R*(7*K*V+2*V)` scalar arithmetic operations for state decay, memory
read, delta, update and output read; normalization/exp and stores remain inside
the cycle denominator but are excluded from that numerator. This is an operator
metric, not a model token-throughput measurement. Program-object memory, PE
spread and error fractions use the same definitions as the existing tracks.

Recurrent-GDN numerical replay also exports named per-check error fractions for
normalized Q/K, exp, and every token's local/full-chain delta/state/output checks.
The raw named offsets are checked against the frozen oracle before reporting.
