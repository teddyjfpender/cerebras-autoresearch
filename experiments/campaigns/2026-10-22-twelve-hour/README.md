# Twelve-hour research and dogfood session

User-authorized window: **2026-10-22 17:53:34 UTC through
2026-10-23 05:53:34 UTC** (21:53:34 through 09:53:34 in Asia/Muscat).
Continue the complete four-track implementation goal while researching actual
CSL improvements and checking the append-only TSV evidence trail.

## Work sequence

1. Complete six-case qualification of all eight inference-block profiles under
   the frozen evaluator. Preserve failures and raw outputs/state/timestamps.
2. Reuse matched qualification receipts for candidate screens, full validation
   and fresh paired confirmation. Never promote a screen or failed simulator exit.
3. Review bottlenecks and try further bounded CSL candidates. Keep rejected
   ideas and distinguish shape-specific results from suite-wide improvements.
4. Check TSV growth, terminal-receipt coverage, artifact integrity, board
   derivation and the accepted Git prefix. Recover only after the active writer
   stops; keep a single simulator/journal writer.
5. Investigate the next important benchmark: **resident reusable GEMM**.
   Current repeated execution has a correctness failure, so cold one-shot scores
   cannot establish warmed throughput. Study counter/buffer state, then define
   distinct resident input frames, every output and canonical-state checks,
   timed reset/movement and a fresh sequence oracle before optimizing.
6. Export compact, source-bound results and a completion audit. State numerical,
   context/shape, memory and timing boundaries. No hosted/model/hardware TPS claim.

The first frozen campaign uses
`experiments/campaigns/inference-blocks/qualification-plan.json` and an ignored
checkpoint in `runs/campaigns/inference-blocks-qualification/`. Its bounded
follow-up imports exact baseline paths into the optimization campaign. Neither
process liveness nor a planned manifest is execution proof.

New benchmark development starts in `experiments/incubator/` while those trusted
oracles are frozen. Integration changes need a new evaluator track and new
matched baselines for subsequent retention; existing source capsules are kept.
Stop or pivot on concrete numerical/resource failures. Do not fill time by
repeating unchanged measurements without a remaining verification question.


## Initial dogfood findings

- At 18:04 UTC, the journal contained all 88 terminal V2 receipts, with no
  missing events. Full artifact replay and the accepted Git prefix passed;
  `results/reports/2026-10-22-inference-blocks-journal-audit.json` anchors the
  corresponding prefix. Live receipts were correctly excluded.
- The host fell below the 1 GiB free-space gate (243 MiB available). Native
  unit tests exposed it before the next campaign launch. Managed uv pruning
  removed reproducible package cache entries and restored 7.2 GiB free.
  SDK binaries, numerical arrays, logs, capsules and accepted results were kept.
- New GEMM candidates test direct strided output-column vectors and contiguous
  row-major accumulation with a final transpose. Both preserve the public ABI,
  communication, precision and protected timing controls. They remain unmeasured.
- The next benchmark is staged under `experiments/incubator/reusable-gemm/`.
  The missing repeat reset for `x_shift_step` is a concrete source lead; working
  buffer restoration and queue/pointer state still require SDK investigation.


## Qualification milestone: 19:01 UTC

Six of eight profiles have full six-case baselines under the unchanged
`2c885ac648dd5bc1d63c2b522de9f71a9e5619f027e84501762a2b8ef20b9082` evaluator.
Both FFN GEMM shapes, both medium FFN GEMV shapes and both gated MLP widths
passed. Decoder context qualification is now running. Exact paths and digests
remain in the qualification checkpoint and automatically populated TSV.
These are qualified baselines, not optimization results.


## Numerical replay milestone

A separate SDK NumPy auditor recomputed 42 cases from seven full independent
baseline receipts, including ctx4 decoder outputs and state. Expected arrays,
fixed bounds, exact-state flags and raw integer clocks matched. See
`results/reports/2026-10-22-inference-blocks-numerics-initial.json` for source-bound
results. Five SDK-side negative replay tests reject forged expected values,
bounds, outputs, timestamps, input precision and early cache corruption.
`make sdk-test` now includes both driver and replay contract checks.

## Qualification and first optimization iteration

All eight baselines qualified (48 SDK cases), and the complete raw numerical
replay passed. The three initial GEMM candidates were screened on each FFN
shape; none met the 1.05x cycle / 1.10x memory gate. Their rejected measurements
remain in the TSV. The medium GEMV screens are next.

A second bounded iteration is queued after the first worker: strided vector
packing for GEMM, vector elementwise normalization/activation for gated MLP,
combined map expansion/bank alignment for medium GEMV, and per-head context
vectors plus across-head RoPE reuse for decoder. Protected timers/sequence
controls are unchanged in all eight proposed cells. Their actual
compile/numerical/performance outcomes remain pending. The follow-up reserves
two hours before the session deadline for the next benchmark and final audit.

The staged reusable-GEMM evaluator passed 69 host tests, nine SDK replay checks
and five packing/state checks. It is isolated from the active evaluator and has
not compiled or qualified a device program. A terminal-campaign exporter was
also dogfooded on the eight real qualification receipts: all 48 cases were
artifact-audited into an immutable local report.

## First retained gain: 20:14 UTC

`gated-mlp.h16-f32` passed full six-case validation and fresh paired confirmation:
**158,849 → 49,097 whole-sequence cycles (3.2354x, 69.09% fewer cycles)**.
Program-object memory grew from 14,096 to 14,208 bytes/PE (+0.79%). Both seeds
produced the same maximum cycle counts. The candidate vectorizes every gate/up/
down projection and preserves RMSNorm, FP32 SiLU, residual and all four frames.
The automatic decision is
`runs/v2/20261022T201404Z-confirmation-0d5d512d/result.json`.
This is a confirmed simulator result for this one profile. The larger MLP and
decoder cells remain pending; no dense-profile improvement was retained by the
first screen batch.


## Quality gates and evidence reuse

The live tree passes 72 host tests. The isolated next evaluator passes 81 host
checks, including incremental recovery, complete auxiliary-Python fingerprinting,
failed/cancelled step metadata and selection of the fastest confirmed result
within each evaluator track. Its earlier SDK NumPy checks passed nine replay
and five sequence packing/state tests; device compilation is still pending.
The complete 24-case raw replay of the first retained MLP result also passed.

The queued second iteration reuses matched confirmation baselines from the first
completed campaign. Their hashes and source identities are checked and imports
are counted separately from new SDK attempts. It adds map-over-K and explicit
four-row GEMM candidates to investigate the unsuccessful first local-loop ideas.

## Larger MLP retained: 20:36 UTC

`gated-mlp.h32-f64` passed all six cases at both seeds:
**559,178 → 114,829 sequence cycles (4.8697x, 79.46% fewer cycles)**.
Program-object memory is 33,168 → 33,344 bytes/PE (+0.53%). The decision is
`runs/v2/20261022T203652Z-confirmation-03cf791e/result.json`. Its complete
24-case raw replay passed too. The two decoder contexts are now under test.

## First decoder retained: 21:04 UTC

`decoder.h16-f32-ctx4` passed six cases at both seeds, and its full four-arm raw
replay passed 24 cases, including every token output and cache snapshot.
**366,243 → 146,366 sequence cycles (2.5022x, 60.04% fewer cycles)**; program
memory fell from 27,968 to 27,024 bytes/PE (-3.38%). The automatic decision is
`runs/v2/20261022T210426Z-confirmation-c20a1d9c/result.json`.
The selected candidate combines vector projections, D=8 constant RoPE frequencies
and one softmax normalization per slot. The longer decoder remains pending.

The second iteration also includes two FP32 SiLU polynomial candidates. Their
30,001-point numerical primitive probe passed, but their CSL performance and
complete-layer accuracy still require actual runs. The proposed source preserves
protected controls in all 16 planned candidate/profile screens across eight cells.

## Longer decoder retained and second iteration started: 21:43 UTC

`decoder.h16-f32-ctx12` passed all six cases at both seeds:
**494,929 → 229,482 sequence cycles (2.1567x, 53.63% fewer cycles)**.
Program memory fell from 33,120 to 32,192 bytes/PE (-2.80%). The decision is
`runs/v2/20261022T214257Z-confirmation-76c5a3c1/result.json`.
The first iteration is complete with four retained composed-block gains. Its
44-receipt terminal report preserves unsuccessful dense-kernel hypotheses too.

The second iteration started automatically after the first controller exited.
The first vector-pack GEMM-up screen is 1.2739x faster, with +8.99% program
memory. This is a provisional one-case screen; full and fresh validation are
required before retention. The evaluator and all queued candidates remain frozen.

## Full GEMM screen validation and status hardening: 22:09 UTC

The second-iteration GEMM-up vector-pack candidate passed all six cases at
**23,230 → 18,236 cycles (1.2739x)**, with 12,096 → 13,184 program bytes/PE.
This remains a provisional full measurement while the fresh paired seed runs.
The map-over-K idea failed compilation; its separately named pointer correction
is unmeasured and has its own future campaign plan. The old candidate is frozen.

Future campaign launchers now save PID/user/boot/start-time identity. Read-only
status distinguishes a matching live worker, an exited/reused PID and an
unverifiable legacy/foreign-environment checkpoint. Four Linux checks passed,
including a real child terminating. The live tree passes 76 host tests and the
staged evaluator 87 (one Linux-only skip on Mac, separately executed in the VM).
The trusted live and staged evaluator hashes still match the integration guards.

Reusable GEMM preparation separates weight/buffer reuse, compute mapping, and
their combination. All nine candidate/profile overlay checks preserve protected
sequence/snapshot/timer functions. Device compilation and numerical results are
still pending. The five remaining historical independent baseline shapes also
have an explicit finite qualification plan; planning does not count as coverage.

## Evidence reuse for complete suites

The new suite assembler passed seven host gate tests. A read-only replay of
existing first-iteration evidence meets both seeds' complete-family gates:
MLP geometric mean 3.9693x (worst cell 3.2354x), decoder 2.3231x (worst cell
2.1567x). These previews are not journaled suite confirmation events. After
the active writer stops, assemble all four arms per family and record the
standard core suite confirmation, preserving exact child receipt hashes.

A live concurrency probe confirmed the assembler refuses another writer's
simulator lease before creating a V2 receipt or changing the TSV. It imports
existing full measurements and reports zero new SDK attempts. Its suite means
summarize separate profile runs, with per-profile source/harness identities.

## First dense kernel retained: 22:29 UTC

`meshgemm.cold-prefill-ffn-up-scaled` confirmed **23,230 → 18,236 cycles**
(**1.2739x, 21.50% fewer**) at both six-case seeds. Program memory is
12,096 → 13,184 bytes/PE (+8.99%), within the fixed 10% limit. The decision
is `runs/v2/20261022T222937Z-confirmation-e5a490d4/result.json`; its separate
24-case raw numerical/timestamp replay passed. The second iteration now tests
the down projection. This is a cold, fixed-shape result; no complete FFN suite
or warmed-throughput gain has been established.

The journal contains all 139 terminal receipts, with no missing events. The
active down-projection receipt remains excluded. The latest local quality
gates are 83 live host tests and 94 staged host tests (one Linux-only skip on
Mac; all four process-status checks passed separately in the VM).


## New MLP incumbents and resource checkpoint: 23:49 UTC

Both polynomial MLP candidates passed fresh full pairing and independent
24-case raw replays: H16 **158,849 → 39,911 cycles (3.9801x)**; H32
**559,178 → 96,303 cycles (5.8064x)**. Their original numerical bounds and full
four-frame work remain unchanged. Program memory grew 0.91% and 0.58%.
The confirmed dense-up result remains 1.2739x; decoder results remain
2.5022x / 2.1567x. The down-GEMM and medium-GEMV refinements failed retention.

At 23:14 UTC the disk guard cancelled the decoder refinement. Its partial
receipt and every earlier measurement remain preserved. The SDK worker and
guard have exited. The original cancelled campaign checkpoint is immutable;
a future resume must copy it to a new state path before continuing.

Metadata-only suite assembly then recorded complete standard confirmations:
MLP geometric mean **4.8073x**, decoder **2.3231x**, with zero new SDK attempts.
The journal now covers all **169 terminal receipts**, with no missing/live
receipts. Full artifact and accepted-Git-prefix verification passed. The live
host checks passed 86 tests and the staged evaluator 97, each with one Linux-only
Mac skip. The production SDK disk reserve remains 1 GiB; metadata-only writes
use the same exclusive writer lock with a separate 64 MiB reserve.

The dedicated VM uses only 1.3 GiB of host storage; trimming its unused guest
blocks recovered about 13 MiB. Host free space remains below the SDK reserve
and continues falling. Reproducible package caches were pruned earlier; a
checksum-protected public crate-source cleanup removed nothing because its
checksum manifests were unavailable. No SDK binaries or research evidence were
deleted. Additional host disk space is required before further simulation.

The new reusable-GEMM evaluator is exported as a portable, unapplied patch and
hash manifest in `experiments/incubator/reusable-gemm/`. Its device programs and
candidate hypotheses remain uncompiled and unqualified. The batched MLP pack
and corrected cold map-K hypotheses are also unmeasured. These preparations do
not count as benchmark execution or performance gains.


## Session blocked by host capacity

After three consecutive resource audits, the session is blocked below the
1 GiB SDK reserve, with no simulator or campaign worker running. The committed
results and all raw evidence remain intact. Resume requires additional host
disk capacity; the twelve-hour request has not been completed. The cancelled
F1 refinement and the next reusable-GEMM device qualification remain pending.
