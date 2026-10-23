# Results and qualification

## Distributed attention — first confirmed SDK improvement

The [GQA challenge](../docs/benchmarks/decode-attention.md) has five configured
profiles. The [stage-one report](reports/2026-10-23-attention-sdk-stage1.json)
confirms **1.816x** on the 4-PE `d64-ctx128` profile after six cases at both
seeds: 380,121 → 209,313 sequence cycles, with program-object memory
44,064 → 43,552 bytes per PE. The independent raw replay checks all 24 cases
from the four full paired runs. This event enters the confirmed leaderboard.
Other profiles still need execution; hardware/model TPS and >5x remain unproven.
The [readiness report](reports/2026-10-23-attention-readiness.json) preserves
the earlier host-only state. Older-machine raw artifacts are absent here;
new-session ledger rows were individually audited against their raw evidence.

## Canonical research journal

[`results.tsv`](results.tsv) automatically records terminal V2 attempts and
decisions. [`leaderboard.json`](leaderboard.json) derives measured bests by
declared approach, a cycle/memory tradeoff front, and confirmed events.
Measured views remain provisional; fresh paired confirmation events appear
separately in the confirmed view.

The historical [ledger setup report](reports/ledger-qualification.json) records 43 events:
40 historical receipts, two replayed comparison decisions, and one live SDK
screen proving automatic logging with an evaluator source capsule. All 48 host
tests and artifact replay passed. This is an unsigned local ledger; see the
[ledger contract](../docs/experiments/ledger.md) for guarantees and limits.

## Four inference tracks — qualified baselines

All eight profiles passed six independent numerical families under one frozen
evaluator: **48 successful SDK case runs**. The
[qualification report](reports/2026-10-22-inference-blocks-qualification.json)
preserves exact receipts, source/profile/harness/SDK identities, compiler argv,
case errors, timing and memory. These are baselines; optimization is in progress.

| Profile | Timed cycles | Program-object bytes |
| --- | ---: | ---: |
| `meshgemm.cold-prefill-ffn-up-scaled` | 23,230 | 12,096 |
| `meshgemm.cold-prefill-ffn-down-scaled` | 31,381 | 12,480 |
| `meshgemv.ffn-up-medium` | 1,790.1 | 17,296 |
| `meshgemv.ffn-down-medium` | 1,754.1 | 17,296 |
| `gated-mlp.h16-f32` | 158,849 | 14,096 |
| `gated-mlp.h32-f64` | 559,178 | 33,168 |
| `decoder.h16-f32-ctx4` | 366,243 | 27,968 |
| `decoder.h16-f32-ctx12` | 494,929 | 33,120 |

GEMM is one cold invocation; GEMV is per timed repetition. MLP and decoder
scores cover the complete four-frame/token sequence, including output/state
stores. Compare matched arms within one profile. The blocks are scaled FP32
local programs; no Qwen/model/hardware TPS claim follows.

Across historical evaluator tracks, the current 34-profile catalog now has
**13 full independent baselines, 14 reference checks and two diagnostics**:
29 exercised profiles. Five independent shapes remain unqualified:
`meshgemm.cold-square32`, `meshgemm.cold-rectangular`, `meshgemv.ffn-down-scaled`,
`meshgemv.square128`, and `pipeline.k64-n32`.

The [complete numerical replay](reports/2026-10-22-inference-blocks-numerics-qualified.json)
independently recomputed all 48 cases from the eight qualified baselines,
including both decoder contexts and their per-token cache state. The earlier
[initial replay](reports/2026-10-22-inference-blocks-numerics-initial.json) covers
the first seven baselines. [Bringup history](reports/2026-10-22-inference-blocks-bringup.json)
preserves earlier screens and compile failures. The
[contracts](../docs/benchmarks/inference-blocks.md) and
[active session](../experiments/campaigns/2026-10-22-twelve-hour/README.md)
describe the optimization and further investigation. Earlier counts below refer
to their original 30-profile tree.

## Inference-block optimization — confirmed results

| Profile | Baseline sequence cycles | Candidate sequence cycles | Speedup | Program bytes/PE |
| --- | ---: | ---: | ---: | ---: |
| `gated-mlp.h16-f32` | 158,849 | 39,911 | 3.9801x | 14,096 → 14,224 |
| `gated-mlp.h32-f64` | 559,178 | 96,303 | 5.8064x | 33,168 → 33,360 |
| `decoder.h16-f32-ctx4` | 366,243 | 127,134 | 2.8808x | 27,968 → 28,144 |
| `decoder.h16-f32-ctx12` | 494,929 | 200,280 | 2.4712x | 33,120 → 33,312 |

In the first iteration, `mlp-vector-projections` passed all six cases at both seeds, with **69.09% and
79.46% fewer cycles**, respectively. The four-arm numerical replays for
[H16](reports/2026-10-22-mlp-h16-confirmed-numerics.json) and
[H32](reports/2026-10-22-mlp-h32-confirmed-numerics.json) each recomputed all
24 baseline/candidate cases and their raw clocks. Retention decisions are
journaled at `runs/v2/20261022T201404Z-confirmation-0d5d512d/result.json` and
`runs/v2/20261022T203652Z-confirmation-03cf791e/result.json`.
The first-iteration ctx4 decoder's `decoder-vector-rope-softmax` passed both six-case arms and
fresh confirmation, with **60.04% fewer cycles** and **3.38% less program memory**.
Its [24-case raw replay](reports/2026-10-22-decoder-ctx4-confirmed-numerics.json)
checks every token output and cache snapshot; retention is recorded at
`runs/v2/20261022T210426Z-confirmation-c20a1d9c/result.json`.

The second-iteration `mlp-silu-polynomial` now improves both MLP widths further:
**74.87% and 82.78% fewer cycles than their scalar references**, with +0.91%
and +0.58% program memory. This combines vector elementwise work with bounded
FP32 polynomial SiLU and an SDK-exp fallback; the original end-to-end error
bound is unchanged. It saves another 18.71% / 16.13% cycles over the first
winners. The [H16](reports/2026-10-22-mlp-h16-polynomial-confirmed-numerics.json)
and [H32](reports/2026-10-22-mlp-h32-polynomial-confirmed-numerics.json) raw
four-arm replays each passed all 24 cases. Decisions:
`runs/v2/20261022T225349Z-confirmation-5195d91f/result.json` and
`runs/v2/20261022T230641Z-confirmation-b1946980/result.json`.

The first-iteration ctx12 decoder also passed six cases at both seeds: **53.63% fewer cycles**
and **2.80% less program memory**. Its
[24-case raw replay](reports/2026-10-22-decoder-ctx12-confirmed-numerics.json)
checks outputs, cache state and timing summaries; retention is recorded at
`runs/v2/20261022T214257Z-confirmation-76c5a3c1/result.json`.

The [completed first-iteration report](reports/2026-10-22-inference-blocks-first-iteration.json)
artifact-audits 44 run/decision receipts. Its dense GEMM/GEMV candidates did not
clear the retention gates; those unsuccessful screens remain in the journal.
These confirmed results cover the complete four-frame/token FP32 workloads on
one PE under the frozen inference-block evaluator. The original twelve-hour session was interrupted. The second iteration was safely
cancelled by the low-disk guard during decoder validation; see the interrupted
checkpoint below. Historical results below retain their original tracks.

### Suite confirmations before decoder refinement

The [source-bound suite report](reports/2026-10-22-inference-blocks-confirmed-suites.json)
records standard core confirmations for both complete families:

| Suite | Profiles | Conservative geometric mean | Worst profile |
| --- | ---: | ---: | ---: |
| `gated-mlp` | 2 widths | 4.8073x | 3.9801x |
| `decoder` | 2 contexts | 2.3231x | 2.1567x |

All four arms per suite import existing six-case measurements with exact receipt
hashes and unchanged evaluator/SDK identities. Each assembled receipt explicitly
records **zero new SDK attempts**. The means summarize separate profile launches;
they do not measure a combined kernel or model throughput.

### Decoder refinement completed — 2026-10-23

`decoder-silu-polynomial` passed both six-case seed arms in both contexts:
**127,134 / 200,280 sequence cycles**, or **2.8808x / 2.4712x** versus the fixed
scalar references. This saves another **13.14% / 12.73%** cycles over the prior
retained decoder. Program memory is +0.63% / +0.58% versus the references, within
the unchanged gate. The original numerical tolerances and complete token/cache
work remain fixed.

The [complete suite confirmation](reports/2026-10-23-decoder-refinement-confirmed-suite.json)
records **2.6681x geometric mean**, worst cell **2.4712x**, at both seeds.
Independent raw replays passed 24 cases each for
[ctx4](reports/2026-10-23-decoder-ctx4-refinement-confirmed-numerics.json) and
[ctx12](reports/2026-10-23-decoder-ctx12-refinement-confirmed-numerics.json).
The [resumed interruption report](reports/2026-10-23-inference-blocks-resumed-interrupted.json)
preserves a server-restart broken pipe and its partial fresh run. The
[durable continuation](reports/2026-10-23-decoder-ctx12-durable-confirmation.json)
reused the accepted initial arm and retried the failed fresh arm in full.
The original cancelled checkpoint and all historical records remain unchanged.

### Second iteration: first dense gain

`meshgemm.cold-prefill-ffn-up-scaled` now has fresh paired confirmation:
**23,230 → 18,236 cycles (1.2739x, 21.50% fewer cycles)** on the unchanged
4x4 compute grid, FP16 M16 K80 N272 cold invocation. Program-object memory
is **12,096 → 13,184 bytes/PE (+8.99%)**. The selected
`gemm-row-major-vector-pack` accumulates contiguous output-column vectors and
packs the full result with strided vector copies, preserving communication
and all reduction terms. The
[24-case raw replay](reports/2026-10-22-gemm-ffn-up-confirmed-numerics.json)
passed for the four initial/fresh baseline/candidate arms. Retention is
recorded at `runs/v2/20261022T222937Z-confirmation-e5a490d4/result.json`.

This is one confirmed cold up-projection cell; the down projection and other
second-iteration cells remain under investigation. It is the session's fifth
confirmed profile gain. No complete-prefill-suite or warmed-throughput claim
follows from this single cell.

### Resource interruption and preserved evidence

At 23:14 UTC the host fell below the 1 GiB disk gate. The guard sent SIGTERM;
the controller preserved the cancelled decoder receipt and all prior gains.
The [interrupted second-iteration report](reports/2026-10-22-inference-blocks-second-iteration-interrupted.json)
audits 36 imported/run/decision receipts, including the partial attempt.
All 159 terminal V2 receipts at that checkpoint were journaled, with zero
missing/live receipts. The subsequent
[resource checkpoint](reports/2026-10-22-inference-blocks-resource-checkpoint.json)
verifies **169 / 169 terminal receipts** after the two suite confirmations,
with full artifact replay, accepted Git prefix, and no live or missing receipts.
At that checkpoint, decoder refinements and reusable-GEMM device qualification
were outstanding. Decoder refinement is now complete as recorded above; the
original eight baseline profiles remain qualified.

## Kernel dogfood campaign — 2026-10-22

The [experiment report](../docs/experiments/2026-10-22-dogfood.md) and
[machine evidence](reports/2026-10-22-dogfood.json) record ten screens over five
profiles and three fresh paired improvements in the main campaign. Two GEMM
ideas were slower; the scaled FFN GEMV gain did not clear the retention gate.
Rejected experiments remain in the journal. A bounded
[epilogue follow-up](reports/2026-10-22-dogfood-epilogue.json) adds an eleventh
screen and a fourth confirmed optimization step, improving the small pipeline
further. The table shows the latest retained implementation per profile.

| Profile | Confirmed cycles | Speedup | Program memory change |
| --- | ---: | ---: | ---: |
| `pipeline.k8-n8` | 971 → 389.25 | 2.4945× | +5.05% |
| `pipeline.k16-n32` | 14,708 → 2,567.25 | 5.7291× | +1.17% |
| `meshgemv.square64` | 264.1 → 240.7 | 1.0972× | +1.45% |

Each retained cell passed all six numerical families and a fresh baseline /
candidate pair. These are fixed-profile results; no complete-suite or model
speed claim follows. The campaign extends measured baseline/reference coverage
to **22 of 30 profiles**, including six independent full baselines. Eight shapes
remain unexecuted. Earlier qualification snapshots below are preserved.

## Initial repository qualification — 2026-10-22

**20 of 30 configured profiles exercised:** four full six-case independent
baselines, fourteen upstream reference checks, and two transfer diagnostics.
Two additional variants passed their full six-case numerical checks.
The other ten shapes were configured and unexecuted in this initial qualification.

- [Qualification report](reports/repository-qualification.json): exact coverage,
  metrics, receipt paths/digests, source identities, failures, and comparisons.
- [Run index](reports/run-index.json): 40 V2 receipts, including failed bringup
  attempts and partial suites. Raw evidence stays in ignored `runs/v2/`.
- [V1 archive](archive/2026-10-22.json): original bootstrap search, preserved
  byte for byte. Its aligned cycle span differs from V2's local elapsed metric.

Coverage combines the initial suite, targeted retries, and a final stencil
retry. The original partial-failure receipts remain unchanged. The report
verified the selected receipts' profile/source/harness identities and artifact
digests; full independent runs also passed the promotion evidence loader.

Local checks: **33 host unit tests**, **4 SDK contract tests**, lint, formatting,
catalog validation, Python/shell syntax, Lima configuration, compatibility entry
points, and local documentation links. Remote CI is configured and was not run.

## Baselines

Cycle regions differ between adapters; compare matched runs within one profile.
An untimed reference has no cycle score. Memory is the maximum program-object
observation, not full-wafer SRAM or physical area. See the
[measurement contract](../docs/benchmarks/metrics.md).

| Profile | Evidence | Cycle metric | Cycles | Program-object bytes |
| --- | --- | --- | ---: | ---: |
| `meshgemv.square64` | six-case independent | `max_local_cycles` | 264.1 | 6,624 |
| `meshgemm.cold-tiny` | six-case independent | `max_local_cycles` | 1545 | 5,936 |
| `meshgemm.cold-prefill-ffn-up-scaled` | six-case independent | `max_local_cycles` | 23230 | 12,096 |
| `pipeline.k8-n8` | six-case independent | `max_local_cycles` | 971 | 6,016 |
| `sdk.gemv-collectives` | upstream reference | — | — | 17,152 |
| `sdk.gemm-collectives` | upstream reference | — | — | 7,488 |
| `sdk.residual` | upstream reference | — | — | 5,072 |
| `sdk.wide-multiply256` | upstream reference | — | — | 5,056 |
| `sdk.game-of-life` | upstream reference | — | — | 9,008 |
| `sdk.fft1d-forward` | upstream reference | — | — | 5,408 |
| `sdk.fft1d-inverse` | upstream reference | — | — | 5,408 |
| `sdk.fft2d-forward` | upstream reference | — | — | 5,824 |
| `sdk.fft2d-inverse` | upstream reference | — | — | 5,824 |
| `sdk.fft3d` | upstream reference | `upstream_cycles` | 117821 | 15,568 |
| `sdk.local-matvec` | upstream reference | `upstream_cycles` | 780 | 7,680 |
| `sdk.spmv7` | upstream reference | `upstream_cycles` | 2127 | 8,656 |
| `sdk.broadcast-row` | upstream reference | `upstream_cycles` | 2259 | 21,216 |
| `sdk.broadcast-column` | upstream reference | `upstream_cycles` | 2525 | 21,216 |
| `sdk.memcpy-h2d` | transfer diagnostic | `upstream_cycles` | 2748 | 21,216 |
| `sdk.memcpy-d2h` | transfer diagnostic | `upstream_cycles` | 3830 | 21,216 |

## Initial variant checks

| Profile / variant | Baseline → candidate cycles | Memory ratio | Decision |
| --- | --- | ---: | --- |
| `pipeline.k8-n8` / fused | 971 → 1,001 | 0.9415× | Rejected: 3.09% more cycles; 5.85% less program-object memory |
| `meshgemv.square64` / editable `gemv-unroll` | 264.1 → 240.7 | 1.0145× | Provisional: 8.86% fewer cycles; fresh V2 confirmation required |

At this initial qualification, no new V2 improvement was marked retained.
Subsequent fresh V2 confirmations are recorded above. Historical V1 confirmation
remains scoped to its original contract. The editable candidate passed the source
preparation, compilation, numerical, timing, memory, and comparison workflow.

## Integration findings

- Repeated GEMM failed its independent output bound; cold one-shot profiles
  passed without relaxing accuracy. See [reusable GEMM state](../docs/research/gemm-state.md).
- Simulator temporary-directory removal exposed a runner traversal race; the
  monitor now tolerates disappearing directories/files and retains resource caps.
- Sparse stencil needed shared SDK libraries and generated import paths inside
  the isolated source tree. Those dependencies are now copied and fingerprinted.

The [experiment index](../EXPERIMENTS.md) links the original research.
For new attempts, use the [protocol](../docs/experiments/protocol.md), preserve raw
artifacts, and export a new uniquely named report. Simulator results alone do
not establish hardware TPS, energy, sustained service capacity, or model speed.
