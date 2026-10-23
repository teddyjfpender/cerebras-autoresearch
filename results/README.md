# Results and qualification

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

## Latest resident GEMM result

The complete three-shape dispatcher suite is freshly confirmed at **1.146754x
geometric mean speedup** (worst1.025274x), with all72 paired numerical/core
cases independently replayed. FFN-up is95,611 ->69,104 cycles and FFN-down
128,315 ->120,701. Tiny is6,653 ->6,489 within the suite guard.
[Complete evidence](reports/2026-10-23/resident-gemm/dispatch-suite-confirmed.json).

## Distributed resident SwiGLU — confirmed complete chain

`swiglu-projection-silu-reuse` passed full validation and fresh paired confirmation
on both profiles, including every G/U/V/D/Y intermediate and canonical/working
state. All **48 cases** passed independent numerical and saved-core/timer replay.
The conservative suite geometric mean is **1.415571x**, worst cell **1.376291x**.

| Profile | Initial reference → candidate cycles | Fresh reference → candidate cycles | Program bytes/PE |
| --- | ---: | ---: | ---: |
| `resident-swiglu.small` | 58,502 → 42,506 | 58,502 → 42,506 | 10,496 → 10,768 |
| `resident-swiglu.ffn-scaled` | 586,834 → 400,069 | 586,834 → 403,053 | 43,712 → 45,184 |

[Complete evidence](reports/2026-10-23/resident-swiglu/initial-suite-confirmed.json)
binds the two seeds and complete suite. The scaled cell is data dependent, so
both measured seed arms are shown. The candidate combines projection layout,
bounded FP32 SiLU evaluation and invariant-backed receive-buffer reuse. This
track retains its original FP16 accumulation/storage and FP32 residual contract;
BF16/FP32-accumulation work requires separate profiles and new baselines.

## Native BF16 resident SwiGLU — confirmed complete chain

`bf16-swiglu-vector-scratch-tail4-runtime` is freshly confirmed on both original
profiles, with **1.537186x conservative geometric mean**, worst cell **1.499687x**.
All **48 paired cases** passed independent numerical and complete saved-core,
canonical/working-state and raw-timer replay. One frozen parameterized source
uses vector activation where scratch capacity permits and scalar activation on
smaller tiles; every BF16 stage boundary and FP32 accumulation remains fixed.

| Profile | Initial reference → candidate cycles | Fresh reference → candidate cycles | Program bytes/PE |
| --- | ---: | ---: | ---: |
| `resident-swiglu-bf16.small` | 57,573 → 38,390 | 57,573 → 38,390 | 11,088 → 11,456 |
| `resident-swiglu-bf16.ffn-scaled` | 585,178 → 368,626 | 585,178 → 371,395 | 43,600 → 47,184 |

[Complete refinement evidence](reports/2026-10-23/resident-swiglu-bf16/runtime-suite-confirmed.json)
binds both seeds and the entire original suite. The scaled cell uses +8.22%
program memory within the unchanged 10% gate. Scratch reuse fixes the earlier
vector candidates' memory overrun; the rejected comptime pointer cast and
compiler interruption remain journaled.

The [initial scalar composition](reports/2026-10-23/resident-swiglu-bf16/initial-suite-confirmed.json)
remains confirmed at 1.451350x conservative mean with less memory. Its
[paired stage audit](reports/2026-10-23/resident-swiglu-bf16/initial-paired-stage-differences.json)
found every G/U word identical in all 24 matched case pairs (233,472 values per
stage). The bounded approximation changes some V/D/Y words; the original local
and ideal full-chain bounds passed separately. The
[baseline qualification](reports/2026-10-23/resident-swiglu-bf16/qualification.json)
and all historical reports are unchanged. These are scaled FFN operator results.

Two additional workloads now have full six-case baselines and independent
numerical/core replay: M4/H80/F272 at 383,514 cycles (36,544 program bytes/PE),
and M32/H64/F192 at 651,264 cycles (37,648 bytes/PE), each T4 on 16 PEs. They form the distinct four-cell
`resident-swiglu-bf16-generalization` suite with the original profiles. Candidate
source identities were frozen before any held-out device observations. The
frozen source is freshly confirmed on M4 at1.8139x conservative speedup. Its M32
screen reached1.5755x but exceeded the fixed memory budget (+12.49%); it is
rejected there. See the [held-out campaign](reports/2026-10-23/resident-swiglu-bf16/heldout-frozen-campaign.json).
The follow-up two-buffer source is now separately qualified on all four shapes, as described below.

## BF16 four-shape generalization — confirmed follow-up

`bf16-swiglu-two-buffer-tail4` passes the complete four-profile suite with
**1.589946x conservative geometric mean**, worst cell **1.499687x**. All **96
paired cases** pass independent numerical, complete core/state and raw timer
replay. Two live FP32 tiles replace the previous three-tile expansion/activation
layout; timed transpose/copy-back and every projection/stage/frame remain.

| Profile | Initial reference → candidate cycles | Fresh candidate cycles | Program bytes/PE |
| --- | ---: | ---: | ---: |
| `resident-swiglu-bf16.small` | 57,573 → 38,390 | 38,390 | 11,088 → 11,328 |
| `resident-swiglu-bf16.ffn-scaled` | 585,178 → 376,762 | 379,531 | 43,600 → 46,064 |
| `resident-swiglu-bf16.batch4-ffn` | 383,514 → 212,861 | 212,713 | 36,544 → 38,432 |
| `resident-swiglu-bf16.batch32-h64-f192` | 651,264 → 424,872 | 424,870 | 37,648 → 40,784 |

Memory growth is at most8.33%, including the batch32 boundary that rejected the
previous source. The specialized earlier candidate is slightly faster on some
cells; this shared source qualifies the entire fixed four-cell suite. It is a
follow-up after held-out observations, not an unseen-shape claim.
[Complete confirmation](reports/2026-10-23/resident-swiglu-bf16/two-buffer-suite-confirmed.json),
[all campaign attempts](reports/2026-10-23/resident-swiglu-bf16/two-buffer-campaign.json),
[96-case numerical replay](reports/2026-10-23/resident-swiglu-bf16/two-buffer-suite-numerics.json),
[96-case core/state/timer replay](reports/2026-10-23/resident-swiglu-bf16/two-buffer-suite-snapshots.json).
These are simulator operator-chain results; Qwen TPS is unmeasured.

## Native-width recurrent operator — confirmed normalization reuse

Native K128/V128, three value heads, both T4/T8 profiles and two paired seeds
confirm **1.251334× geometric mean**, worst **1.247939×**, against an already
vectorized state baseline. All **48 paired cases** pass full numerical and
core/state/timer replay. Every token and full FP32 state remains checked.

| Tokens | Baseline → candidate cycles | Speedup | Program memory growth |
| --- | ---: | ---: | ---: |
| T4 | 115,634 → 92,660 | 1.2479× | 0.446% |
| T8 | 211,278 → 168,384 | 1.2547× | 0.462% |

[Complete evidence](reports/2026-10-23/recurrent-gdn-native/vector-suite-confirmed.json).
This is one key-head group with native state width. The full 16-key/48-value-head
operator is the next fixed-contract challenge; no model TPS is established.

## Resident recurrent gated-delta operator — confirmed vectorization

One frozen `recurrent-gdn-vector-state` source passes both complete profiles and
fresh paired seed: **7.350146x conservative geometric mean**, worst **7.161137x**.
All **48 paired cases** pass independent numerical and complete core/state/timer
replay, including every token's FP32 full state, normalized Q/K, decay, delta,
BF16 output and all immutable canonical inputs. Memory growth is at most1.35%.

| Profile | Initial reference → vector cycles | Fresh vector cycles | Program bytes/PE |
| --- | ---: | ---: | ---: |
| `recurrent-gdn.k16-v16` | 316,866 → 44,248 | 44,239 | 28,480 → 28,864 |
| `recurrent-gdn.k32-v16` | 621,215 → 82,344 | 82,347 | 47,808 → 48,192 |

The baseline is the repository's independent scalar CSL correctness reference,
with fixed FP32 state and BF16 Q/K/V/output. It is not Cerebras's production
runtime. This operator is after projections/convolution and precomputed g/beta,
before gated output normalization/projection. No layer/model TPS is established.
The initial native `math.exp` accuracy failure remains journaled; the reference
was repaired before qualification without changing the fixed numerical bounds.

[12-case baseline qualification](reports/2026-10-23/recurrent-gdn/qualification.json),
[complete fresh suite](reports/2026-10-23/recurrent-gdn/vector-suite-confirmed.json),
[48-case numerical replay](reports/2026-10-23/recurrent-gdn/vector-suite-numerics.json),
[48-case core/state/timer replay](reports/2026-10-23/recurrent-gdn/vector-suite-snapshots.json).
The native K128/V128 follow-up will use the vector implementation as its starting
reference, with its own fixed PE budgets, full baselines and new qualification.

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

Across historical evaluator tracks, the current 48-profile catalog has
**26 full independent baselines, 14 reference checks and two diagnostics**:
42 exercised profiles. The new full-native-head recurrent profile is in SDK
qualification. Five older independent shapes remain unqualified:
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

## Resident reusable GEMM — qualified baselines

All three profiles passed the six independent families under the selected
`lima-vz-rosetta-v1` execution profile: **18 successful SDK case runs**. Each
sequence computes four distinct resident inputs against shared weights. These
are baselines; no optimized resident GEMM is retained by this qualification.

| Profile | Four-frame cycles | Average cycles/GEMM | Program bytes/PE |
| --- | ---: | ---: | ---: |
| `reusable-gemm.tiny-sequence` | 6,653 | 1,663.25 | 7,584 |
| `reusable-gemm.prefill-ffn-up-scaled` | 95,611 | 23,902.75 | 29,792 |
| `reusable-gemm.prefill-ffn-down-scaled` | 128,315 | 32,078.75 | 30,160 |

The [qualification report](reports/2026-10-23-reusable-gemm-qualification.json)
binds the source/profile/evaluator/SDK identities, all 48 compiled compute ELFs,
commands, errors, memory and case receipts. The
[independent numerical replay](reports/2026-10-23-reusable-gemm-qualified-numerics.json)
recomputed all 18 cases. The
[snapshot replay](reports/2026-10-23-reusable-gemm-qualified-snapshots.json)
also matched every output, canonical input, working-weight history and raw timer
word against its saved SDK core. All 18 cores are artifact-hashed and preserved.
Maximum error-bound fractions were 0.164, 0.122 and 0.0485 respectively; every
canonical/working-state bit check passed. The
[suite receipt](reports/2026-10-23-reusable-gemm-qualified-suite.json) imports these
three measurements and executes no additional SDK attempts.

The [host-route diagnostic](reports/2026-10-23-sdk-rosetta-platform-screen.json)
and [SDK setup](../docs/sdk/setup.md) explain the faster execution route and its
distinct evaluator track. Earlier QEMU measurements and failed/interrupted
readback attempts remain immutable history. Compare optimized candidates with
matched baselines on this track; these scores do not establish hardware or model TPS.

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

## Resident GEMM first optimization pass (2026-10-23)

On the separately qualified Rosetta SDK platform, contiguous row accumulation
and a timed vector pack reduce FFN-up's complete four-frame sequence from
95,611 to 75,962 cycles (1.25867x; 20.55% fewer cycles). Program-object memory
increases from 29,792 to 30,848 bytes per PE (+3.54%). All six cases pass at
seeds 20261027 and 20261028. The fixed canonical-input and complete per-frame
weight-state checks pass. Independent replay corroborates all 24 paired cases,
including every saved output/state/timestamp against the simulator cores.

Tiny and FFN-down retain their reference implementations: none of the four
first approaches passed the 1.05x cycle gate. This is a profile-specific gain.

- [Campaign and rejected screens](reports/2026-10-23-resident-gemm-initial-campaign.json)
- [24-case numerical replay](reports/2026-10-23-resident-gemm-initial-up-numerics.json)
- [24-case core/state/timer replay](reports/2026-10-23-resident-gemm-initial-up-snapshots.json)

## Resident GEMM descriptor refinement (2026-10-23)

FFN-up's separately confirmed fixed-contract sequence improves further to
69,104 cycles from the matched 95,611 reference (1.38358x; 27.724% fewer cycles).
Program-object memory is 30,640 vs 29,792 bytes/PE (+2.846%). Hoisting the weight
descriptor and eliminating invariant-proven redundant resets preserves every
FP16 K term, complete output packing and boundary-state snapshot. Both seeds
20261029/20261030 pass all six cases. The 24 paired cases independently replay
numerically and match complete simulator core outputs/state/timestamp words.
Tiny and FFN-down retain the original layout; grouped-map simulator failures
remain failures in the journal. Further candidate work is unmeasured until run.

- [Second campaign including all failures](reports/2026-10-23-resident-gemm-iteration2-campaign.json)
- [24-case numerical replay](reports/2026-10-23-resident-gemm-iteration2-up-numerics.json)
- [24-case simulator-core replay](reports/2026-10-23-resident-gemm-iteration2-up-snapshots.json)

The third resident GEMM campaign confirmed the same 69,104-cycle FFN-up result
using the parameterized aspect-ratio dispatcher; tiny/down screens keep the
reference column-map layout. This confirms FFN-up only: a cross-shape suite
claim still needs full tiny/down pairs. Additional mapped and fixed-descriptor
experiments failed compilation or simulation and were not retained.

- [Third campaign and all failures](reports/2026-10-23-resident-gemm-iteration3-campaign.json)
- [Dispatcher 24-case numerical replay](reports/2026-10-23-resident-gemm-iteration3-up-numerics.json)
- [Dispatcher 24-case complete core replay](reports/2026-10-23-resident-gemm-iteration3-up-snapshots.json)

## Resident GEMM contraction refinement (2026-10-23)

FFN-down's reference column map benefits from array-form output-descriptor
base resets: 128,315 -> 120,701 cycles (1.06308x; 5.934% fewer cycles), with
30,112 vs 30,160 program bytes/PE. All six cases pass both seeds20261029/20261030;
all 24 paired cases pass independent FP64 numerical replay and complete core
output/state/timestamp corroboration. Tiny's same-approach screen is only
1.02527x and is not individually retained. The combined dispatcher is awaiting
full complete-suite qualification.

- [Array-base campaign](reports/2026-10-23-resident-gemm-iteration5-campaign.json)
- [FFN-down 24-case numerical replay](reports/2026-10-23-resident-gemm-iteration5-down-numerics.json)
- [FFN-down 24-case full-core replay](reports/2026-10-23-resident-gemm-iteration5-down-snapshots.json)

## Resident GEMM complete-suite confirmation (2026-10-23)

One parameterized dispatcher passes the complete three-profile suite at both
seeds20261029/20261030: geometric mean1.146754x, worst cell1.025274x. All72
paired cases pass independent FP64 replay and complete simulator-core
output/state/timer corroboration. Individual FFN-up/down confirmations are also
recorded; tiny qualifies within the suite guard and remains below the individual
1.05x retention threshold. Seven new SDK runs reuse five matched baselines.

| Profile | Reference cycles | Dispatcher cycles | Program bytes/PE reference -> dispatcher |
| --- | ---: | ---: | ---: |
| Tiny four-frame sequence | 6,653 | 6,489 | 7,584 -> 7,520 |
| Scaled resident FFN-up | 95,611 | 69,104 | 29,792 -> 30,640 |
| Scaled resident FFN-down | 128,315 | 120,701 | 30,160 -> 30,112 |

[Confirmed suite and profile decisions](reports/2026-10-23/resident-gemm/dispatch-suite-confirmed.json),
[72-case numerical replay](reports/2026-10-23/resident-gemm/dispatch-suite-numerics.json),
[72-case core replay](reports/2026-10-23/resident-gemm/dispatch-suite-snapshots.json).

## Complete distributed resident SwiGLU — qualified baselines

Both profiles pass all six cases on16 compute PEs, with every G/U/V/D/Y
intermediate/output, exact canonical inputs/weights, complete per-frame final
working-weight state and timer corroborated in the12-case replay.

| Profile | Whole4-frame cycles | Program bytes/PE | Max error fraction |
| --- | ---: | ---: | ---: |
| `resident-swiglu.small` | 58,502 | 10,496 | 0.410 |
| `resident-swiglu.ffn-scaled` | 586,834 | 43,712 | 0.483 |

[Qualified receipts and full identities](reports/2026-10-23/resident-swiglu/qualification.json),
[12-case FP64 replay](reports/2026-10-23/resident-swiglu/qualified-full-numerics.json),
[12-case complete core/state/timer replay](reports/2026-10-23/resident-swiglu/qualified-full-snapshots.json).
These are new chain baselines, not retained optimization results. The three
initial compile failures remain in the journal with separate bringup reports.
