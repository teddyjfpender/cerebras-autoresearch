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
