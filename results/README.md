# Results and qualification

## Canonical research journal

[`results.tsv`](results.tsv) automatically records terminal V2 attempts and
decisions. [`leaderboard.json`](leaderboard.json) derives measured bests by
declared approach, a cycle/memory tradeoff front, and confirmed events.
Measured views remain provisional; the confirmed view is currently empty.

The [ledger setup report](reports/ledger-qualification.json) records 43 events:
40 historical receipts, two replayed comparison decisions, and one live SDK
screen proving automatic logging with an evaluator source capsule. All 48 host
tests and artifact replay passed. This is an unsigned local ledger; see the
[ledger contract](../docs/experiments/ledger.md) for guarantees and limits.

## Repository qualification — 2026-10-22

**20 of 30 configured profiles exercised:** four full six-case independent
baselines, fourteen upstream reference checks, and two transfer diagnostics.
Two additional variants passed their full six-case numerical checks.
The other ten shapes are configured and unexecuted in this qualification.

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

## Variant checks

| Profile / variant | Baseline → candidate cycles | Memory ratio | Decision |
| --- | --- | ---: | --- |
| `pipeline.k8-n8` / fused | 971 → 1,001 | 0.9415× | Rejected: 3.09% more cycles; 5.85% less program-object memory |
| `meshgemv.square64` / editable `gemv-unroll` | 264.1 → 240.7 | 1.0145× | Provisional: 8.86% fewer cycles; fresh V2 confirmation required |

No new V2 improvement is marked retained. Historical V1 confirmation remains
scoped to its original contract. The new editable candidate passed the source
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
