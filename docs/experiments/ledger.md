# Research journal and practices

## Files and meaning

| File | Purpose |
| --- | --- |
| `results/results.tsv` | Canonical local event journal: terminal V2 runs and recorded decisions |
| `results/leaderboard.json` | Views derived only from the TSV, with row count and chain head |
| `runs/v2/*/result.json` | Detailed immutable receipts behind each event |
| `runs/v2/*/work/` | Isolated CSL, trusted numerical drivers, arrays, timers, logs, resource evidence |
| `runs/v2/*/evaluator-source/` | Exact trusted host/driver source capsule for new attempts |

The TSV includes failed attempts, screens, upstream references, and diagnostics.
These are useful research history and cannot enter the measured independent
board. Full independent measurements appear as **measured**, not retained.
Only replayed fresh confirmation events enter the confirmed view. Suite
confirmation events are listed separately and remain scoped to their full suite.

Existing V2 receipts are imported with `origin=historical_import`. Unknown
historical Git/evaluator/author/model information remains unknown. V1 stays in
its separate archive and is never mixed into the V2 cycle league.

## One experiment

1. Choose the bottleneck and an approach worth testing. An approach is a declared
   design family, not proof of novelty. Preserve experiments from new approaches
   even when they are initially slower.
2. Create a candidate workspace. Write a concrete `approach`, `hypothesis`, and
   `falsification` in candidate.json; templates must be completed before running.
3. Fix workload, dtype, PE grid, numerical bounds, timing, and resource budgets.
   Change one stated idea per attempt. Give changed contracts distinct profiles.
4. Run a finite screen, then a complete matched challenge. Record failures and
   rejected hypotheses as well as gains. Preserve the exact CSL and evaluator.
5. Require fresh paired seeds and regression coverage before widening a claim.
   Inspect equivalent timed work; numerical checks alone do not prove it.
6. Verify artifacts and commit the appended journal and derived board alongside
   the candidate and notes. Never rewrite a previous row to improve its outcome.

Optional attribution is declared through candidate.json fields `author`, `model`,
and `agent_harness`, or these environment variables (environment takes priority):

```sh
tools/sdk-run env CEREBRAS_RESEARCH_AUTHOR=YOUR_NAME \
  CEREBRAS_RESEARCH_MODEL=YOUR_MODEL CEREBRAS_RESEARCH_HARNESS=YOUR_HARNESS \
  python3 research.py challenge meshgemv.square64 \
  --variant candidate --candidate YOUR_CANDIDATE --max-seconds 1800
```

Do not invent missing model versions or identities. Attribution is declared,
not authenticated. A new evaluator fingerprint requires matched new baselines.
New receipts capture the actual trusted source files even when Git is dirty;
the recorded commit alone is not treated as sufficient provenance.

## TSV schema

The header is fixed and checked exactly. UTF-8 text uses tabs and LF, without
CSV quoting. Control characters and non-ASCII text within cells use JSON-style
escapes so a hypothesis cannot create another cell or row. Empty numeric cells
mean unavailable, never zero. Non-finite numbers are rejected.

| Columns | Meaning |
| --- | --- |
| `sequence`, `event_id`, `recorded_utc` | Stable event identity and journal ordering; import time is not measurement time |
| `kind`, `origin` | Run/suite/challenge/comparison/confirmation; live vs historical import |
| `outcome`, `reported_outcome`, `evidence_audit`, `decision` | Observed status, receipt claim, audit result, and provisional/confirmed standing |
| `profile`, `family`, `approach`, `candidate`, `variant` | Fixed workload and declared implementation family |
| `seed`, `case_count` | Reproduction and numerical coverage |
| `primary_metric`, `primary_value`, `memory_bytes`, `max_error_fraction` | Cycle/resource/correctness observations; scope follows the profile |
| `speedup`, `memory_ratio` | Matched comparison; confirmed rows use the conservative ratio across both pairs |
| `git_commit`, `git_dirty`, `evaluator_sha256` | Controller version and trusted-source identity |
| `profile_sha256`, `baseline_tree_sha256`, `source_sha256`, `harness_sha256`, `sdk_sha256` | Contract, compiled CSL, driver, and SDK identities |
| `author`, `model`, `agent_harness`, `hypothesis`, `falsification`, `note` | Research context and failure/rejection explanation |
| `receipt`, `receipt_sha256` | Exact detailed receipt behind the row |
| `previous_sha256`, `row_sha256` | SHA-256 chain over all row cells |

A comparison/confirmation receipt is saved before its TSV event is appended.
Successful run evidence is audited before insertion. Invalid historical evidence
is recorded as `evidence_invalid`, preserving the original claimed outcome;
it cannot enter the board. Confirmed labels are recomputed using the existing
scoring gates, rather than accepted from a JSON status string.

## Derived views

The board retains the best full independent measurement per declared approach,
and a measured cycle/memory Pareto front. These views are explicitly provisional.
All comparisons are separated by profile, baseline tree, numerical harness,
SDK image, and evaluator fingerprint. Different shapes or evaluator versions
are never combined into one fastest-kernel claim. Exact ties remain reproducible
events; only strict improvement replaces an approach's measured best.

The confirmed view contains fresh confirmations, not every fast measurement.
A memory-saving but slower candidate can remain on the measured Pareto front
while failing the cycle-retention gate. Reference-only and transfer diagnostics
remain in the journal but do not compete in these views.

## Verification and recovery

```sh
python3 research.py ledger verify
python3 research.py ledger verify --artifacts
python3 research.py ledger verify --base-ref COMMIT
python3 research.py ledger sync
python3 research.py ledger board
```

- Default verification checks TSV structure, event identities, the hash chain,
  and exact agreement with the derived board. It needs no SDK or raw arrays.
- `--artifacts` also checks receipt hashes, available artifact/source/capsule
  identities, and decision replay. Keep raw evidence to make this possible.
- `--base-ref` requires the exact accepted TSV byte prefix from a Git commit.
  CI applies this to pull-request bases and prior push commits, detecting edits,
  reordering, and truncation of accepted history even if hashes are recomputed.
- `sync` idempotently recovers terminal receipts after a logging interruption.
  Changed already-journaled receipts are rejected. Incomplete/unsupported
  receipts are listed under `skipped_receipts`; a running receipt is not silently
  relabeled as completed after a hard kill.
- `board` regenerates the derived JSON without rewriting the TSV.

The writer serializes in one execution environment and atomically replaces the
TSV with its unchanged prefix plus appended rows. Use one writer environment
(the SDK VM here); do not mix concurrent Mac and VM writers. A logging failure
makes the command fail while preserving its detailed receipt. Recover and
verify before continuing. If a write stopped between TSV and board updates,
regenerate the board and verify again. Raw cleanup remains manual after archive.

## Comparison with the quantum challenge

The [quantum ledger contract](https://github.com/teddyjfpender/quantum-autoresearch-challenge/blob/main/spec/LEDGER.md)
uses judge-authenticated rows, a secret-derived evaluation seed, architecture
standings, and ledger-derived views. This repository adopts explicit approaches,
content identities, immutable history, replay, tradeoff views, and separate
measurement/confirmation states for local CSL research.

**This local ledger is unsigned.** A SHA-256 chain is not authentication: a person
who controls the repository can construct a new chain. Git history and CI prefix
checks anchor accepted history; they do not establish an independent numerical
judge. Current seeds are public, and the SDK runner is local. A signed hosted
SDK judge, secret held-out seeds, adversarial submission isolation, and public
publication/attestation infrastructure are not provided. SDK-free CI validates
repository/ledger integrity; actual simulated results still require SDK runs.

## Numerical replay of raw evidence

`ledger verify --artifacts` checks the chain, receipt/source/artifact hashes and
replays saved comparison/confirmation decisions. For independent profiles,
`tools/replay_numerics.py` additionally recomputes FP64 outputs and fixed bounds
from saved inputs, checks every resident output/cache state, and decodes the raw
integer timestamps again. It runs in SDK NumPy and launches no device program.

Provide a JSON manifest with schema version 1 and `receipts`, each containing a
repository-relative `path` and the receipt's `sha256`. Only full independent
validated runs qualify. Duplicate receipts, changed files, incomplete shapes,
wrong input precision, mismatched expected values/bounds, state corruption and
inconsistent timestamps are rejected. Existing reports cannot be overwritten.

```sh
tools/sdk-run cs_python tools/replay_numerics.py \
  --manifest runs/audits/selected-receipts.json \
  --output results/reports/my-numerical-replay.json
```

The report includes the auditor/helper hashes, each run's profile/source/harness/
SDK/evaluator identities and recomputed case metrics. This audit corroborates
numerical evidence; it does not promote a variant or replace fresh paired
confirmation. Raw inputs, outputs and state remain in the original ignored runs.

The separate numerical replay also checks saved error statistics against their
actual/expected/bound arrays, then checks run summaries against verified case
metrics and recorded step durations. This includes PE timing/spread, error
fraction, declared work per cycle, PE-cycle cost and resident sequence averages.
Historical replay reports keep their original auditor hashes; later audits add
new reports rather than rewriting earlier evidence.

## Full simulator SRAM corroboration

For supported resident GEMM and distributed SwiGLU contracts,
`tools/replay_snapshots.py` reads every saved stage/output, immutable input,
boundary weight history and timer word from the complete clean-stop SDK core.
It checks the raw artifacts and ELF/source identities, then compares bit-exact
reconstructed arrays with the saved numerical evidence. This launches no device
and grants no retention decision. It complements the independent FP64 replay.

```sh
tools/sdk-run cs_python tools/replay_snapshots.py \
  --manifest runs/audits/selected-receipts.json \
  --output results/reports/my-core-replay.json
```

The manifest schema matches numerical replay. Reports cannot be overwritten.
Unsupported contracts, duplicate receipts, changed/missing cores, mismatched
snapshot helpers and any array/timestamp mismatch fail the audit. Distributed
SwiGLU is registered; its current device qualification is recorded in results.
