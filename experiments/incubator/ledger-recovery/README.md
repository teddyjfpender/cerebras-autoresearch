# Staged ledger recovery improvement

Current status: applied with reusable-GEMM evaluator
`315443cd2ad3f1f2fadba98d225e8cae57e9bb27e5d7272d22f72f1318c338d0`.
The live tree passed 97 host checks (one Mac skip) and 27 SDK contract checks.
Historical development notes below preserve the pre-integration findings.

Observed problem: SDK VM recovery previously spent about six minutes replaying
already-journaled receipts. `append_receipts` currently calls `receipt_row` and
its full numerical/artifact audit before noticing the receipt was recorded.

The proposed fast path checks the contained receipt path and its saved SHA first.
An unchanged prior receipt is skipped; a changed receipt still fails. New events
retain the complete existing audit and decision replay. SHA chaining, history,
board derivation and promotion gates remain unchanged.

This deliberately separates incremental recovery from the explicit full audit:
`ledger verify --artifacts` must still detect corrupt raw artifacts behind an
unchanged receipt. It remains a required acceptance gate after recovery.

Before applying, add tests proving no old audit on unchanged recovery, changed
receipt rejection, full-audit raw corruption detection and new-event replay.
Measure recovery cost without weakening complete verification. Do not infer
simulator/kernel speed from journal latency.


## Additional harness integrity finding

`runner.py` currently fingerprints the trusted driver copies, but not auxiliary
Python copied from a baseline (for example `host_common.py`). Candidate overlays
cannot replace Python, and pinned-source preflight checks the original source;
however, post-run helper mutation should also fail evidence replay.

`complete-harness.patch` proposes hashing every prepared Python source. It must
be tested by modifying an auxiliary helper after a recorded run and requiring
replay rejection. Apply only with the next evaluator track and new paired
baselines, preserving the current campaign's source capsules.

Five regression gates now pass against an isolated copy with both patches
applied (`check_recovery.py`). The unchanged live evaluator fails the intended
new fast-path/helper-coverage checks; changed-receipt rejection already passes.
The staged next evaluator also passes the complete 69-test host suite. These
checks are preparation evidence; both live patches remain unapplied while
current matched experiments run.

## Failed-step cost capture staged with the next adapter

The isolated next evaluator also preserves the attempted command, log path,
elapsed wall time and failure/cancellation status when an SDK step raises.
Total experiment wall time includes that failed step. A host regression gate
covers both RuntimeError and KeyboardInterrupt; it launches no SDK job and
preserves zero numerical cases. This addresses the earlier compile failures,
whose receipts retained logs but omitted the unsuccessful step's wall cost.
The live evaluator is still frozen. Historical receipts are not rewritten.

The staged board also derives `best_confirmed_by_track`, selecting by the conservative paired speedup, then cycles and memory within
complete source/profile/harness/SDK/evaluator identities.
A regression test preserves a faster incumbent when a later confirmed approach
is slower, keeps evaluator tracks separate, and excludes a faster unconfirmed
measurement from that selection. All confirmation history stays visible.

Prepared copies now also exclude Python bytecode caches in the isolated next
evaluator. A subprocess regression fixture creates stale bytecode with the
same source size and timestamp: the current copier executes the old helper,
while the proposed copier executes the fingerprinted new source. The expanded
isolated host suite passed 82 tests. This change remains staged until the active
matched campaigns finish.


Candidate preparation in the staged adapter also carries the pinned upstream
LICENSE/NOTICE files. A candidate-init regression checks the copied license and
source together. The staged host suite now contains 83 passing checks. Its source
and destination file digests are recorded in a guarded integration manifest;
application is prohibited while either frozen campaign/handoff is active.

## Compiled binary identity added to the staged evaluator

Inspection found 33 compiled ELF objects (about 1 MiB) in the completed GEMM-up
run, with zero ELF hashes in its original artifact manifest. The staged runner
now hashes `.elf` outputs alongside the existing source/array/log artifacts.
The failure/cancellation regression creates an opaque synthetic partial ELF,
checks that the actual runner records it, then requires evidence replay to
reject a changed binary. It does not simulate a numerical kernel.

Historical manifests keep their original coverage. The latest reviewed staged
evaluator is `315443cd2ad3f1f2fadba98d225e8cae57e9bb27e5d7272d22f72f1318c338d0`;
the guarded apply manifest has been regenerated and still contains 26 paths.
The live evaluator remains unchanged. The latest staged host count before the
metadata-only lease change was 94 tests with one Mac Linux-only skip.


The complete integration is now preserved in the reusable-GEMM incubator's
`next-evaluator.patch` and per-file hash manifest. Latest staged checks: **97
host tests**, with one Linux-only Mac skip. The live F1 tree passes 86 tests.
The SDK device qualification remains blocked by host disk space; these test
counts do not imply kernel execution.
