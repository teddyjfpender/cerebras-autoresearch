# Batched local research campaigns

Use `tools/run_campaign.py` inside the SDK environment to screen multiple ideas
without recompiling the same full baseline for every candidate. Timing,
numerical checks, resource budgets, comparison, and fresh confirmation still
come from the standard core runner. The helper does not modify those contracts.

```sh
tools/sdk-run env CEREBRAS_RESEARCH_HARNESS=YOUR_HARNESS \
  python3 tools/run_campaign.py PATH/plan.json \
  --state runs/campaigns/YOUR_CAMPAIGN/result.json --max-seconds 7200
```

### Long sessions and persistent logs

For a long campaign, detach it inside the VM so an SSH/tool output disconnect
cannot fail the current run with a broken pipe. Create the log directory first;
the log and checkpoints remain in ignored `runs/`.

```sh
tools/sdk-run bash -c '
  mkdir -p runs/campaigns/YOUR_CAMPAIGN
  nohup env CEREBRAS_RESEARCH_HARNESS=YOUR_HARNESS \
    python3 tools/run_campaign.py PATH/plan.json \
    --state runs/campaigns/YOUR_CAMPAIGN/result.json --max-seconds 7200 \
    >> runs/campaigns/YOUR_CAMPAIGN/console.log 2>&1 < /dev/null &
  echo "Campaign PID: $!"
'
tail -n 20 runs/campaigns/YOUR_CAMPAIGN/console.log
```

The returned PID indicates launch, not successful completion. Inspect the log
and saved status. The standard simulator lease still rejects overlapping runs.
Use `--retry-failed` when resuming a cached failed operation; passed work is
reused. To cancel, send SIGINT to the campaign's own VM PID and wait for its
checkpoint before resuming. Do not kill unrelated SDK or VM processes.

## Plan

```json
{
  "schema_version": 1,
  "name": "local-vector-search",
  "seed": 7,
  "minimum_speedup": 1.05,
  "max_memory_growth": 1.10,
  "cells": [
    {
      "profile": "pipeline.k8-n8",
      "candidates": [
        {"variant": "fused"},
        {"candidate": "YOUR_CANDIDATE"}
      ]
    }
  ]
}
```

Candidate workspaces require concrete approach, hypothesis, and falsification
fields. A cell may provide `baseline_receipt` to reuse an existing full baseline
with the exact profile, initial seed, and evaluator fingerprint. Historical
receipts without evaluator provenance cannot serve as a new campaign baseline.

## Sequence

1. Validate/reuse a full baseline for the profile and initial seed.
2. Screen every listed candidate. Reject absent cycle gain or excess memory
   before spending time on a full candidate run. Numerical failures remain
   failed attempts with their raw receipts.
3. Rank eligible screens by cycles. Fully validate the fastest; if it fails,
   consider the next eligible candidate.
4. Compare the full matched pair. If eligible, run a fresh baseline/candidate
   pair and replay the standard confirmation gate.
5. Retain at most one confirmed candidate per cell. The selection is scoped to
   that fixed profile and the screened candidate set, not a global optimum.

Every run and full comparison/confirmation is journaled through the normal TSV
writer. The campaign state separately records screen selection decisions and
all receipt pointers; a numerically successful screen is not a retained result.
Different candidates can win different profiles. Do not interpret a portfolio
of shape-specific implementations as one universally faster kernel.

## Interrupt and resume

Read progress from the host without claiming or blocking the simulator lease:

```sh
python3 tools/run_campaign.py PATH/plan.json \
  --state runs/campaigns/YOUR_CAMPAIGN/result.json --status
```

The summary keeps the saved checkpoint status separate from `worker_liveness`.
New launchers capture a worker PID, user, Linux boot identity and process start
time. When the reader can inspect that same environment, it reports `alive` or
`not_running`; a reused PID cannot masquerade as the original worker. Legacy
checkpoints, another user/boot environment, or unavailable process metadata report
`unknown`. Reading status never relabels or rewrites the checkpoint.

On this Mac, use `tools/sdk-run python3 tools/run_campaign.py ... --status` to
inspect a newly launched VM worker. A native Mac reader cannot inspect Linux
processes. Older workers need an explicit VM process check because adding this
feature does not retrofit their checkpoints. A live worker observation is a
point-in-time fact, not proof of benchmark success or future completion.

The state is checkpointed after each completed run and decision. Repeat the same
command and state path to reuse completed work. The completed run, rather than
an in-progress simulator step, is the checkpoint boundary. An interrupted step
may need compilation/execution again; its cancelled receipt remains evidence.
Cancelled operations are recorded before the interruption propagates and are
automatically retried on resume. Failed operations require `--retry-failed`.

The plan, evaluator, and candidate source/manifest identities are frozen for a
state file. Changed cached receipt hashes are rejected. Changed source or
contracts require a new plan/state and matched baselines. Keep the raw files:
promotion still verifies complete numerical and resource evidence.

`--retry-failed` permits another attempt at failed operations with unchanged
sources. Previous attempts remain in the state history and TSV. Confirmed cells
are skipped, and other cells reuse cached successful runs and decisions. A
numerical redesign belongs in a new candidate directory and campaign, rather
than editing a frozen candidate.

A campaign can complete with no confirmed gains; rejected hypotheses are useful
results. A failed baseline leaves a cell incomplete. Time exhaustion and
cancellation preserve partial state. Run the ledger artifact verification after
the campaign and record exact coverage in a dated report.

The helper is standard-library Python. Its interruption, cache integrity,
evaluator identity, memory-screen, and shared-baseline behavior have host tests.
Its orchestration does not replace a signed or adversarial hosted judge.

## Export and review

After the campaign stops, verify the journal and raw artifacts, then export a
compact receipt index to a new report path:

```sh
python3 research.py ledger verify --artifacts
python3 research.py export results/reports/YOUR_UNIQUE_RUN_INDEX.json
```

Keep the plan, candidate workspaces, numerical coverage, rejected ideas,
confirmation receipt pointers, and developer-experience findings with a dated
experiment report. The state and console log are recovery aids; the TSV and
replayed confirmation evidence determine retained results. Preserve raw arrays
and evaluator source capsules locally for subsequent artifact replay.


## Cancelling detached campaigns

New campaign launchers explicitly handle SIGINT and SIGTERM as catchable
cancellation, even when a background shell inherited SIGINT as ignored.
Send `kill -TERM PID` to the verified campaign Python process. The runner kills
its isolated SDK process group, preserves the cancelled receipt and checkpoint,
and releases the simulator lease. Resume the same plan/state to retry cancelled
work. Verify the process and checkpoint before starting another writer.

This handler does not retrofit launchers that were already running before the
change. Qualification and candidate attempts remain subject to their finite
step/campaign budgets.

## Export a completed campaign

After the worker stops with a terminal checkpoint, export its evidence:

```sh
python3 tools/campaign_report.py PLAN.json STATE.json --output REPORT.json
```

The exporter refuses running checkpoints, changed frozen plans, modified
receipts, invalid artifacts and unreplayable promotion decisions. It includes
imported baselines, screens, failures, full validations and decisions, with
source/profile/harness/SDK identities and numerical case metadata. It refuses
to overwrite a report. Keep the referenced raw artifacts alongside the compact
report. This integrity audit is separate from SDK NumPy numerical recomputation;
use `tools/replay_numerics.py` for the latter.

### Reuse a matched confirmation baseline

An iteration can set `fresh_baselines_from_campaign` to a completed campaign's
checkpoint path, or give a cell a `fresh_baseline_receipt` object with `path`
and `sha256`. Reuse requires the expected seed (`seed + 1`), baseline variant,
profile, source, compiler arguments, harness, SDK, architecture, simulator thread
count, baseline tree and numerical contract to match. The source campaign's
checkpoint digest is frozen too. Imported baselines are listed separately from
newly executed attempts. Candidate arms still need their own full validation.

Confirmation measures improvement against the pinned reference. Across
iterations, compare confirmed results within the same contract/toolchain track
and keep the fastest confirmed implementation; another confirmation does not
by itself supersede a faster earlier result.

If SDK execution succeeded but receipt/journal delivery raised, the controller
stops before promotion and preserves the successful raw receipt. Run the ledger
recovery and full artifact audit before resuming that interrupted campaign.

### Assemble complete suite evidence from existing runs

`tools/assemble_suite.py` groups already validated full runs into a named suite.
Its manifest uses schema version 1 and a `receipts` list of repository-relative
`path` and `sha256` pointers. Include exactly one receipt per current suite
member. Screens, missing/duplicate cells, stale profile contracts, changed raw
artifacts and mixed seeds/variants/evaluator/toolchains are rejected. Each
profile's exact source, baseline and harness identities remain explicit; paired
suite comparison still verifies each profile's matching harness.

```sh
tools/sdk-run python3 tools/assemble_suite.py gated-mlp MANIFEST.json
# Assemble separate initial baseline/candidate and fresh baseline/candidate arms.
tools/sdk-run python3 research.py confirm-suite \
  BASE_SUITE CANDIDATE_SUITE FRESH_BASE_SUITE FRESH_CANDIDATE_SUITE
```

Run the writer after the active campaign stops. It takes the same exclusive
writer lock, writes an immutable suite receipt and appends its TSV event.
Metadata-only assembly requires 64 MiB free; SDK simulation retains its separate
1 GiB minimum. This permits small evidence records during low-space recovery
while preserving exclusion between metadata and SDK writers. Each bundle
records imported runs and zero newly executed attempts. Successful assembly
establishes complete measured coverage; the ordinary suite confirmation command
applies the geometric-mean, per-cell regression, memory and fresh-source gates.
The aggregate describes the exact recorded implementations across separate
profile runs. Keep model throughput and a combined kernel launch as separate
measurement contracts.

## Durable consoles across agent restarts

A long SDK campaign needs a console file owned by the execution environment.
A server restart during decoder dogfooding closed an attached output pipe;
the next progress print raised `BrokenPipeError` and correctly preserved a
failed partial receipt. Numerical success in its first case did not qualify it.

For unattended runs, launch the campaign inside the SDK VM with stdin redirected
from `/dev/null`, stdout/stderr redirected to a persistent `console.log`, and a
new process session. Save its PID and use the campaign's boot/start-time identity
when checking liveness. This keeps execution independent of the host tool's
output handle. Inspect the actual worker before retrying after any disconnect.

Retain the interrupted checkpoint and receipts. A focused continuation may reuse
hash-bound completed arms with unchanged profile, source, harness and evaluator;
the unsuccessful fresh arm must run again in full. Record that continuation
separately, then run artifact verification and independent numerical replay.
