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

The summary reports saved checkpoints, not whether a simulator process is alive.

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
