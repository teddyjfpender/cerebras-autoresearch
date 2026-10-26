# Speed and memory tradeoffs

The research objective has two independent costs: simulated cycles and measured
program-object bytes per PE. A correct implementation that uses substantially
more memory can be valuable. The default 1.05x speed / 1.10x memory promotion
policy is a conservative convenience; it does not define the Pareto frontier.
The user explicitly authorized exploring and submitting these memory tradeoffs
on 2026-10-26. Existing contracts, policies and historical decisions stay intact.

Fully validate promising points outside that policy instead of stopping at a
screen. Run all six numerical families at the initial and fresh paired seeds,
with matched full baselines, source/evaluator identities and fixed workload,
precision, PE budget and timing. All runs still enter the ordinary TSV journal.
Record the default comparison even when it rejects the relative memory cost.
Hardware/compilation limits still apply; failed allocations are failed attempts.

## Reusable full frontier loop

`tools/run_frontier.py` runs every selected candidate through full initial/fresh
validation. It does not screen out more memory or smaller cycle gains, and does
not pick only one winner. A minimal plan is:

```json
{
  "schema_version": 1,
  "kind": "full_fresh_frontier_pilot",
  "seed": 20261061,
  "candidates": ["YOUR-CANDIDATE"],
  "profiles": [{"profile": "meshgemm.cold-prefill-ffn-up-batch64"}]
}
```

```sh
tools/sdk-run python3 tools/run_frontier.py experiments/YOUR-PLAN.json \
  --state runs/campaigns/YOUR-FRONTIER.json --max-seconds 7200
tools/sdk-run python3 tools/paired_frontier.py \
  --manifest runs/campaigns/YOUR-FRONTIER.pairs.json \
  --output results/reports/UNIQUE-frontier.json
tools/sdk-run cs_python tools/replay_numerics.py \
  --manifest runs/campaigns/YOUR-FRONTIER.full.json \
  --output results/reports/UNIQUE-numerics.json
```

Each profile may optionally import digest-bound `initial_baseline` and
`fresh_baseline` pointers with the same format shown below. Omit these fields
when reproducing without the original local artifacts; new matched controls
will be executed. A checkpoint freezes plan/evaluator/candidate identities and
checks cached receipts. Resume with the same command; cancelled work retries,
failed work requires `--retry-failed`, and source changes require a new state.
Use `--status` to read saved progress and worker liveness without running work.
After retrying failures, use a new `--export-prefix` if the immutable exported
manifests already exist. All ordinary runs and conservative policy decisions
are journaled automatically. `completed_with_failures` explicitly preserves
unqualified attempts; only complete fresh pairs enter the frontier manifest.

## Fresh-paired reports

`results/leaderboard.json` already contains a measured cycle/memory frontier
without the promotion cutoff. It includes provisional full measurements.
For a stronger, explicitly fresh-paired view, prepare a manifest:

```json
{
  "schema_version": 1,
  "points": [{
    "label": "candidate-name",
    "initial_baseline": {"path": "runs/v2/BASE/result.json", "sha256": "..."},
    "initial_candidate": {"path": "runs/v2/CAND/result.json", "sha256": "..."},
    "fresh_baseline": {"path": "runs/v2/FRESH-BASE/result.json", "sha256": "..."},
    "fresh_candidate": {"path": "runs/v2/FRESH-CAND/result.json", "sha256": "..."}
  }]
}
```

```sh
tools/sdk-run python3 tools/paired_frontier.py \
  --manifest runs/frontier-pairs.json \
  --output results/reports/UNIQUE-frontier.json
```

The reporter checks full evidence and matching fresh sources even when the
default retention policy rejects a point. It reports maximum cycles/bytes across
the two seeds, conservative paired speedup, memory growth, per-seed values and
receipt digests. It includes each baseline and marks dominated points rather
than hiding them. Exact ties survive. Different contracts, toolchains, baseline
sources or seed pairs never dominate one another. There is no weighted scalar
score or arbitrary memory-growth cutoff for this view.

This derived report does not grant promotion or append invented journal events.
Its full runs and ordinary comparison decisions remain in the canonical TSV.
Independent raw-array/timestamp replay is recorded separately. Publish the
frontier with numerical errors, fixed PE/workload details and this distinction;
program-object memory is not a measurement of hardware energy or model TPS.
