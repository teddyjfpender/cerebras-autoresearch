# Repository architecture

The standard-library host package is under `src/`. `research.py` works without
installation. An editable install exposes `cerebras-research`; set
`CEREBRAS_RESEARCH_ROOT` when pointing an installed CLI at another checkout.

| Module | Responsibility |
| --- | --- |
| `catalog.py` | Nested manifests and explicit suite coverage |
| `adapters.py` | Source pins, isolated preparation, CSL overlays, compile argv |
| `runner.py` | Numerical runs, memory inspection, atomic evidence |
| `process.py` | Deadlines, process groups, log limits, simulator lease |
| `paths.py` | Fingerprints, streaming digests, contained artifact paths |
| `scoring.py` | Integrity, comparability, budgets, confirmation, aggregation |
| `provenance.py` | Evaluator fingerprints, Git state, declared research context |
| `ledger.py` | Automatic TSV journal, evidence replay, derived research views |
| `cli.py` | Commands and bounded orchestration |
| `legacy.py` | Original commands and V1 receipts |
| `telemetry.py` | Separate hosted API measurement |

`tools/run_campaign.py` is an optional batch controller outside the trusted
evaluator capsule. It freezes evaluator/candidate identities, shares baselines,
and checkpoints work; numerical validation and promotion delegate to the core
runner/scoring modules. Its plans and selection history support workflow review,
and every retained confirmation must still replay from the core evidence.
See the [campaign workflow](../experiments/campaigns.md).

SDK imports live in `benchmarks/drivers/`. The runner copies trusted drivers
into isolated work directories. Candidate overlays apply CSL only. Python
oracles cannot be replaced through the candidate mechanism; helper collisions
with upstream files are rejected.

`config/sources.lock.json` pins exact revisions. Pristine-source checks verify
Git blobs without refreshing Mac-created indexes from Linux. Profiles and
complete CSL trees have canonical fingerprints.
SDK profiles can declare shared `benchmark-libs` dependencies. These are copied
inside the isolated tree and included in its fingerprint. The adapter relocates
their relative import paths; the pinned checkout and library arithmetic stay fixed.

## Adding a benchmark

1. Add a nested profile manifest and register it in the catalog's explicit includes.
2. Declare source, precision, dimensions, grid, timing scope, work units,
   validation level, and variants.
3. Implement/reuse an adapter and independent oracle. Keep input generation,
   expected results, bounds, and timers outside candidate control.
4. Add representative cases to qualification and a relevant regression suite.
5. Test new failure gates and run an actual SDK integration check.
6. Export receipts and distinguish measured coverage from configuration.

Changes to oracle, timing, precision, layout budget, or workload require new
baselines and an explicit contract/profile change. Preserve historical records.
Formatting a trusted driver changes its digest and requires a fresh paired
baseline for promotion.

The [CSL bringup findings](csl-pitfalls.md) cover concrete pointer/export errors
and the repeated-state failure found while dogfooding this workflow.
