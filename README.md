# Cerebras kernel autoresearch

A reproducible laboratory for optimizing **CSL kernels and resident operator
chains** with the Cerebras SDK fabric simulator. Score cycles under fixed
numerical, precision, and PE budgets; record memory and useful work per cycle.
No hosted inference API key is required.

## Start here

```sh
python3 research.py list
python3 research.py validate
python3 research.py fetch
limactl start --tty=false cerebras-sdk-rosetta
tools/sdk-run python3 research.py doctor
make smoke
```

This checkout has SDK 2.10.1 in ignored `.sdk/sdk-2.10.1` and a Linux VM.
[SDK setup](docs/sdk/setup.md) describes reproduction. The host CLI uses the
Python standard library; NumPy and the Cerebras runtime come from the container.

## Research workflow

```sh
# Full numerical baseline.
tools/sdk-run python3 research.py run meshgemv.square64

# Create an editable CSL workspace; describe its hypothesis before editing.
python3 research.py candidate-init meshgemv.square64 my-gemv
# Edit experiments/candidates/my-gemv/src/*.csl.
tools/sdk-run python3 research.py challenge meshgemv.square64 \
  --variant candidate --candidate my-gemv --max-seconds 1800

# Built-in staged/fused operator-chain experiment.
tools/sdk-run python3 research.py challenge pipeline.k8-n8 --variant fused

# Evaluate a family and export compact evidence.
tools/sdk-run python3 research.py suite gemv --variant unroll --screen
python3 research.py export results/reports/local.json
```

The challenge runs a baseline, a provisional screen, full validation, and a
fresh paired confirmation. Default retention requires **1.05× cycle speedup**,
at most **10% growth in the program-object memory metric**, and every numerical
case. Retention applies to the measured profile.

See the [research protocol](docs/experiments/protocol.md) for regression suites,
candidate review, budgets, and evidence. The checked-in
[GEMV workspace](experiments/candidates/gemv-unroll/README.md) is an actual
editable candidate. Baselines and trusted host oracles stay separate.

For several ideas on one or more profiles, the [batch campaign workflow](docs/experiments/campaigns.md)
reuses matched baselines, screens candidates, checkpoints progress, and performs
fresh confirmation on eligible winners. The standard scoring gates remain fixed.

Every terminal V2 attempt and CLI decision automatically appends to
[`results/results.tsv`](results/results.tsv), including failures and screens.
[`results/leaderboard.json`](results/leaderboard.json) derives measured bests by
declared approach, the cycle/memory tradeoff front, and confirmed improvements.
Measured points remain provisional. See the [ledger contract](docs/experiments/ledger.md).

```sh
python3 research.py ledger verify             # Chain, schema, derived board
python3 research.py ledger verify --artifacts # Also replay receipts/evidence
python3 research.py ledger sync               # Recover unjournaled terminal receipts
```

## Benchmark coverage

**45 configured profiles, 16 families, 21 suites.** The
[catalog](docs/benchmarks/catalog.md) describes coverage; [results](results/README.md)
separate measured profiles from configured ones.

| Area | Workloads |
| --- | --- |
| Dense inference | Six GEMV shapes; five cold GEMM shapes; three resident GEMM sequences; scaled FFN up/down projections |
| Distributed resident FFN | Four distinct batches through gate/up GEMMs, SiLU/product, down GEMM and residual on 16 PEs; separate FP16 and BF16/FP32-accumulator contracts, every intermediate/state checked |
| Recurrent attention operator | Four resident tokens through gated delta-rule updates; BF16 Q/K/V/output, FP32 normalized operands and complete recurrent state; three value heads per key head on 16 PEs |
| Complete inference blocks | RMSNorm/SwiGLU MLP at two widths; causal RoPE attention with KV state and MLP at two contexts |
| Composed programs | Three FP32 matvec → bias → ReLU → residual chains with staged/fused variants; SDK residual norm |
| Communication/I/O | Row/column broadcast; H2D/D2H diagnostics |
| Numerical/scientific | Sparse 7-point stencil; 1D/2D/3D FFT; local matvec bandwidth |
| Exact/spatial | 256-bit multiplication; Game of Life |

Twenty-nine profiles have independent six-case oracles. Fourteen use pinned
upstream reference checks; two are diagnostics. Only the independent-oracle
tier can pass automatic promotion. Untimed examples receive no cycle score.

The [four-track inference challenge](docs/benchmarks/inference-blocks.md) groups
the larger FFN GEMM/GEMV pairs and complete gated MLP/decoder-layer workloads.
Execution qualification is recorded separately from configuration.

After registering profiles or suites, run `make docs` to regenerate the catalog
tables. `make check` rejects stale tables and requires the `all` aggregate to
contain every configured profile exactly once.

## What to optimize

Cycles are primary, with constraints. Also inspect PE count, program-object
memory, cycle spread, error bounds, and work per cycle. Host compile/experiment
wall time describes iteration cost. [Metric definitions](docs/benchmarks/metrics.md)
state exactly what each number measures.

Fusion can remove intermediate storage and improve communication across a
complete resident chain. The initial chain checks every input frame and output.
[Fusion and megakernels](docs/research/fusion.md) explains the path to larger
inference programs.

The original challenge established **264.7 → 241.3 upstream-aligned cycles**
for a 64×64 FP16 GEMV. Its [history](EXPERIMENTS.md) is preserved. V2 uses maximum
per-PE local elapsed cycles and a separate result schema; compare within one
contract. Full Qwen decoder performance and hardware TPS remain unmeasured.

The [V2 dogfood campaign](docs/experiments/2026-10-22-dogfood.md) records fresh
paired kernel improvements, rejected experiments, and workflow fixes from
actual repeated use. Its results remain scoped to the measured profiles.

## Repository map

```text
src/cerebras_research/       CLI, catalogs, execution, candidates, scoring
benchmarks/manifests/        Nested dense, composed, and SDK profiles
benchmarks/suites/           Smoke, regression, family, qualification suites
benchmarks/drivers/          Trusted SDK-side oracles and measurement helpers
kernels/                    Local composed kernels and dense-kernel notes
experiments/candidates/     Editable CSL overlays and hypotheses
experiments/campaigns/      Frozen search plans and portable fresh-run plans
config/                     Source pins, SDK manifest, VM and model targets
tests/unit/                 SDK-free acceptance and orchestration tests
tests/integration/          SDK-side numerical/timestamp contract checks
results/                    Compact reports and historical snapshots
results/results.tsv         Append-only V2 attempt and decision journal
docs/                       Metrics, protocols, architecture, setup, research
runs/                       Ignored raw evidence and compiled artifacts
.deps/  .sdk/               Ignored source checkouts and licensed SDK
```

## Quality checks

```sh
make check       # Pinned Ruff, catalog validation, unit tests, syntax checks
make sdk-test    # SDK-side timestamp and numerical contract properties
```

CI runs SDK-free checks on Python 3.10, 3.11, and 3.14. SDK qualification stays
local because the compiler/container is not redistributed. See
[contributing](CONTRIBUTING.md) and [architecture](docs/development/architecture.md).

The original `tools/kernel_research.py` and `tools/inference_benchmark.py`
commands remain compatibility entry points. Hosted API telemetry follows its
[separate protocol](docs/telemetry/api-protocol.md) and has no kernel score.
