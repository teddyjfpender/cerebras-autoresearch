# Autoresearch protocol

1. Run `make check`; fetch pinned sources and pass `doctor` in the SDK environment.
2. Choose an independent-oracle profile. State the bottleneck, proposed change,
   expected tradeoff, and a test that could reject the hypothesis.
3. Run `candidate-init PROFILE NAME`. Edit CSL inside the candidate workspace.
   Keep the pinned baseline and trusted Python drivers fixed.
   Fill in `approach`, `hypothesis`, and `falsification` in candidate.json first;
   unfinished templates are rejected before compilation.
4. Run `challenge PROFILE --variant candidate --candidate NAME` with a finite
   `--max-seconds` budget. Start with small profiles on the emulated host.
5. Review numerical errors, source hashes, raw evidence, and memory observations.
   The challenge performs a fresh paired confirmation before retention.
6. Record exact paths/profile, then evaluate the relevant regression suite before
   proposing a family-wide improvement.

The supplied `gemv-unroll` workspace is shape-specific: its literal expansion
uses 16 local rows. Parameterized candidates can span shapes, but every claimed
cell needs its own measurement.

## Suites

```sh
tools/sdk-run python3 research.py suite gemv --variant baseline --seed 7
tools/sdk-run python3 research.py suite gemv --variant unroll --seed 7
python3 research.py compare-suite BASE_SUITE/result.json CAND_SUITE/result.json
# Run both arms again with --seed 8, then verify fresh evidence and identities:
python3 research.py confirm-suite BASE_SUITE/result.json CAND_SUITE/result.json \
  FRESH_BASE_SUITE/result.json FRESH_CAND_SUITE/result.json
```

Repeat both arms with a fresh seed. `compare-suite` checks coverage, evidence,
memory, geometric mean, and worst cell; its result remains provisional. Retain
only after `confirm-suite` accepts the fresh pair with unchanged candidate/toolchain.
Profile-level `challenge` automates that sequence for one cell.

For several hypotheses, use the [batch campaign helper](campaigns.md). It shares
matched baselines, ranks screens, and resumes at completed-run boundaries.

The `gemv`, `gemm`, and `fusion` suites contain independent-oracle profiles
only. `fusion --variant fused` evaluates all three composed-chain sizes.
Mixed reference/diagnostic suites support qualification and investigation;
they cannot enter the automatic suite promotion gate.

Upstream-reference and diagnostic profiles remain useful separate experiments.
They cannot enter the independent-oracle promotion league until their adapter
has an adequate output oracle, timed region, and adversarial cases.

## Evidence and lifecycle

Each run under `runs/v2/TIMESTAMP-PROFILE-UUID/` contains an atomic `result.json`,
isolated `work/` source, compile/run logs, raw `.npz` numerical/timestamp data,
per-case JSON receipts, and per-object memory observations. Source, harness,
profile, and raw artifact digests bind the result to its inputs. Failed and
cancelled attempts preserve partial evidence.

Suite receipts enumerate requested profiles, completed runs, and failures.
Budget exhaustion is explicit. A shutdown fault cannot qualify even after a
numerical PASS. Process groups are killed on timeout, cancellation, or log
flooding. A lease prevents overlapping runner commands in the execution
environment; use one benchmark runner in the VM.

`export OUTPUT.json` saves compact review metadata; preserve the corresponding
raw artifacts. `storage` reports their size without deleting evidence. Cleanup
is manual after archiving retained results.

Terminal V2 runs, suites, challenges, and CLI comparisons/confirmations append
to `results/results.tsv`. The [ledger contract](ledger.md) specifies attribution,
source capsules, recovery, Git history checks, and the derived board. A validated
measurement is provisional until a fresh confirmation event passes replay.

For already measured matching arms, `confirm INITIAL_BASE INITIAL_CAND
FRESH_BASE FRESH_CAND` performs the profile confirmation without repeating the
simulator runs. Both arms require the same recorded evaluator fingerprint and
compiler arguments across seeds. Legacy receipts without evaluator capsules
need new baselines; their measurements remain in the historical journal.

## Candidate review

The complete requested arithmetic must execute inside the timer. Do not cache
a warmup answer, skip repetitions, weaken a bound, change outputs, or move work
outside the region. Dense/composed overlays lock timestamp and repeat-control
functions. This is a guard, not a formal proof: review other source changes for
equivalent work and resource use.

Precision, layout, routing, and model changes are valid research axes. Give them
distinct profiles and compare Pareto tradeoffs. A fixed-profile candidate cannot
silently use more PEs or weaker accuracy.

Simulator results do not establish hardware TPS, energy, sustained service
capacity, or a faster hosted model. API telemetry has its own protocol.

## Descriptive paired-array audit

After full validation, `tools/paired_array_report.py` compares explicitly named
raw arrays from matched baseline/candidate receipts. Its manifest binds each
receipt digest; it checks profile, seed, evaluator, SDK, compiler arguments and
harness identity before reporting changed words and numeric differences. Use
`--bf16-keys` for raw BF16 uint16 arrays so their values are decoded correctly.

```sh
tools/sdk-run cs_python tools/paired_array_report.py \
  --manifest runs/paired-arrays.json \
  --keys bits_g bits_u bits_v bits_d bits_y \
  --bf16-keys bits_g bits_u bits_v bits_d bits_y \
  --output results/reports/paired-arrays.json
```

This read-only report describes these arrays. It does not grant retention,
replace the independent numerical/core replay, or prove equivalent requested work.
Keep its immutable output and audit-tool identity alongside the full evidence.
