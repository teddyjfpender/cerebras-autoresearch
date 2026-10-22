# Autoresearch protocol

1. Run `make check`; fetch pinned sources and pass `doctor` in the SDK environment.
2. Choose an independent-oracle profile. State the bottleneck, proposed change,
   expected tradeoff, and a test that could reject the hypothesis.
3. Run `candidate-init PROFILE NAME`. Edit CSL inside the candidate workspace.
   Keep the pinned baseline and trusted Python drivers fixed.
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
