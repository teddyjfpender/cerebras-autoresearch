# Cerebras autoresearch protocol

Objective: reduce simulated cycles for correct CSL kernels and complete resident
operator chains. Preserve each benchmark's precision, workload, PE budget, and
timing contract. Record memory, work per cycle, imbalance, and numerical errors.

1. Read `README.md`, `docs/experiments/protocol.md`, and nested instructions.
   Run `make check`; use `research.py fetch`, then `doctor` in the SDK environment.
2. Use the nested profile/suite manifests. Configuration is not execution proof.
   Distinguish independent oracles, upstream references, and diagnostics.
3. Keep `.deps/` pristine. Use `candidate-init` and edit CSL in
   `experiments/candidates/NAME/`. Describe the hypothesis and falsification test.
4. Keep trusted inputs, oracle bounds, timers, and repetition controls fixed.
   Never cache warmup outputs, skip timed work, or weaken precision to pass.
   Review equivalent work even when numerical checks pass.
5. Use a matched baseline and full six-case candidate validation. Automatic
   retention requires the configured cycle/memory gates and fresh paired seed.
   Cross-shape claims require complete suite coverage and worst-cell guards.
6. Use `tools/sdk-run` on this host. Default simulator threads = 1, with a fresh
   process per case. Reject shutdown failures. Bound work with `--max-seconds`.
7. Preserve source/profile/harness/SDK identity, raw output/timestamps, numerical
   errors, commands, resource observations, and artifact digests. Export compact
   reports; retain raw evidence for improvements. Historical records are immutable.
8. A new oracle, timing definition, precision, layout budget, or workload needs
   a documented contract/profile change and new baselines. V1 and V2 scores are
   distinct. Do not rewrite archived metrics during refactors.
9. Follow `docs/experiments/ledger.md`. Never hand-edit `results/results.tsv` or
   the derived board. Every V2 run and decision is journaled automatically.
   Run `ledger verify --artifacts`; use `ledger sync` after interruption and
   inspect its skipped receipts. Incomplete runs are not completed attempts.
10. Before a candidate run, set concrete `approach`, `hypothesis`, and
    `falsification` fields in candidate.json. Declare known author/model/harness
    attribution; use unknown when unavailable. New retention requires matched
    evaluator source capsules. Historical receipts need fresh baselines.

Larger composed programs are a research direction. Require full-chain output and
state validation before layer/decoder claims. Model targets, including Qwen's
BF16/hybrid attention, need explicit implementation. Simulator cycles alone do
not establish hardware TPS, energy, or hosted-service speed.

Hosted API telemetry remains separate under `docs/telemetry/api-protocol.md`.
Never describe its concurrency gain as a kernel or model decode improvement.
