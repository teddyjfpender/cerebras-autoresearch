# Four inference challenge tracks

Scope: implement and qualify the larger cold FFN GEMM pair, medium FFN GEMV
pair, complete resident gated MLP, and stateful multi-token decoder layer.
All scores are simulator cycles, with explicit precision and PE budgets.

| Track | Required workload | Completion evidence |
| --- | --- | --- |
| Prefill | Existing M16 K80 N272 GEMM and down projection | Six-case current-evaluator baseline receipts for both; editable candidates and suite |
| Projection | Existing 160x544 and 544x160 GEMV | Six-case current-evaluator baselines for both; candidate/suite workflow |
| Gated MLP | RMSNorm, gate/up projections, SiLU product, down projection, residual; distinct input frames | Independent FP64 nonlinear oracle, fixed error contract, actual SDK runs |
| Decoder | Pre-norm causal multi-head attention with RoPE, resident KV cache, attention residual, pre-norm gated MLP and residual over four successive tokens | Every token output, every cache snapshot and final cache checked, actual SDK runs |

The new resident workloads are deliberately explicit FP32 local blocks at a
fixed one-PE budget. They implement the entire named computation; they are not
a port of WaferLLM's distributed FP16 decoder or a Qwen implementation. These
profiles enable composition research. Scaling their spatial decomposition needs
new separately measured profiles, preserving the original local scores.

New decoder inputs include an existing cache and four independent token vectors.
All token computation and cache snapshots execute inside one timed launch.
The input vectors are teacher-forced layer inputs; this is a decoder-layer
sequence, not a language-model sampling loop. Cache slots not yet written must
remain unchanged. Existing GEMM retains its cold single-invocation contract.

Qualification is separate from retained optimization. Screens cannot qualify;
retention requires six cases and fresh paired confirmation at the standard gates.
Raw receipts stay under ignored runs/v2; compact evidence reports are versioned.

## Initial implementation evidence

All eight profiles are registered in `inference-blocks`, with separate
`prefill-ffn`, `decode-ffn`, `gated-mlp` and `decoder` suites. Both new adapters
support candidate overlays and the standard challenge/ledger pipeline.

The first small gated MLP and ctx4 decoder SDK screens passed. Two earlier
CSL compile failures (export mutability and pointer type) are preserved in the
journal. [Bringup report](../../results/reports/2026-10-22-inference-blocks-bringup.json)
contains their exact receipts and source/evaluator identities. Screens do not
establish full qualification. Host checks passed 61 unit tests and the SDK
passed 10 driver contract tests.

At initial bringup, a detached, bounded eight-profile campaign was launched from
`experiments/campaigns/inference-blocks/fresh-plan.json`, with checkpoint
`runs/campaigns/inference-blocks/result.json` and persistent `console.log`.
That initial campaign is superseded by the qualification handoff below.
Completion requires all eight full-case baselines and reviewed candidate attempts;
no new improvement was retained at bringup.


## Qualification handoff after the strict state check

The cache oracle now saves a separate exact-bit gate, tested with signed-zero
and minimum-subnormal mutations. SDK contract coverage is 11 tests. The first
campaign was intentionally stopped during its signed GEMM case; its failed
receipt and frozen-evaluator refusal remain recorded, rather than being reused.

The corrected qualification worker uses `qualification-plan.json` and
`runs/campaigns/inference-blocks-qualification/result.json`. It measures all eight
full baselines first. A bounded local follow-up waits for that specific process
to exit successfully, then imports its receipts into a candidate campaign under
`runs/campaigns/inference-blocks-optimization/`. Only one benchmark writer runs
at a time. Neither launched work nor a checkpoint is completion evidence.


## Full qualification achieved

All eight required baselines passed six cases under the frozen evaluator. The
[qualification report](../../results/reports/2026-10-22-inference-blocks-qualification.json)
contains exact paths, digests and resource/numerical data. This completes the
baseline qualification gate for each track. The completed first iteration
reviewed candidates for all eight cells. Both MLP widths and both decoder
contexts retained fresh-paired gains; the first dense GEMM/GEMV candidates
did not clear the gates. The [first-iteration report](../../results/reports/2026-10-22-inference-blocks-first-iteration.json)
audits all 44 imported/run/decision receipts, and the
[results table](../../results/README.md#inference-block-optimization--confirmed-results)
links the four complete raw numerical replays.

The second optimization iteration and twelve-hour dogfood session remain
active. Completion of the four baseline tracks does not establish suite-wide
optimization, Qwen implementation or hardware throughput. The additional
reusable-GEMM benchmark is still staged and unqualified.
