# Inference acceleration campaign

The 2026-10-26 session prioritizes distributed decode attention, complete decoder
layers and prefill GEMM. The complete resident BF16 FFN and native recurrent
operators remain important inference tracks. Existing frozen family suites and
historical journal rows are preserved; added workloads receive distinct IDs,
profiles and fresh controls.

The initial attention plan ports the PR #1 hypotheses, with new paired baselines
under this checkout's evaluator. No PR performance record is imported or claimed
as a new measurement. The next plans compare a decoder softmax optimization
against its incumbent and test complete prefill projections at 16 and 64 rows.
All plans use seed 20261061 and fresh confirmation seed 20261062, minimum 1.05x
speedup, no more than 1.10x program-object memory, and the original numerical,
state, timing and PE contracts. Current evaluator identity binds every attempt.

New coverage:

- Seven distributed GQA decode profiles, including empty shards, 128/1024/4096
  context capacities, D64/D128, and fixed-budget 16/32-PE latency configurations.
- `decoder.h16-f32-ctx28`: full four-token decoder sequence through context 28–32,
  including all output and complete cache-prefix snapshots. Scaled FP32, one PE.
- `meshgemm.cold-prefill-ffn-{up,down}-batch64`: 64-row cold FP16 projections,
  including full communication. Distinct from the original 16-row profiles.

These are simulator operator workloads, not a complete model or hardware TPS.
Do not pool absolute cycles from different contracts or multiply their speedups.
Record every terminal attempt automatically; export full numerical replay and
artifact audits before retaining or reporting new results. Use the standard
finite `tools/run_campaign.py` workflow and only one SDK device worker.

## Evidence gates

The priority portfolio comprises seven attention profiles, three complete
FP32 decoder contexts and four cold FFN prefill projections. Qualify all cells
actually claimed, apply complete suite/worst-cell guards, and independently
replay their numerical/state/timestamp artifacts. Do not report the portfolio
complete merely because the first latency profile confirms.

New kernel hypotheses are `decoder-vector-softmax-reciprocal`,
`gemm-prefill-dsr-dispatch`, and `attention-newton-polynomial`. The latter tests
bounded exponential arithmetic after the PR descriptor bottlenecks are removed;
the host properties test its combination with reciprocal normalization. Compare
with the strongest qualified incumbent, not only the reference speedup ratio.

The initial attention worker is followed serially by decoder and prefill plans;
each campaign has a finite budget. Its durable controller/audit checkpoints are
under ignored `runs/campaigns/inference-acceleration/`. Every retained result
still comes from the ordinary runner, full/fresh gates and journal. The remaining
attention coverage and math follow-up plans are separate subsequent research.


## Wider-output projection and follow-up hypotheses

`vocab-projections.json` adds 80→512 and 80→1024 resident FP16 output-head
GEMVs with complete reduction on 16 PEs. [Contract](../../../docs/benchmarks/vocab-projection.md).
These are two additional portfolio cells requiring new full/fresh controls;
no sampling or native-model speedup is claimed.

`gemm-prefill-column-dsr` adds a no-extra-tile prefill fallback: preload invariant
accumulator descriptors, then reset their addresses perK while retaining the
original column layout and FP16 FMA order. It is screened alongside the stronger
row-layout candidates before the queued prefill plan is frozen.

`decoder-qk-followup.json` tests causal vector QK accumulation on capacities 16/32
with the qualified short-context path preserved. It imports the current matched
initial/fresh controls only after the first decoder campaign is complete.
The original timing/oracle/PE contracts and every full cache-prefix/output store
remain fixed. Source preparation alone establishes no new gain.


## Padded-layout memory repair

The first 64-row screens show why storage matters: the fast full-tile map-K
variant grows program memory by 22.8% on FFN-up, beyond the fixed 10% gate.
`gemm-prefill-padded-map` instead adds one halfword per output column and a
single-column temporary, with a complete timed ABI compaction. Its first
compiler attempt failed on an i16 constant range check; the failed source and
receipts remain frozen. `gemm-prefill-padded-map-v2` widens that compile-time
check without changing runtime arithmetic. `gemm-prefill-padded-stream-pack`
tests a compact map-driven copy implementation. The standalone host layout
properties verify address/bit preservation but do not execute CSL or establish
a speedup. Both corrected variants require normal SDK full/fresh validation.

The pending `decoder-qk-followup.json` plan now prioritizes these two 64-row
prefill repairs before its three decoder QK cells. It has not changed the
already-running `prefill-padded-followup.json` plan or any frozen candidate.
Source/plan identity is frozen separately when each campaign starts.
Bounded column-unroll4/8 variants are also recorded; their slower screens
remain rejected rather than entering the retained portfolio.

## Cold batch64 buffer lifetime and the memory frontier

The 2026-10-26 user clarification explicitly asks for the speed/memory frontier,
including substantially larger memory footprints. The default 10% promotion
policy remains separately labeled. `prefill64-frontier.json` requires full
initial/fresh six-case execution of each correct implementation without a
relative-memory screen. Its higher-memory full-tile implementation is measured
alongside buffer reuse, packing/output aliasing and bank-offset hypotheses.
The derived [paired frontier](../../../docs/experiments/frontier.md) reports
all qualified points, their conservative cycles/bytes and their default-policy
eligibility. It does not invent new journal decisions or hardware/model TPS.

The first dead-weight-buffer gather implementation is freshly qualified on
FFN-up: 34929 to 31984 cycles, 14688 to 14752 program-object bytes per PE,
1.092077x speedup and 0.44% memory growth. Its four full paired arms pass
independent raw-array/timestamp replay (24 case executions). The corresponding
down attempt was rejected by an overly narrow `shape_specific` manifest flag,
before executing the kernel. `gemm-prefill-dead-weight-gather-v2` corrects that
metadata in a new workspace and preserves the failed receipt. Two scatter
packing attempts also preserve compiler failures from the reserved CSL keyword
`packed`; their frozen source is not silently repaired.

Cold buffer reuse is valid only after the last local send/receive completion
barriers and all four multiply phases. The fixed profiles use no warmup and one
timed iteration. Every transpose/copy remains timed. The output-alias candidate
instead exports the completed column-major result from the dead weight buffer,
avoiding the final copy while retaining the named output ABI. None of these
destructive working-buffer variants is a repeated/resident GEMM implementation.
