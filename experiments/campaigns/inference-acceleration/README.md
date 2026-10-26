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
