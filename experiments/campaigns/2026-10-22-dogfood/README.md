# Core-kernel dogfood campaign

This campaign targets five independent-oracle cells: a small and medium complete
resident chain, tiny cold GEMM, square GEMV, and a scaled FFN-up GEMV. Ten screens
test several implementation ideas under unchanged arithmetic/PE/timing budgets.

| Idea | Hypothesis |
| --- | --- |
| Fused scalar chain | Negative control: epilogue fusion alone previously reduced memory but increased cycles |
| Scalar K-loop expansion | Expose constant indexing and remove loop control |
| Vector column accumulation | Reuse output accumulators and strided weights with FP32 vector FMAs |
| Expanded vector K-loop | Remove residual reduction-loop control from the vector implementation |
| GEMV map expansion | Remove local FMA map callback overhead |
| GEMV expansion + alignment | Test bank-layout improvements beyond unrolling |
| GEMM column expansion | Remove per-column map dispatch while retaining DSR progression |
| GEMM full local-tile expansion | Remove local K-loop control as well as column dispatch |

Initial seed: 20261023. Fresh confirmation seed: 20261024. Default gates:
1.05× cycle speedup, at most 1.10× program-object memory, all six numerical
families and every output, same evaluator/compiler/SDK/profile across pairs.

```sh
tools/sdk-run env CEREBRAS_RESEARCH_HARNESS=Codex \
  python3 tools/run_campaign.py experiments/campaigns/2026-10-22-dogfood/plan.json \
  --state runs/campaigns/dogfood-core-v1/result.json --max-seconds 7200
```

The original `plan.json` pins the measured small-chain baseline receipt and is
kept unchanged for exact resume. For a new checkout without the original raw
artifacts, use [`fresh-plan.json`](fresh-plan.json) with a new state directory:

```sh
tools/sdk-run env CEREBRAS_RESEARCH_HARNESS=YOUR_HARNESS \
  python3 tools/run_campaign.py experiments/campaigns/2026-10-22-dogfood/fresh-plan.json \
  --state runs/campaigns/dogfood-reproduce/result.json --max-seconds 7200
```

It tests the same candidate set and measures every baseline anew. See the
[persistent-log recipe](../../../docs/experiments/campaigns.md#long-sessions-and-persistent-logs)
before launching a long session.

Both finite campaigns completed. The [dogfood report](../../../docs/experiments/2026-10-22-dogfood.md)
records exact results, negative experiments and developer-experience findings.
Configuration and screens alone do not establish an improvement.

## Bounded epilogue follow-up

`followup-plan.json` tests one additional small-chain candidate after the main
campaign: vector bias, ReLU, residual addition and per-frame output stores.
Its speed gate is 2.33× the original baseline, requiring more than 5% fewer
cycles than the prior confirmed 440.5-cycle implementation. The 1.10× memory
gate, numerical bounds, PE budget, timers and fresh-pair requirements stay fixed.

Use `followup-fresh-plan.json` on another checkout to measure every baseline
anew. Give it a separate state directory. This follow-up measures one profile;
the generic source does not establish cross-shape performance.

The receipt-bound plans archive the measured invocations. A subsequent journal
output fix changes the current evaluator fingerprint; use the fresh-run plans
for new attempts on the current tree. Historical confirmations preserve their
original source capsules.
