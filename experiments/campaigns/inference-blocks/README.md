# Four inference tracks

`fresh-plan.json` measures all eight profiles from fresh matched baselines,
then screens an explicit hypothesis per cell. Eligible screens proceed through
six numerical families and a fresh paired confirmation at 1.05x speedup and
1.10x maximum program-object memory. Rejected ideas remain recorded.

GEMM hypotheses cover direct FMA dispatch, strided output-column vectors and
contiguous row-major accumulation with a final ABI transpose. GEMV hypotheses
cover literal unrolling and bank alignment. New complete block
candidates vectorize dense projections while retaining normalization,
nonlinearities, residuals, RoPE, attention, cache writes and all output/state
stores. A decoder follow-up also reuses normalized attention probabilities
and fixed D=8 RoPE frequencies. Parameterized source is useful only on profiles
actually measured.

```sh
tools/sdk-run python3 tools/run_campaign.py \
  experiments/campaigns/inference-blocks/fresh-plan.json \
  --state runs/campaigns/inference-blocks/result.json --max-seconds 28800

python3 tools/run_campaign.py experiments/campaigns/inference-blocks/fresh-plan.json \
  --status --state runs/campaigns/inference-blocks/result.json
```

For long emulated runs, use the documented detached launch with a persistent
console log. Resume the exact plan/state under the same evaluator and candidate
identities. `--retry-failed` retries incomplete cells without discarding failures.
Do not run a qualification suite and a campaign concurrently in the same VM.

The [numerical/timing contract](../../../docs/benchmarks/inference-blocks.md)
describes the complete workloads and their claim boundaries. The portable plan
is not a claim that any candidate passes or wins.


## Qualification before optimization

`qualification-plan.json` runs six-case baselines with no candidates. Its
`no_confirmed_gain` cells mean no optimization was attempted; each cell's
baseline receipt establishes qualification. Reuse those matched baseline paths
as `baseline_receipt` in a copy of the fresh optimization plan. The controller
replays imported evidence and requires the current profile, seed and evaluator.
This avoids repeating qualification baselines during the search.

An incomplete or failed qualification must be resolved before claiming all
four tracks qualify. A candidate failure does not remove a valid baseline.
The standard full-case and fresh-confirmation gates remain unchanged.

## Map-over-K pointer correction

`map-k-correction-plan.json` tests the separate `gemm-row-major-map-k-base`
revision on both FFN GEMM shapes. The original `gemm-row-major-map-k` failed
CSL compilation and stays frozen in its first checkpoint. The revision sets
a many-item DSD base, then adds the FP16 row offset. It remains unmeasured.

This plan creates fresh baselines by default. Within a matching evaluator, a
session copy can import exact validated baseline receipts and completed-campaign
fresh-baseline evidence as documented above. A plan, source correction or host
overlay check is not a retained result. Run one campaign at a time.
