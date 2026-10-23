# Resident reusable GEMM qualification

Three FP16 profiles compute four resident input matrices against shared resident
weights on a 4x4 compute grid. Each six-case run validates every output frame,
canonical inputs, each intermediate working-W0 snapshot and raw timestamps.
See the reusable-GEMM contract after integration for exact bounds and timing.

The tiny profile must qualify before starting the two scaled FFN profiles.
These plans contain no optimization candidates: a `no_confirmed_gain` cell
means no search was attempted. Qualification requires its baseline receipt to
have `status=validated`, all six numerical families and intact artifact/source
provenance. A failed baseline produces `baseline_failed` and is not qualified.

```sh
tools/sdk-run python3 tools/run_campaign.py \
  experiments/campaigns/reusable-gemm/qualification-tiny.json \
  --state runs/campaigns/reusable-gemm-qualification/tiny-result.json \
  --max-seconds 1800
# Verify the tiny baseline before launching the scaled pair.
tools/sdk-run python3 tools/run_campaign.py \
  experiments/campaigns/reusable-gemm/qualification-ffn.json \
  --state runs/campaigns/reusable-gemm-qualification/ffn-result.json \
  --max-seconds 7200
```

Preserve failures; correct implementation problems without relaxing the oracle,
PE budget, precision or timing. Keep one SDK writer. A changed evaluator needs
new matched baselines. After all three qualify, assemble the named suite from
hash-bound full receipts and replay all 18 numerical/state/timestamp cases.
Baseline qualification does not establish an optimized implementation.
