# Resident vocabulary projection

`vocab-projection` adds two scaled output-head GEMVs: 80 FP16 hidden values
projected to 512 or 1024 logits. Both use the existing fixed 4×4 PE MeshGEMV
contract, one warmup and ten timed repetitions, with the complete reduction
inside every timed repetition. Every replicated output is checked against the
independent FP64 reference on exactly quantized inputs and its original dense
forward-error bound. The six input families and fresh-pair gates are unchanged.

The purpose is to expose a wider-output projection and its compute/communication
tradeoff after the decoder. These are representative scaled dimensions, not a
native model vocabulary, and the fixed FP16 accumulation contract differs from
BF16/FP32-accumulator inference. Token selection, top-k, sampling and a model
loop are outside this benchmark. No model TPS is established.

The parameterized source is still pinned WaferLLM. Initial hypotheses test
literal expansion of all local terms, bank alignment, and their combination.
`vocab-unroll-align` asserts the fixed local row count of 20, preserves the
weight ABI, full reduction, all warmup/timed iterations and FP16 arithmetic,
and is tested separately on each output width. Initial source preparation is
not execution proof: both full six-case controls and fresh seeds are required.

```sh
tools/sdk-run python3 tools/run_campaign.py \
  experiments/campaigns/inference-acceleration/vocab-projections.json \
  --state runs/campaigns/inference-acceleration/vocab-projections-state.json \
  --max-seconds 10800
```

Use the complete two-cell suite and worst-cell guard before a shared-source
speedup claim. Preserve all rejected screens and raw numerical/timer evidence.
