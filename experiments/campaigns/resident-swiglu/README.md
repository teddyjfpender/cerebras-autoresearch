# Distributed resident SwiGLU research

Qualify the full six-case small baseline first, then the scaled FFN cell. These
plans contain no candidate: a completed validated baseline is execution evidence,
not an improvement. Preserve every compiler/numerical/shutdown failure in TSV.

```sh
tools/sdk-run env CEREBRAS_RESEARCH_HARNESS=Codex python3 tools/run_campaign.py experiments/campaigns/resident-swiglu/qualification-small.json --state runs/campaigns/resident-swiglu/small-result.json --max-seconds 1800
tools/sdk-run env CEREBRAS_RESEARCH_HARNESS=Codex python3 tools/run_campaign.py experiments/campaigns/resident-swiglu/qualification-scaled.json --state runs/campaigns/resident-swiglu/scaled-result.json --max-seconds 3600
```

After qualification, use candidate-init and frozen full/fresh campaigns for
projection layout, receive-buffer resets, activation evaluation and combinations.
All five intermediates/outputs and canonical/boundary weight state remain checked.
A fixed new precision, state or timer contract needs new baselines; existing
resident GEMM and local FP32 decoder results stay on their original tracks.

The first optimization is complete: both full/fresh cells and the complete suite
are confirmed. See the [48-case evidence](../../../results/reports/2026-10-23/resident-swiglu/initial-suite-confirmed.json).
The conservative suite gain is 1.415571x with unchanged precision/timing/state
contracts. Candidate sources and all unsuccessful screens remain in the journal.
