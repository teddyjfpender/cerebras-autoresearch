# Explicit two-term mapped reduction

The earlier two-input-map variants compiled but failed in the simulator. This
separate candidate tests explicit single-step map arguments, as specified by
[SDK map documentation](https://sdk.cerebras.ai/csl/language/builtins#map).
The alias explanation is a hypothesis pending numerical/device validation.
DSRs 5/6 are shift-only and reused after shift completion; concurrent two-hop
transfers retain DSRs 2/3. X sources use1/4. All K terms and order, full output
and weight state, PE/precision/timing controls remain fixed.

## Measured outcome

Explicit map descriptors did not repair the simulator failures on any of the three profiles. The allocation-alias explanation remains unproven.

See the [immutable campaign report](../../../results/reports/2026-10-23-resident-gemm-iteration3-campaign.json). Failed receipts remain in the TSV; this source is not retained.
