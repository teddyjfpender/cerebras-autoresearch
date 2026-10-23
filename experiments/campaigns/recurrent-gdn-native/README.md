# Native-width recurrent head-group challenge

The registered T4/48PE and T8/96PE profiles use native K128/V128 dimensions for
one key-head group and three value heads. All native-width states and outputs
are reconstructed, every physical immutable/prefix replica is checked, and all
normalization/decay/state operations and full snapshots remain timed.

The reference already uses the freshly confirmed FP32 vector-state algorithm
from the scaled operator. Its native-width/rectangular adaptation was frozen
before device observations. Qualify T4 first, then T8; registration and upstream
CPU correspondence alone do not qualify device programs.

Initial paired seed20261038; fresh confirmation seed20261039. Every terminal V2
attempt and decision is automatically journaled in results/results.tsv. Preserve
source/profile/harness/SDK identities and full raw evidence. This is a model-shaped
operator after QKV/convolution and precomputed g/beta, before gated normalization
and output projection, with one of the model's key-head groups. No model TPS or
production-runtime comparison follows. Each profile fixes its own PE budget.


## PR1 transfer and fresh reproduction

The follow-up at seeds20261043/20261044 keeps the same numerical, timing and PE
contracts. Persistent DSR row addresses, a bank-offset accumulator and bounded
ascending-K unrolling preserve every FP32 FMA term. T4 unroll8 confirms2.428316×
with complete24-case raw replay and bit-identical compared arrays. T8 and exact
complete-suite confirmation are separate required gates.

After the root protocol, SDK fetch/doctor and checks, this portable plan creates
new matched controls at seeds20261045/20261046. It has no references to ignored
local run receipts:

```sh
tools/sdk-run python3 tools/run_campaign.py \
  experiments/campaigns/recurrent-gdn-native/reproduce-pr1-unroll8.json \
  --state runs/campaigns/local/native-pr1-unroll8.json --max-seconds 7200
python3 research.py ledger verify --artifacts
```

The campaign screens, fully validates and freshly confirms each profile. A
cross-profile suite claim additionally requires exact nested-suite coverage and
`confirm-suite`, including the configured worst-cell guard. See the root
experiment protocol and suite assembly tools. A plan is not execution proof.

Plans containing `baseline_receipt` or `fresh_baselines_from_campaign` are local
evidence continuations. Cached controls require matching source/profile/harness/
SDK identities. Use the fresh reproduction plan on a new checkout; historical
raw artifacts require their evidence bundle for re-audit. Never edit the TSV to
reproduce a score. Full16-key/48-value-head qualification is a separate contract.
