# Distributed resident decode attention

Seven independent-oracle profiles exercise one grouped-query attention group
across context shards. Four distinct supplied FP32 queries append K/V and produce
four complete outputs without host traffic between tokens. Projections, RoPE,
MLP, sampling and inter-wafer transport are outside this operator's contract.

The baseline, independent FP64 oracle and candidate arithmetic are ported from
[PR #1 at 5e818e1](https://github.com/teddyjfpender/cerebras-autoresearch/pull/1).
Its historical speedups are not imported into this checkout's journal. All new
retention requires this evaluator's matched six-case runs and fresh paired seed.
Candidate workspaces were initialized against the imported baseline through
`candidate-init`; source/manifest identities remain frozen per campaign.

| Profile suffix | PEs | Slots/PE | Head width | Queries/KV head | Initial/final context |
| --- | ---: | ---: | ---: | ---: | ---: |
| `d64-ctx128` | 4 | 32 | 64 | 4 | 124/128 |
| `d64-boundary` | 4 | 32 | 64 | 4 | 30/34 |
| `d64-ctx1024` | 32 | 32 | 64 | 4 | 1020/1024 |
| `d64-ctx4096` | 128 | 32 | 64 | 4 | 4092/4096 |
| `d128-ctx512` | 32 | 16 | 128 | 2 | 508/512 |
| `d64-ctx128-latency16` | 16 | 8 | 64 | 4 | 124/128 |
| `d64-ctx128-latency32` | 32 | 4 | 64 | 4 | 124/128 |

Every ID starts with `decode-attention.`. The context 128 profiles have identical
global inputs and useful work at different PE budgets. Compare speedups only
within one fixed profile. The boundary case crosses cache-shard ownership and
includes empty shards; it tests causal masking and zero local contributions.

The baseline already uses vector QK/PV FMA. Local maxima are gathered, the global
maximum is broadcast, and packed denominators/numerators are sum-reduced.
The coordinator normalizes and stores every token's output. Final cache bits,
including untouched slots and signed zeros, must match the independent oracle.
Every output obeys the fixed `1e-4 + 1e-4*abs(reference)` bound. No full cache
prefix snapshots are instrumented in this contract. Do not equate its cycle
counts with the decoder-layer benchmark, which stores every cache prefix.

Primary `coordinator_cycles` is PE(P-1,0)'s own 48-bit clock interval, beginning
before dispatch and ending after the last output store. It encloses the resident
sequence and collectives. Every PE's local interval is also recorded; maximum
local cycles are diagnostic here. `cycles_per_token` divides the full interval
by four and does not establish model TPS. Host movement is excluded.

Six families and fresh seeds exercise signed/positive values, zero queries,
cancellation, one-hot queries, and sharply peaked/mixed-scale softmax. The SDK-side
properties also check causal masking, exact cache bits, repeated normalized FP32
mantissas for the reciprocal candidate, source controls and timing replay.
The layout, cache appends, dispatch/collectives, causal maximum scan and timers
are protected; local arithmetic is editable. These text guards are not a proof
of arbitrary candidate semantics. Review equivalent timed work.

```sh
tools/sdk-run python3 tools/run_campaign.py \
  experiments/campaigns/inference-acceleration/attention-latency32.json \
  --state runs/campaigns/inference-acceleration/attention-latency32-state.json \
  --max-seconds 7200
```

The first campaign rechecks descriptor reuse, vector maxima and Newton
normalization under the current SDK route. Full shape coverage and independent
raw-array/timestamp replay remain separate completion gates. Configuration and
host properties are not device execution proof. Hardware/model TPS is unmeasured.


## Current SDK evidence

The 32-PE latency profile is freshly confirmed at 5.103046×, 220273→43165
coordinator cycles, with 4.33% program-object memory growth. All 24 full paired-run
case executions pass independent numerical/state/timestamp replay. This
reproduces the frozen PR candidate rather than improving beyond it.
[Complete current evidence](../../results/reports/2026-10-26/inference-acceleration/attention-latency32-confirmed.json).
The other six profiles still require current full/fresh qualification.
