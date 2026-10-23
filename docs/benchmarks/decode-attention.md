# Sharded GQA decode attention

A resident decode attention operation for **one KV head shared by G query
heads**, with its cache sharded across P on-wafer PEs. Inputs are supplied
after positional encoding. Projections, RoPE, MLP, model sampling and
inter-wafer transport are outside this operator's contract. Precision is FP32.

**5.103x confirmed on `d64-ctx128-latency32`, with identical 32-PE budgets.**
The [five-times report](../../results/reports/2026-10-23-attention-5x.json)
and [24-case raw replay](../../results/reports/2026-10-23-attention-5x-replay.json)
record six cases at each of two seeds in both arms. This is one profile's
simulated kernel improvement. Hardware/model TPS and training throughput remain
unmeasured. The earlier
[readiness report](../../results/reports/2026-10-23-attention-readiness.json)
is an immutable record from before SDK installation.

This operator targets resident KV access, grouped-query reuse, stable softmax
and on-wafer communication: concrete latency costs in inference. Separate
coarse and fine shards expose the cost of distributing work across the fabric.

| Profile suffix | Compute PEs | Slots/PE | Head width | Queries/KV head | Initial/final context |
| --- | ---: | ---: | ---: | ---: | ---: |
| `d64-ctx128` | 4 | 32 | 64 | 4 | 124/128 |
| `d64-boundary` | 4 | 32 | 64 | 4 | 30/34 |
| `d64-ctx1024` | 32 | 32 | 64 | 4 | 1,020/1,024 |
| `d64-ctx4096` | 128 | 32 | 64 | 4 | 4,092/4,096 |
| `d128-ctx512` | 32 | 16 | 128 | 2 | 508/512 |
| `d64-ctx128-latency16` | 16 | 8 | 64 | 4 | 124/128 |
| `d64-ctx128-latency32` | 32 | 4 | 64 | 4 | 124/128 |

The latency points run the identical global context-128 workload on finer
cache shards. A host property checks every quantized input bit and dense output
against the original 4-PE profile. They expose the tradeoff between latency and
PE/program-memory cost. Speedup claims compare baseline and candidate within
one fixed profile, with identical PE counts; cross-profile parallelism is a
separate resource tradeoff.

SDK compilation rejected the initial larger shards for insufficient PE memory.
Those configurations now use twice as many PEs, at unchanged context and
precision. New profile fingerprints and fresh baselines keep those budgets
separate from the original failed attempt. The stage-one report qualifies the
4-PE source track; the final >5x report qualifies the 32-PE latency source track.
Other profile results must be labeled by their actual validation tier.

All IDs start with `decode-attention.`. Four distinct supplied tokens execute
in one resident sequence, without host traffic between tokens. The boundary
cell crosses shard ownership and includes completely empty shards. These are
representative head widths and context shards; FP32 inputs and teacher-forced
queries do not implement a production language model.

## Semantics

For token t at position L+t, its owner appends K/V to the resident cache.
Each PE computes all G queries against its local causal keys. A gather of local
maxima and a broadcast establish the global stable-softmax shift. Each PE
computes denominators and weighted values, then a single packed sum reduction
communicates G*(D+1) floats. The coordinator normalizes and stores every output.

Device cache layout is dimension-major inside each shard; the independent host
oracle uses `[P,S,D]`. The starting reference already uses DSD vector FMA for
QK/PV and hoists square-root scaling outside inner loops. It has not been
compared against Cerebras's proprietary production kernels. Collective setup
follows the public SDK example at
`06ffd5961e8d35b6324e0ebf6595fe4a5a87c2e1` (Apache-2.0).

## Correctness and timing

`sharded_gqa_decode_fp32_v1` uses dense NumPy FP64 semantics from exactly
quantized FP32 inputs, independently of the device reduction order. Every
token output must satisfy `abs(error) <= 1e-4 + 1e-4*abs(reference)`. This is a
fixed accuracy requirement, not a universal error theorem. Every final K/V
cache bit must match, including untouched slots and signed zeros. Readback is
outside timing; full cache history instrumentation is not timed.

Six families cover positive, signed, zero queries, cancellation, one-hot and
sharply peaked/mixed-scale inputs. Zero queries still require nonzero outputs.
Fresh seeds and six-case device evidence are mandatory for promotion. Hidden
tests and model-derived activation traces would further strengthen a public
judge. Source-region guards do not prove arbitrary arithmetic equivalent.

Primary **`coordinator_cycles`** uses PE `(P-1,0)`'s own 48-bit clock, from
before the initial dispatch broadcast through the final output store. It
includes KV appends, local work and all reductions. Local PE intervals remain
diagnostic. `cycles_per_token` divides the sequence by four; it is not model
TPS. Compilation, initialization and host transfers are excluded.

Candidates cannot change layout, fixed data declarations, cache appends, the
causal local-max scan, collective control, output dispatch or timing. Local
arithmetic, the global-max merge and normalization/stores inside the guarded
compute region are editable. Both arithmetic hooks contain the original
reference bodies unchanged. Each source track has matched paired baselines.
Precision, PE budget,
numerical bounds and <=1.10x program-object
memory gate stay fixed. Actual compiled memory/footprint needs SDK inspection;
source buffer estimates do not certify SRAM fit. Work units count QK/PV
multiply/add arithmetic; nonlinear and communication work remains timed.

## Incremental results

1. `attention-vector-exp`: vector subtraction and map of exp, with future slots
   masked before exponentiation; QK/PV and all distributed control stay fixed.
2. `attention-polynomial-exp`: stage one plus degree-ten exp(x/16), squared four
   times, on [-16,0], and an SDK-exp fallback outside that range. Host checks
   stress approximation boundaries and all profile/case/seed combinations.

Both compiled and passed a positive-case screen on the original source track,
but gave no meaningful gain: 380,038 baseline cycles, 378,094 vector-exp cycles,
and 381,154 polynomial-map cycles. Neither is retained.

Phase instrumentation identified descriptor setup and repeated normalization
divisions as larger costs. `attention-dsr` keeps accumulator descriptors in
DSRs; `attention-dsr-normalize` adds one reciprocal per head and a vector output
multiply. Against the fresh baseline, the latter improves the maximum of six
cases from **380,121 to 209,313 sequence cycles**, or 95,030.25 to 52,328.25
cycles per token. The same numbers hold at seed 20261024. Maximum program-object
memory falls from **44,064 to 43,552 bytes per PE**. This confirmation applies
to the earlier 4-PE `d64-ctx128` source track only; see the
[stage-one report](../../results/reports/2026-10-23-attention-sdk-stage1.json).

Subsequent candidates tested bank-offset accumulators, bounded QK/PV unrolling,
vector merging of gathered maxima and reciprocal normalization. Polynomial
softmax and bank-separated PV variants failed to improve their prior stages.
The strict gate also rejected the vector-max candidate's 4.97x screen.

`attention-newton` combines descriptor reuse, a bank-offset QK accumulator,
vector query scaling and normalization, bounded unrolling, a vector maximum
merge, and four FP32 Newton reciprocal steps. The stable-softmax denominator
is finite and at least one. The candidate's supported PE/shard bounds keep it
inside the independently checked [1,8192] reciprocal range. All 8,388,608
normalized FP32 mantissas pass a host error check below 1.4e-7; separate device
checks validate the complete outputs and cache. No precision or tolerance was
weakened.

On **`d64-ctx128-latency32`**, the maximum of six cases falls from **220,273 to
43,165 sequence cycles**, or 55,068.25 to 10,791.25 cycles per token. Both
seeds, 20261023 and 20261024, give **5.103x**. Maximum program-object memory is
**30,480 versus 29,216 bytes per PE**, a 4.3% increase under the 10% gate.
Both arms use 32 compute PEs and identical workload/timing contracts. These
results are independent of the earlier 4-PE ratio; gains are not multiplied.

The incremental campaign covers the coarse and fine context-128 profiles with
the ordinary 1.05x gate. The focused target campaign requires
**5.01x versus the original vectorized reference**, at both fresh seeds and
under the memory gate, and is confirmed on the 32-PE latency profile.
Family-wide >5x requires every profile to pass. Never multiply unrelated gains or treat CPU
model time as WSE performance.

```sh
# Optional NumPy-only checks, with no device scores.
python3 -m pip install -r requirements-numerics.txt
make host-numerics

# Qualify the actual device programs with a finite budget.
tools/sdk-run python3 research.py suite decode-attention --max-seconds 14400

tools/sdk-run python3 tools/run_campaign.py \
  experiments/campaigns/decode-attention/incremental.json \
  --state runs/campaigns/decode-attention-incremental.json --max-seconds 14400

tools/sdk-run python3 tools/run_campaign.py \
  experiments/campaigns/decode-attention/target-5x.json \
  --state runs/campaigns/decode-attention-target.json --max-seconds 14400
```

Preserve raw receipts and replay with `tools/replay_numerics.py` before a
submission. Changed evaluator/candidate sources require new baselines and
campaign state. Physical-WSE and model-level checks follow simulator validation.
