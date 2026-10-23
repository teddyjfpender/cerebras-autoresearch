# Sharded GQA decode attention

A resident decode attention operation for **one KV head shared by G query
heads**, with its cache sharded across P on-wafer PEs. Inputs are supplied
after positional encoding. Projections, RoPE, MLP, model sampling and
inter-wafer transport are outside this operator's contract. Precision is FP32.

**Configured and host-checked; CSL compilation and device performance remain
unmeasured. The requested >5x improvement is not established.** See the
[readiness report](../../results/reports/2026-10-23-attention-readiness.json).

| Profile suffix | Compute PEs | Slots/PE | Head width | Queries/KV head | Initial/final context |
| --- | ---: | ---: | ---: | ---: | ---: |
| `d64-ctx128` | 4 | 32 | 64 | 4 | 124/128 |
| `d64-boundary` | 4 | 32 | 64 | 4 | 30/34 |
| `d64-ctx1024` | 16 | 64 | 64 | 4 | 1,020/1,024 |
| `d64-ctx4096` | 64 | 64 | 64 | 4 | 4,092/4,096 |
| `d128-ctx512` | 16 | 32 | 128 | 2 | 508/512 |

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

Candidates cannot change layout, data declarations, cache appends, causal
maxima, collective control, stores or timing. Only the local-compute region is
editable. Precision, PE budget, numerical bounds and <=1.10x program-object
memory gate stay fixed. Actual compiled memory/footprint needs SDK inspection;
source buffer estimates do not certify SRAM fit. Work units count QK/PV
multiply/add arithmetic; nonlinear and communication work remains timed.

## Staged hypotheses and >5x target

1. `attention-vector-exp`: vector subtraction and map of exp, with future slots
   masked before exponentiation; QK/PV and all distributed control stay fixed.
2. `attention-polynomial-exp`: stage one plus degree-ten exp(x/16), squared four
   times, on [-16,0], and an SDK-exp fallback outside that range. Host checks
   stress approximation boundaries and all profile/case/seed combinations.

Both are **uncompiled proposals**, not retained improvements. The incremental
campaign uses the ordinary 1.05x gate. A separate target campaign requires
**5.01x versus the original vectorized reference**, at both fresh seeds and
under the memory gate. Family-wide >5x requires every profile to pass; otherwise
report individual confirmed cells. Never multiply unrelated gains or treat CPU
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
