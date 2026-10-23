# Current inference research priority

The active target is the **full native Qwen recurrent operator**:16 key heads,
48 value heads, K128/V128, BF16 Q/K/V/output and FP32 state, with every requested
state update/readout validated. The proposed full-head workload is a fixed
two-token decode pair on768 PEs. Existing one-group T4/T8 profiles establish
longer recurrence coverage. These are distinct contracts with new baselines.

The priority is to finish native confirmations, freeze a transferable candidate,
qualify the full-head reference and confirm a full-head improvement with fresh
paired seeds and complete raw numerical/core/timer replay. All terminal V2
attempts/decisions remain in the append-only TSV. Profile-specific simulator
cycles do not establish whole-model latency, hardware TPS or production impact.
No end-to-end hardware profile currently proves this operator is the model's
largest latency bottleneck.

## What the external projects contribute

[DeepInfer at9bd9cf2](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9)
provides a useful inference-system reference: kernel/runtime integration,
asynchronous scheduling, speculative verification and reproducibility. Its
[sweep driver](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/examples/sweep/run_sweep.sh)
examines latency/throughput operating points. Transfer design ideas and validation
requirements to CSL; this is not a Cerebras backend port of that GPU/NPU engine.
For future speculative work, account separately for proposed, verified and
committed steps, acceptance traces, rollback, memory and draft-generation cost.

[Model Zoo atd76e028](https://github.com/Cerebras/modelzoo/tree/d76e028d6a5f784954b1329b3ae44721ca8da36a)
provides concrete model/layer/configuration references for later training/model
challenges. Its [requirements](https://github.com/Cerebras/modelzoo/blob/d76e028d6a5f784954b1329b3ae44721ca8da36a/requirements.txt)
pin the separate Cerebras PyTorch package. The installed CSL SDK does not provide
`cerebras.pytorch`; no Model Zoo training execution or throughput is claimed.
Pinned RMSNorm, feed-forward and activation sources are reference targets only.
Match their selected configuration and precision boundaries before claiming
correspondence with an existing CSL contract. Training work also needs backward,
gradient and loss validation plus the actual training backend for speed claims.

## Focused follow-up gates

1. Full native recurrent operator: current priority, complete state validation.
2. A complete resident hybrid decode block: add the missing projections,
   convolution and gated normalization/output projection with a full-chain oracle.
3. Speculative state commit/rollback: a new explicit workload/precision/timing
   contract, fixed acceptance traces and all committed states/outputs checked.
4. Model Zoo model/training integration when its backend is available: measure
   actual model/step throughput under fixed quality, batch and workload conditions.

The later items are deferred directions, not configured or executed benchmarks.
Broader model coverage and serving features do not interrupt the current target.
Hardware handoff should contain optimized CSL, fixed profile contracts, reference
sources, complete correctness/state evidence, memory/PE costs and matched results.
