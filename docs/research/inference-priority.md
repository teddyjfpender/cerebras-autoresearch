# Inference acceleration priorities

The active 2026-10-26 goal is to optimize distributed decode attention, complete
decoder layers and prefill GEMM, and add important inference coverage. The
[campaign plans](../../experiments/campaigns/inference-acceleration/README.md)
add seven GQA profiles, a full 28-to32 context decoder case and two 64-row prefill
projections. These are configuration until SDK execution qualifies each cell.
The existing resident BF16 FFN and native recurrent tracks remain relevant.

Prioritize matched full/fresh gains in the three requested tracks, then complete
shape guards and raw numerical/timestamp replay. Candidate hypotheses include
PR #1 descriptor/vector/bank transfers, one-reciprocal vector softmax inside the
complete decoder, and resident-GEMM descriptor strategies inside cold prefill.
The attention full-sequence coordinator clock is a separate metric from maximum
local cycles in the other tracks. Hardware/model TPS remains unmeasured.

The 2026-10-23 session was stopped at the user's request; its evidence below is preserved.
One frozen native recurrent implementation is freshly confirmed across both
K128/V128 profiles: **2.428316× on T4/48 PEs**, **2.719851× on T8/96 PEs**,
**2.569953× conservative suite geometric mean**. All 48 baseline/candidate case
executions pass independent numerical and saved-core/state/timer replay; all 24
paired cases have bit-identical compared operands, states and outputs.
[Complete suite evidence](../../results/reports/2026-10-23/recurrent-gdn-native/unroll-suite-confirmed.json).

The full 16-key/48-value-head operator remains unqualified. Its planned 20-minute
positive bringup check was cancelled before launch when the user requested an
immediate wrap-up. Earlier failed/cancelled runtime probes remain preserved;
short timeouts do not establish a general simulator deadlock or SDK limitation.
[Full-head closeout](../../results/reports/2026-10-23/recurrent-gdn-full/closeout.json).
Simulator cycles do not establish model latency, hardware TPS or production gain.

For a future Cerebras collaboration, propose a native-size low-batch decode FFN
chain: distributed GEMV, SwiGLU, down projection and the required collectives/data
movement. Ask their inference engineers to select or confirm the target from a
production latency profile, supply a representative baseline/harness, and allow
custom CSL compilation/execution on appliance hardware. Recurrent DeltaNet is
another candidate if profiling supports its priority. Neither proposal starts
new work; current evidence does not rank production bottlenecks.

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

1. Full native recurrent operator: deferred qualification, complete state validation.
2. A complete resident hybrid decode block: add the missing projections,
   convolution and gated normalization/output projection with a full-chain oracle.
3. Speculative state commit/rollback: a new explicit workload/precision/timing
   contract, fixed acceptance traces and all committed states/outputs checked.
4. Model Zoo model/training integration when its backend is available: measure
   actual model/step throughput under fixed quality, batch and workload conditions.

The later items are deferred directions, not configured or executed benchmarks.
The user resumed inference acceleration on 2026-10-26. Current plans follow the
fixed-contract protocol and require fresh matching controls.
Hardware handoff should contain optimized CSL, fixed profile contracts, reference
sources, complete correctness/state evidence, memory/PE costs and matched results.
