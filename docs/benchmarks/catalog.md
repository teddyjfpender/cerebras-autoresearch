# Benchmark catalog

Manifests describe configured workloads. Actual execution coverage is recorded in
[`results/`](../../results/README.md); configuration alone is not a validation result.

<!-- catalog:profiles:start -->
| Profile | Family | Precision | Validation | Primary |
| --- | --- | --- | --- | --- |
| `decoder.h16-f32-ctx12` | resident-decoder | fp32 | independent_six_case | max_local_cycles |
| `decoder.h16-f32-ctx4` | resident-decoder | fp32 | independent_six_case | max_local_cycles |
| `gated-mlp.h16-f32` | resident-gated-mlp | fp32 | independent_six_case | max_local_cycles |
| `gated-mlp.h32-f64` | resident-gated-mlp | fp32 | independent_six_case | max_local_cycles |
| `meshgemm.cold-prefill-ffn-down-scaled` | dense-gemm | fp16 | independent_six_case | max_local_cycles |
| `meshgemm.cold-prefill-ffn-up-scaled` | dense-gemm | fp16 | independent_six_case | max_local_cycles |
| `meshgemm.cold-rectangular` | dense-gemm | fp16 | independent_six_case | max_local_cycles |
| `meshgemm.cold-square32` | dense-gemm | fp16 | independent_six_case | max_local_cycles |
| `meshgemm.cold-tiny` | dense-gemm | fp16 | independent_six_case | max_local_cycles |
| `meshgemv.ffn-down-medium` | dense-gemv | fp16 | independent_six_case | max_local_cycles |
| `meshgemv.ffn-down-scaled` | dense-gemv | fp16 | independent_six_case | max_local_cycles |
| `meshgemv.ffn-up-medium` | dense-gemv | fp16 | independent_six_case | max_local_cycles |
| `meshgemv.ffn-up-scaled` | dense-gemv | fp16 | independent_six_case | max_local_cycles |
| `meshgemv.square128` | dense-gemv | fp16 | independent_six_case | max_local_cycles |
| `meshgemv.square64` | dense-gemv | fp16 | independent_six_case | max_local_cycles |
| `pipeline.k16-n32` | composed | fp32 | independent_six_case | max_local_cycles |
| `pipeline.k64-n32` | composed | fp32 | independent_six_case | max_local_cycles |
| `pipeline.k8-n8` | composed | fp32 | independent_six_case | max_local_cycles |
| `recurrent-gdn-native.k128-v128-t4` | recurrent-gated-delta-native-head-group | bf16-fp32-state | independent_six_case | max_local_cycles |
| `recurrent-gdn-native.k128-v128-t8` | recurrent-gated-delta-native-head-group | bf16-fp32-state | independent_six_case | max_local_cycles |
| `recurrent-gdn.k16-v16` | recurrent-gated-delta-rule | bf16-fp32-state | independent_six_case | max_local_cycles |
| `recurrent-gdn.k32-v16` | recurrent-gated-delta-rule | bf16-fp32-state | independent_six_case | max_local_cycles |
| `resident-swiglu-bf16.batch32-h64-f192` | distributed-resident-swiglu | bf16-fp32-accum | independent_six_case | max_local_cycles |
| `resident-swiglu-bf16.batch4-ffn` | distributed-resident-swiglu | bf16-fp32-accum | independent_six_case | max_local_cycles |
| `resident-swiglu-bf16.ffn-scaled` | distributed-resident-swiglu | bf16-fp32-accum | independent_six_case | max_local_cycles |
| `resident-swiglu-bf16.small` | distributed-resident-swiglu | bf16-fp32-accum | independent_six_case | max_local_cycles |
| `resident-swiglu.ffn-scaled` | distributed-resident-swiglu | fp16-fp32 | independent_six_case | max_local_cycles |
| `resident-swiglu.small` | distributed-resident-swiglu | fp16-fp32 | independent_six_case | max_local_cycles |
| `reusable-gemm.prefill-ffn-down-scaled` | resident-gemm-sequence | fp16 | independent_six_case | max_local_cycles |
| `reusable-gemm.prefill-ffn-up-scaled` | resident-gemm-sequence | fp16 | independent_six_case | max_local_cycles |
| `reusable-gemm.tiny-sequence` | resident-gemm-sequence | fp16 | independent_six_case | max_local_cycles |
| `sdk.broadcast-column` | communication | fp32 | upstream_reference | upstream_cycles |
| `sdk.broadcast-row` | communication | fp32 | upstream_reference | upstream_cycles |
| `sdk.fft1d-forward` | fft | complex-fp32 | upstream_reference | correctness only |
| `sdk.fft1d-inverse` | fft | complex-fp32 | upstream_reference | correctness only |
| `sdk.fft2d-forward` | fft | complex-fp16 | upstream_reference | correctness only |
| `sdk.fft2d-inverse` | fft | complex-fp16 | upstream_reference | correctness only |
| `sdk.fft3d` | fft | complex-fp32 | upstream_reference | upstream_cycles |
| `sdk.game-of-life` | spatial | integer | upstream_reference | correctness only |
| `sdk.gemm-collectives` | dense-gemm | fp32 | upstream_reference | correctness only |
| `sdk.gemv-collectives` | dense-gemv | fp32 | upstream_reference | correctness only |
| `sdk.local-matvec` | local-bandwidth | fp32 | upstream_reference | upstream_cycles |
| `sdk.memcpy-d2h` | host-io | fp32 | diagnostic_only | upstream_cycles |
| `sdk.memcpy-h2d` | host-io | fp32 | diagnostic_only | upstream_cycles |
| `sdk.residual` | composed | fp32 | upstream_reference | correctness only |
| `sdk.spmv7` | sparse-stencil | fp32 | upstream_reference | upstream_cycles |
| `sdk.wide-multiply256` | integer | u16-limbs | upstream_reference | correctness only |
<!-- catalog:profiles:end -->

## Suites

<!-- catalog:suites:start -->
| Suite | Profiles | Purpose |
| --- | ---: | --- |
| `all` | 47 | Current aggregate of every configured profile; each receipt freezes its actual membership. Mixed validation tiers cannot receive automatic promotion. |
| `communication` | 4 | Broadcast and emulated host transfers |
| `decode-ffn` | 2 | Medium FFN up/down GEMV projections |
| `decoder` | 2 | Stateful causal decoder layer at two initial contexts |
| `dense` | 11 | Dense inference primitives |
| `fusion` | 3 | Independent composed chains; staged/fused suite comparison |
| `gated-mlp` | 2 | Complete RMSNorm/SwiGLU/residual at two widths |
| `gemm` | 5 | Five cold GEMM shapes |
| `gemv` | 6 | Six GEMV shapes |
| `inference-blocks` | 8 | All four inference challenge tracks |
| `prefill-ffn` | 2 | Larger cold FFN up/down GEMM projections |
| `qualification` | 27 | Representative independent oracles and every SDK profile |
| `recurrent-gdn-native` | 2 | Complete native K128/V128 one-group recurrent operators, separate fixed T4/48PE and T8/96PE contracts. |
| `recurrent-gdn` | 2 | Complete resident recurrent gated delta-rule state updates, two state widths. |
| `regression` | 5 | Fixed dense/composed cross-shape guard set |
| `resident-swiglu-bf16-generalization` | 4 | Fixed original two-cell suite plus predeclared smaller-batch and different-aspect-ratio workloads; independent full/fresh coverage and per-cell memory/worst-speed guards. |
| `resident-swiglu-bf16` | 2 | Distinct complete BF16-storage/FP32-accumulator resident FFN contracts. |
| `resident-swiglu` | 2 | Complete distributed four-batch resident SwiGLU chains, scaled FP16/FP32 precision. |
| `reusable-gemm` | 3 | Four distinct resident GEMMs with full output and canonical/working weight state qualification. |
| `sdk` | 16 | All pinned SDK reference/diagnostic profiles |
| `smoke` | 4 | Quick positive-input screens; no retention |
<!-- catalog:suites:end -->

## Validation tiers

- **Independent:** six families (positive, signed, zero, cancellation, one-hot, mixed scales), fixed numerical contracts against FP64 of quantized inputs (analytic forward-error bounds for linear chains; fixed end-to-end tolerances for nonlinear resident blocks), full output collection. Automatic promotion supported.
- **Upstream reference:** pinned original CPU oracle and assertions, with raw D2H captures. Useful baselines and regressions; independent adversarial promotion is not supported yet.
- **Diagnostic:** H2D/D2H timing only; no numerical-kernel performance claim.

The FFN profiles preserve the Qwen projection aspect ratio at reduced dimensions.
They are synthetic tensor problems, not full Qwen layers or decoder benchmarks.

## Adding coverage

Add profile data in a nested family manifest, register it in `manifests/catalog.json`,
and include appropriate suite members. Read the [architecture guide](../development/architecture.md).
Use `python3 research.py list --suite NAME --json` to inspect exact resolved commands,
timing protocols, dimensions, PE budgets, and units.

See [complete inference block contracts](inference-blocks.md) for equations,
state coverage, precision, timing, work counts and qualification commands.

See [resident reusable GEMM](reusable-gemm.md) for its distinct sequence/state contract.

The [distributed resident SwiGLU contract](resident-swiglu.md) defines two newly
registered cells. Device qualification is outstanding until full receipts are
linked in results. These are scaled FP16/FP32 FFNs with no RMSNorm or attention.

The [BF16-storage/FP32-accumulator chain](resident-swiglu-bf16.md) is a distinct
precision contract. Native format/FMA probes passed; complete-chain device
qualification remains outstanding until actual receipts are linked.

Two additional BF16 SwiGLU workloads are registered but not yet device-qualified:
`resident-swiglu-bf16.batch4-ffn` and `.batch32-h64-f192`. They share the fixed
BF16/FP32 contract and add a distinct four-cell generalization suite. Current
configuration including recurrent GDN is 47 profiles, 16 families, 21 suites; 31 independent-oracle profiles.

## Resident recurrent gated delta rule

Two new profiles (`recurrent-gdn.k16-v16`, `.k32-v16`) add four resident tokens
with 16 key heads and 48 value heads on 16 PEs, BF16 Q/K/V/output and FP32 state.
Both are independently-oracled configurations; device qualification is pending.
The [contract](recurrent-gdn.md) covers all normalized operands, decay, delta,
full token-state histories, immutable inputs and final live-state bits. Current
configuration is 47 profiles, 16 families, 21 suites; 31 independent-oracle
profiles. Historical execution counts remain separate.
