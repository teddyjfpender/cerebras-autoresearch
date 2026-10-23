# Benchmark catalog

Manifests describe configured workloads. Actual execution coverage is recorded in
[`results/`](../../results/README.md); configuration alone is not a validation result.

| Profile | Family | Precision | Validation | Primary |
| --- | --- | --- | --- | --- |
| `pipeline.k8-n8` | composed | fp32 | independent_six_case | max_local_cycles |
| `pipeline.k16-n32` | composed | fp32 | independent_six_case | max_local_cycles |
| `pipeline.k64-n32` | composed | fp32 | independent_six_case | max_local_cycles |
| `meshgemm.cold-tiny` | dense-gemm | fp16 | independent_six_case | max_local_cycles |
| `meshgemm.cold-square32` | dense-gemm | fp16 | independent_six_case | max_local_cycles |
| `meshgemm.cold-prefill-ffn-up-scaled` | dense-gemm | fp16 | independent_six_case | max_local_cycles |
| `meshgemm.cold-prefill-ffn-down-scaled` | dense-gemm | fp16 | independent_six_case | max_local_cycles |
| `meshgemm.cold-rectangular` | dense-gemm | fp16 | independent_six_case | max_local_cycles |
| `meshgemv.square64` | dense-gemv | fp16 | independent_six_case | max_local_cycles |
| `meshgemv.ffn-up-scaled` | dense-gemv | fp16 | independent_six_case | max_local_cycles |
| `meshgemv.ffn-down-scaled` | dense-gemv | fp16 | independent_six_case | max_local_cycles |
| `meshgemv.square128` | dense-gemv | fp16 | independent_six_case | max_local_cycles |
| `meshgemv.ffn-up-medium` | dense-gemv | fp16 | independent_six_case | max_local_cycles |
| `meshgemv.ffn-down-medium` | dense-gemv | fp16 | independent_six_case | max_local_cycles |
| `sdk.broadcast-row` | communication | fp32 | upstream_reference | upstream_cycles |
| `sdk.broadcast-column` | communication | fp32 | upstream_reference | upstream_cycles |
| `sdk.residual` | composed | fp32 | upstream_reference | correctness only |
| `sdk.gemm-collectives` | dense-gemm | fp32 | upstream_reference | correctness only |
| `sdk.gemv-collectives` | dense-gemv | fp32 | upstream_reference | correctness only |
| `sdk.fft1d-forward` | fft | complex-fp32 | upstream_reference | correctness only |
| `sdk.fft1d-inverse` | fft | complex-fp32 | upstream_reference | correctness only |
| `sdk.fft2d-forward` | fft | complex-fp16 | upstream_reference | correctness only |
| `sdk.fft2d-inverse` | fft | complex-fp16 | upstream_reference | correctness only |
| `sdk.fft3d` | fft | complex-fp32 | upstream_reference | upstream_cycles |
| `sdk.memcpy-h2d` | host-io | fp32 | diagnostic_only | upstream_cycles |
| `sdk.memcpy-d2h` | host-io | fp32 | diagnostic_only | upstream_cycles |
| `sdk.wide-multiply256` | integer | u16-limbs | upstream_reference | correctness only |
| `sdk.local-matvec` | local-bandwidth | fp32 | upstream_reference | upstream_cycles |
| `sdk.spmv7` | sparse-stencil | fp32 | upstream_reference | upstream_cycles |
| `sdk.game-of-life` | spatial | integer | upstream_reference | correctness only |
| `gated-mlp.h16-f32` | resident-gated-mlp | fp32 | independent_six_case | max_local_cycles |
| `gated-mlp.h32-f64` | resident-gated-mlp | fp32 | independent_six_case | max_local_cycles |
| `decoder.h16-f32-ctx4` | resident-decoder | fp32 | independent_six_case | max_local_cycles |
| `decoder.h16-f32-ctx12` | resident-decoder | fp32 | independent_six_case | max_local_cycles |

| `reusable-gemm.tiny-sequence` | resident-gemm-sequence | fp16 | independent_six_case | max_local_cycles |
| `reusable-gemm.prefill-ffn-up-scaled` | resident-gemm-sequence | fp16 | independent_six_case | max_local_cycles |
| `reusable-gemm.prefill-ffn-down-scaled` | resident-gemm-sequence | fp16 | independent_six_case | max_local_cycles |

## Suites

| Suite | Profiles | Purpose |
| --- | ---: | --- |
| `all` | 37 | Every configured profile; potentially slow under emulation |
| `communication` | 4 | Broadcast and emulated host transfers |
| `dense` | 11 | Dense inference primitives |
| `fusion` | 3 | Independent composed chains; staged/fused suite comparison |
| `gemm` | 5 | Five cold GEMM shapes |
| `reusable-gemm` | 3 | Four resident frames, shared weights, complete output/state oracle |
| `gemv` | 6 | Six GEMV shapes |
| `qualification` | 27 | Representative independent oracles and every SDK profile |
| `regression` | 5 | Fixed dense/composed cross-shape guard set |
| `sdk` | 16 | All pinned SDK reference/diagnostic profiles |
| `prefill-ffn` | 2 | Larger cold FFN up/down GEMM projections |
| `decode-ffn` | 2 | Medium FFN up/down GEMV projections |
| `gated-mlp` | 2 | Complete RMSNorm/SwiGLU/residual at two widths |
| `decoder` | 2 | Stateful causal decoder layer at two initial contexts |
| `inference-blocks` | 8 | All four inference challenge tracks |
| `smoke` | 4 | Quick positive-input screens; no retention |

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
