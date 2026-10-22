# CSL bringup findings

These findings come from this repository's SDK 2.10.1 runs. Preserve failed
receipts and change a new candidate when a campaign has frozen the old one.
Passing host-side overlay checks does not establish CSL compilation.

## Exported pointer binding mutability

The first gated-MLP compile failed with `exported symbol mutability mismatch`.
The layout declared `@export_name("Y", [*]f32, false)`, while the PE declared
the exported pointer binding with `var`. The qualified implementation uses:

```csl
var Y: [T*H]f32;
const ptr_Y: [*]f32 = &Y;
```

The output array remains writable by the kernel. Match the exported binding
to the layout declaration. The failure and subsequent successful screen are
preserved in the [bringup report](../../results/reports/2026-10-22-inference-blocks-bringup.json).

## Single-element versus many-item pointers

The next MLP compile rejected `rmsnorm(&X[frame*H], &SCALE1)` because the
function accepts `[*]f32`, whereas the element address has type `*f32`.
The qualified code explicitly casts the address:

```csl
rmsnorm(@ptrcast([*]f32, &X[frame*H]), &SCALE1);
```

The caller must still prove the complete H-element range is in bounds.
This correction is present in the fully qualified resident kernels; both
the pointer failure and later screen appear in the bringup report above.

A related GEMM candidate failed at
`@set_dsd_base_addr(row_input_dsd, &ptr_X_send[row])` with
`base address of a tensor expression must be an array`.
The separate `gemm-row-major-map-k-base` revision instead sets a many-item
base, then applies the row offset in element units:

```csl
var input = @set_dsd_base_addr(row_input_dsd, ptr_X_send);
input = @increment_dsd_offset(input, row, f16);
```

That revision is awaiting its own SDK test. The original failed receipt is
`runs/v2/20261022T214935Z-meshgemm.cold-prefill-ffn-up-scaled-8fe31c63/result.json`.
Keep compiler failures separate from numerical failures and performance screens.

## Repeated execution needs observable state

The pinned GEMM's warmed sequence failed the independent output check even
though its cold invocation passed. Check every output when investigating
reuse; checking an earlier one-shot result does not validate later repetitions.
The [state investigation](../research/gemm-state.md) preserves the failure and
the separate cold timing contract. Reusable execution needs its own workload,
state oracle and baselines.

## Changing a frozen experiment

Create a new candidate directory for a source correction. Do not edit CSL or
candidate metadata referenced by a live campaign. Trusted driver, oracle and
core evaluator changes need a new evaluator track and new matched baselines.
Keep timing/control protection enabled while debugging. See the
[campaign workflow](../experiments/campaigns.md) for resumption and recovery.
