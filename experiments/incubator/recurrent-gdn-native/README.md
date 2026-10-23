# Native-width recurrent head-group staging archive

The canonical contract is now registered in docs/benchmarks/recurrent-gdn-native.md,
with trusted code in benchmarks/drivers/recurrent_gdn_native*.py and kernels/attention/recurrent-gdn-native/. This archive has
**no SDK device qualification or performance result**. The two proposed fixed
profiles preserve native K128/V128 dimensions for one key-head group with three
value heads, for four or eight resident tokens. See [the proposed contract](CONTRACT-DRAFT.md).

`oracle.py` reuses the already fixed FP32 interval arithmetic from the scaled
oracle without changing its bounds. Its native logical shape and token history
are separate contracts. `layout.py` maps complete V-column shards across 48 or
96 PEs, reconstructs every full state, and bit checks all physical immutable
copies plus replicated normalized operands/decay. A complete K dimension lives
on each shard; no state-dot terms or inter-PE reductions are omitted.

The source in `kernel/src/` is a scalar CSL reference, derived from the scaled
reference with R=1 per PE and rectangular grid parameters. `driver.py` is a draft
for its future trusted adapter. Neither has been compiled or simulated yet.
These are independently authored reference implementations, not Cerebras's
production inference kernels.

The pinned upstream fallback has been exercised on all six input families for
both token budgets using native CPU PyTorch 2.8.0. A read-only line trace observes
its actual normalized operands, decay, delta, every updated state, returned BF16
outputs and final FP32 state, without edits to the mathematical function body.
`verify_upstream_inputs.py` regenerates each input in SDK NumPy, compares every
input bit, and replays all saved tensors through the fixed native oracle. These
are unscored correspondence checks; no model or device timing is measured.

The probe scripts use the registered driver directory by default. Before native
registration, pass `--drivers runs/staging/native-gdn-evaluator/benchmarks/drivers`
to both scripts. Each new probe archives its actual script/oracle/error-model
sources alongside the raw tensors. Historical reports stay immutable. Raw
reports are under `runs/maintenance/recurrent-gdn-native-upstream-*`; compact
SHA-bound reports are under `results/reports/2026-10-23/recurrent-gdn-native/`.

The primary native baseline has been strengthened before any device run: it
uses the scaled, freshly confirmed FP32 vector-state implementation. The scalar
prototype here stays unscored. Native fixed-profile baselines and new paired
qualification are still required; the scaled result does not qualify native
widths or different PE budgets.
