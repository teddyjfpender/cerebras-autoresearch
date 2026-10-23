# Recurrent gated delta-rule staging

The fixed proposed contract is in `CONTRACT-DRAFT.md`. This isolates the recurrent
operator from QKV/convolution, gate projections and following output projection;
it is not a complete model layer. The pinned primary implementation is recorded
in `reference.json`. No device execution, qualification or performance result yet.

Each of 16 PEs owns one shared key head and its three value heads. All four
resident tokens, normalized operands, decay, delta, complete state history and
BF16 output are checked, including nonzero immutable initial FP32 state.
The FP64 recurrence oracle is cross-checked against an independent transition
matrix formulation. Separate FP32 host simulation and corruption tests exercise
both proposed profiles and all six families. Driver tests cover mixed-word core
transport, raw 48-bit clock wrap, stale cores and shutdown failures.

`stage_kernel.py` writes only this incubator's independent scalar CSL reference.
Live benchmark drivers/manifests are untouched while BF16 SwiGLU runs. Before any
device bringup, register the complete numerical/timing/precision contract and
replay tests on a distinct evaluator, then obtain new full six-case baselines.
