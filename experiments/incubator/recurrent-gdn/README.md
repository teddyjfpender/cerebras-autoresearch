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

## Upstream function correspondence checked

The [unscored CPU reference](../../../results/reports/2026-10-23/recurrent-gdn/upstream-reference.json)
executes the pinned recurrent fallback math body unchanged, with dispatch
wrappers removed. A read-only Python trace captures its actual normalization,
exp, delta and every token state. All 12 fixtures (six families, both widths)
passed the fixed independent local and full-chain bounds. Full raw tensors are
hashed and retained. `verify_upstream_inputs.py` also
[replayed those tensors in SDK NumPy](../../../results/reports/2026-10-23/recurrent-gdn/upstream-sdk-replay.json)
and verified bit-identical fixture inputs across both environments. This is
upstream operator correspondence, not a model run or device performance result.

To reproduce on this macOS host, obtain the pinned file named in `reference.json`
at `runs/references/qwen3_5/modeling_qwen3_5.py`, then run:

```sh
uv run --no-project --python 3.12 --with torch==2.8.0 --with numpy==2.2.6 \
  experiments/incubator/recurrent-gdn/upstream_reference_probe.py \
  --output runs/maintenance/my-upstream-reference.json
tools/sdk-run cs_python experiments/incubator/recurrent-gdn/verify_upstream_inputs.py \
  --reference runs/maintenance/my-upstream-reference.json \
  --output runs/maintenance/my-upstream-sdk-replay.json
```

Each output and its raw directory must be new; the tools refuse overwrites.
The reference uses CPU only and installs no Torch dependency into the SDK image.
