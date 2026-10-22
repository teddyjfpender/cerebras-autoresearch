# GEMV unroll example

This editable CSL overlay derives from MeshInfra/WaferLLM at the exact revision
in `candidate.json`. Its source license is provided in
[`third_party/licenses/waferllm-LICENSE`](../../../third_party/licenses/waferllm-LICENSE).
The modified main kernel carries a change notice.

The local map is expanded into 16 FMA instructions for `meshgemv.square64`.
This candidate is shape-specific. Baseline precision, grid, arithmetic order,
timers, and repeat counts remain fixed. The historical V1 gain does not promote
this candidate under V2's measurement contract.

The [dogfood campaign](../../../docs/experiments/2026-10-22-dogfood.md) subsequently
confirmed this workspace under V2 with matched baselines and fresh seeds.

```sh
tools/sdk-run python3 research.py run meshgemv.square64 \
  --variant candidate --candidate gemv-unroll --screen
tools/sdk-run python3 research.py challenge meshgemv.square64 \
  --variant candidate --candidate gemv-unroll
```

Create a new workspace for another hypothesis; keep this measured example
intact. Frozen campaigns reject source or metadata edits. All candidate sources
are copied into isolated run records.
