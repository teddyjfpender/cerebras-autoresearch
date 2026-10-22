# GEMV unroll example

This editable CSL overlay derives from MeshInfra/WaferLLM at the exact revision
in `candidate.json`. Its source license is provided in
[`third_party/licenses/waferllm-LICENSE`](../../../third_party/licenses/waferllm-LICENSE).
The modified main kernel carries a change notice.

The local map is expanded into 16 FMA instructions for `meshgemv.square64`.
This candidate is shape-specific. Baseline precision, grid, arithmetic order,
timers, and repeat counts remain fixed. The historical V1 gain does not promote
this candidate under V2's measurement contract.

```sh
tools/sdk-run python3 research.py run meshgemv.square64 \
  --variant candidate --candidate gemv-unroll --screen
tools/sdk-run python3 research.py challenge meshgemv.square64 \
  --variant candidate --candidate gemv-unroll
```

Edit the hypothesis and CSL for a new attempt, or create a new workspace to keep
this example intact. All candidate sources are copied into isolated run records.
