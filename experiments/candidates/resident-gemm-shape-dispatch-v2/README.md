# Expansion/contraction descriptor dispatch

Keep the previously frozen expansion threshold Nt>=32 and Nt>Kt: use contiguous row accumulation with hoisted weight DSR there, and the reference column map with proven array-form output base resets elsewhere. Preserve every FP16 term and its order, timed full output pack, receive-before-use resets and exact complete frame weight restoration. Test the complete three-shape suite rather than extrapolating the expansion gain.

Reject any six-case output/state/shutdown failure, missing arithmetic term, modified protected controls or memory ratio above1.10x. Suite retention needs full coverage, fresh paired seeds, geometric mean speedup at least1.05x and every fixed worst-cell regression guard; do not label a sub5percent cell individually retained.

## Portable complete-suite validation

Run all four arms under one unchanged evaluator/SDK, preserving the fixed suite:

```sh
tools/sdk-run python3 research.py suite reusable-gemm --seed 20261029 --max-seconds 1800
tools/sdk-run python3 research.py suite reusable-gemm --variant candidate --candidate resident-gemm-shape-dispatch-v2 --seed 20261029 --max-seconds 1800
tools/sdk-run python3 research.py suite reusable-gemm --seed 20261030 --max-seconds 1800
tools/sdk-run python3 research.py suite reusable-gemm --variant candidate --candidate resident-gemm-shape-dispatch-v2 --seed 20261030 --max-seconds 1800
tools/sdk-run python3 research.py confirm-suite INITIAL_BASE INITIAL_CAND FRESH_BASE FRESH_CAND
```

Use each arm's emitted result.json path in the final command. Existing matched
baseline receipts can instead be imported with tools/assemble_suite.py; imports
are metadata receipts and do not count as new SDK executions. This candidate
is awaiting complete-suite qualification; individual component evidence does
not establish all-shape retention.
