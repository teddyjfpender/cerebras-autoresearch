# gemm-row-major-accum

Experimental, parameterized CSL candidate; no measured performance claim yet.

Local M=4 wastes dispatch on short row vectors. Accumulate along contiguous N-wide output vectors in a row-major scratch tile, then transpose every output back to the fixed ABI. Extra tile memory and transpose stores must pay for themselves.

Reject any numerical/exit failure, changed K order/FP16 precision or PE budget, speedup below 1.05x, memory growth above 1.10x or failed fresh paired confirmation.

Communication and protected timestamp/repetition controls remain unchanged.
Every output is returned in the original column-major tile ABI. Candidate
workspaces are measured by the four-track campaign; screens cannot qualify.

Source is derived from [WaferLLM](https://github.com/MeshInfra/WaferLLM), pinned
at `fd1c2daae37cd68706c03fc8009887ecee9900f8`; see `LICENSE.upstream`.
