# gemm-vector-columns

Experimental, parameterized CSL candidate; no measured performance claim yet.

Short local M tiles underutilize row-vector FMA dispatch. Accumulate across all N output columns with strided output descriptors and contiguous weight vectors, preserving communication, FP16 precision and K order. Descriptor stride/bank conflicts may offset the dispatch saving.

Reject any numerical or shutdown failure, changed timed work/PE budget, less than 1.05x matched cycle speedup, more than 1.10x program-object memory, or failed fresh paired confirmation.

Communication and protected timestamp/repetition controls remain unchanged.
Every output is returned in the original column-major tile ABI. Candidate
workspaces are measured by the four-track campaign; screens cannot qualify.

Source is derived from [WaferLLM](https://github.com/MeshInfra/WaferLLM), pinned
at `fd1c2daae37cd68706c03fc8009887ecee9900f8`; see `LICENSE.upstream`.
