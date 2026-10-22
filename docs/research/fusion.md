# From kernels to resident programs

The useful hypothesis is that combining operations can remove intermediate
storage, redundant work, synchronization, or scheduling gaps. Measure the full
chain and its resource costs: fusion is not automatically a win.

[Mirage Persistent Kernel](https://github.com/mirage-project/mirage) compiles GPU
inference into persistent megakernels. Its GPU execution mechanisms motivate
questions about composition; they are not a Cerebras backend. Cerebras CSL
programs expose spatial layout, routing, tasks, and resident modules, so the
benefits must be demonstrated on those mechanisms. Public
[WaferLLM](https://github.com/MeshInfra/WaferLLM) supplies relevant distributed
GEMV/GEMM and inference source to study.

## Runnable first experiment

`pipeline.k8-n8`, `pipeline.k16-n32`, and `pipeline.k64-n32` compute a complete
FP32 matvec → bias → ReLU → residual chain on one PE. Both staged and fused
variants use one host launch. The staged version stores an intermediate vector;
the fused version applies the epilogue per output row.

There are five independent input frames: one warmup and four timed frames.
Every output from every frame is checked against an FP64 reference of the
quantized inputs. The timer encloses the full resident chain, including output
stores. This prevents a final-output-only check from hiding skipped frame work.

This is a small composed-program benchmark. It does not claim to implement
attention, a transformer layer, persistent multi-token decoding, or Qwen.

## Research sequence

| Stage | Correctness gate | Performance question |
| --- | --- | --- |
| Reusable GEMM state | Validate every iteration after different inputs and repeated launches | Restore valid warm repetition before throughput claims |
| Linear epilogues | Full output arrays and fixed precision bound | Reuse registers/intermediates without more PEs |
| RMSNorm / gated MLP | Independent nonlinear reference, range and cancellation cases | Reduce resident passes and synchronization |
| Attention / KV state | Multi-step state oracle, held-out sequence lengths | Overlap communication while preserving state semantics |
| Layer / decoder | Complete layer/token-state outputs against a model reference | Measure full critical path and resource tradeoffs |
| Hardware | Calibrated clocks, actual wafer footprint and I/O | Validate TPS and energy on the target deployment |

The optional Qwen target metadata is in `config/targets/qwen3.8-27b.json`. Its
BF16 and hybrid attention behavior require explicit implementation and validation.
Changing dimension constants in a generic FP16 decoder does not establish that
model's behavior or speed. Keep model claims separate from these kernel results.
