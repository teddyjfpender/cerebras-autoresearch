# Kernel autoresearch protocol

Objective: optimize real CSL inference kernels in the SDK 2.10.1 simulator,
progressing toward a validated Qwen 3.8 decoder. Use simulated kernel cycles
as the first score. A hosted API key is not required for kernel work.

1. Read `research/KERNEL_RESEARCH.md`. Fetch pinned upstream sources with
   `python3 tools/kernel_research.py fetch` and run its `preflight` command.
2. This checkout has the SDK in `.sdk/sdk-2.10.1` and a Linux VM named
   `cerebras-sdk`. Run SDK commands through `tools/sdk-run`. The official
   `smoke` passed; reproduce it after any toolchain change.
3. Reproduce the pinned WaferLLM small MeshGEMV simulator baseline. Build an
   independent numerical oracle and adversarial cases before accepting edits.
   `kernels/meshgemv/verify_kernel.py` supplies six cases with a fixed FP16
   forward-error bound against an FP64 oracle. Preserve that contract.
4. Keep baseline sources fixed. Work in a separate candidate copy. Hold matrix
   dimensions, PE grid, precision, and correctness contract fixed when comparing
   implementation changes; record area/layout alternatives separately.
5. Preserve SDK version, architecture, source hash, compile parameters, seeds,
   raw logs, output errors, and simulated cycle spans in every experiment.
6. Retain improvements only after correctness and repeat confirmation. Do not
   use host wall time as WSE execution time or skip work in a timed loop.
   `challenge --variant NAME` runs the bounded loop for one shape, with a
   default 1.05× cycle speedup gate and a fresh paired seed. Screen runs cannot
   count as full validation. Preserve source/toolchain identity across repeats.
   Run one challenge at a time; this host uses CPU emulation. Reject simulator
   shutdown failures even if numerical checks have already passed.
7. Port and validate Qwen's BF16/precision behavior, linear-attention state,
   gated attention, and projection dimensions before any Qwen decoder claim.
   The upstream decoder timing script has no complete output oracle.

State whether a result is a simulated kernel speedup, a projected model result,
or hardware TPS. Local CSL work does not automatically change Cerebras's hosted
service. Missing SDK access blocks execution, not the feasibility of simulator
kernel research. Keep `EXPERIMENTS.md` factual, including failed attempts.

The earlier hosted API benchmark remains separate. If running it, follow
`research/API_PROTOCOL.md`; its concurrency score cannot support kernel claims.
