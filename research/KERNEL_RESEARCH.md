# Simulator kernel research assessment — 2026-10-22

**Go for CSL kernel research using the local SDK simulator.** SDK 2.10.1 from
the user-provided archive is installed and verified against its bundled
checksums. The official GEMV tutorial and WaferLLM's small MeshGEMV example
both compiled and ran successfully in a local Linux VM. There is no need for
an Inference API key or a physical wafer to begin.

## Verified execution paths

- [Local SDK](https://sdk.cerebras.ai/installation-guide): a separately downloaded
  SDK 2.10 container provides `cslc`, `cs_python`, and the fabric simulator.
  Apple Silicon uses a Linux VM. On this macOS 15.3.1 host, QEMU 11.1.2,
  Lima 2.2.1, the x86 guest agent, and SingularityCE 4.0.2 are now installed.
  The `cerebras-sdk` VM has 4 virtual CPUs and 8 GiB RAM. Only this workspace
  is mounted writable. `tools/sdk-run` supplies the SDK path inside it.
- [Appliance mode](https://sdk.cerebras.ai/appliance-mode): `simulator=True`
  runs a simulation job on an existing cluster's worker node. It still needs
  cluster configuration and access. The PyPI `cerebras_sdk==2.10.0` wheel was
  inspected: it contains appliance clients, not the local compiler/simulator.
- [Official examples](https://github.com/Cerebras/sdk-examples/tree/rel-sdk-2.10.0):
  pinned at `06ffd5961e8d35b6324e0ebf6595fe4a5a87c2e1`.
- [WaferLLM](https://github.com/MeshInfra/WaferLLM): pinned at
  `fd1c2daae37cd68706c03fc8009887ecee9900f8`. `MeshGEMV`, `MeshGEMM`, `Decode`,
  and `Prefill` expose WSE-3 kernels and simulator scripts targeting SDK 2.10.
  The small MeshGEMV example reproduced its upstream output check and reported
  264.7 maximum simulated cycles, with maximum relative error 0.0013.

## Initial experiment and validation gates

1. Obtain the SDK archive from the official request form, install it in the
   documented Linux environment, and pass the official GEMV tutorial smoke test.
2. Reproduce WaferLLM `MeshGEMV` at `P=4, M=64, N=64, group_num=2`.
   Its launcher performs a warmup, ten repeats, output collection, and timestamp
   collection. Use its reported maximum cycle span as the initial metric;
   audit timestamp alignment before treating it as a calibrated makespan.
   Preserve source hashes, compiler version, parameters, logs, and output checks.
3. Strengthen correctness before accepting optimized kernels. The upstream
   test uses positive random FP16 inputs and permits maximum relative error
   below 0.1. Add signed inputs, zeros, cancellation, and range cases with a
   fixed numerical contract. The included verifier now checks six cases against
   an FP64 reference of exactly quantized FP16 inputs using a forward-error bound.
4. Optimize actual CSL compute and communication at the same PE grid, matrix
   size, and precision; record changes in resident memory. Test DSD access and memory banking,
   reduction grouping, and communication overlap. Retain separate layout/area
   alternatives rather than silently giving a candidate more PEs.
5. Map to Qwen projection shapes only after numerical validation. Small
   aspect-ratio-preserving screens for the FFN can use `M=80,N=272` and its
   transpose on a 4×4 grid. These are scaled kernel problems, not full Qwen.
6. Build a validated small decoder path before estimating end-to-end TPS.
   The upstream `Decode/launch_sim.py` inspected here collects timings but
   does not copy and verify the decoder output. Its timing alone is insufficient.

## Qwen-specific work still required

The initial bounded search is complete. Bank alignment was rejected, while
unroll at 64×64 passed a fresh paired confirmation at 241.3 versus 264.7 cycles
(8.84% fewer). The 80×272 unroll screen stayed below the retention gate.
See `../EXPERIMENTS.md` and `results/2026-10-22.json` for paths and provenance.
Intermittent simulator shutdown failures under QEMU were rejected. The completed
challenge used one host simulator thread and one process per numerical case;
all 25 processes exited cleanly. This validates that challenge run, not general
emulator reliability or a complete decoder.

The official [Qwen config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)
has hidden size 5120, FFN size 17408, BF16 dtype, 48 linear-attention layers,
16 full-attention layers, gated attention output, and one MTP layer. The
target metadata is recorded in `qwen3.8-target.json` beside this document.

WaferLLM's FP16 decoder is not a Qwen 3.8 implementation. Matching it requires
a numerical precision decision, the linear-attention state update, Qwen's
attention dimensions and gates, and a model-level reference check. Changing
the JSON dimensions alone does not supply these operations.

Report simulated cycles and verified kernel speedups first. Any projected
TPS needs explicit coverage and clock assumptions. Hardware TPS and hosted
Cerebras service TPS remain separate validation stages; local simulator
improvements do not automatically deploy into Cerebras's service.
