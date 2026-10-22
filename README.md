# Cerebras kernel autoresearch

Research and optimize CSL inference kernels using the **Cerebras SDK 2.10.1
fabric simulator**. Start with the official GEMV tutorial, then validate and
optimize public WaferLLM kernels toward Qwen 3.8 workloads. The primary metric
is simulated kernel cycles with a numerical correctness gate.

This route can run without a physical Cerebras system or an Inference API key.
The SDK supplied by the user is installed in ignored `.sdk/sdk-2.10.1`, and
its bundled checksums all passed. A QEMU/Lima x86 Linux VM named `cerebras-sdk`
provides Singularity. The official WSE-3 GEMV tutorial compiled and passed its
numerical check on 2026-10-22. The appliance workflow instead needs access to
an existing cluster.

```sh
python3 tools/kernel_research.py fetch
limactl start --tty=false cerebras-sdk
tools/sdk-run python3 tools/kernel_research.py preflight
tools/sdk-run python3 tools/kernel_research.py smoke
tools/sdk-run python3 tools/kernel_research.py meshgemv --variant baseline
tools/sdk-run python3 tools/kernel_research.py meshgemv --variant bank-alignment
tools/sdk-run python3 tools/kernel_research.py challenge --variant unroll
```

`fetch` checks out exact revisions from `research/kernel_sources.json` in
ignored `.deps/`. `smoke` compiles the official tutorial for WSE-3 and runs its
numerical check in an isolated `runs/` directory. `meshgemv` compiles a small
WaferLLM kernel and runs six fixed numerical cases against an FP64 reference.
The alignment candidate changes memory placement and adds four explicit storage
bytes per PE; precision, matrix dimensions, and PE count stay fixed.
All results, raw logs, source hashes, numerical errors, raw output arrays, and
timestamps are saved under `runs/`. Commands have time and log size limits.

## Autoresearch loop

`challenge` runs a full baseline, a positive-input candidate screen, full
six-case validation for a promising candidate, then a fresh baseline/candidate
pair with the next seed. The default retention gate is **1.05×** on the
upstream maximum simulated cycle span. A retained result applies only to the
tested shape. Run one challenge at a time in this emulated VM.

```sh
# A second shape preserves the Qwen FFN projection aspect ratio at 1/64 scale.
tools/sdk-run python3 tools/kernel_research.py challenge --variant unroll --M 80 --N 272

# Compare completed full-suite runs; screens cannot pass this command.
python3 tools/kernel_research.py compare BASELINE/result.json CANDIDATE/result.json

# Resume from a validated initial pair, then run a fresh confirmation pair.
tools/sdk-run python3 tools/kernel_research.py challenge BASELINE/result.json CANDIDATE/result.json --variant unroll

python3 -m unittest discover -s tools -p 'test_*.py'
```

The current candidate set tests weight/result memory placement and unrolling
the local FMA loop while preserving arithmetic order. The baseline stays fixed.
Adding a hypothesis means adding a separate candidate transformation in
`tools/kernel_research.py`; the runner evaluates explicit transformations, and
an agent adds further kernel hypotheses as candidate edits. See
[EXPERIMENTS.md](EXPERIMENTS.md) for actual results and rejected runs.

On 2026-10-22, the bounded challenge confirmed **264.7 → 241.3 simulated
cycles (8.84% fewer)** for unroll at 64×64, across six numerical cases and a
fresh paired seed. The driver uses one host simulator thread and fresh
processes per case after intermittent shutdown faults with the default five
threads under QEMU. All 25 processes in the confirmation challenge exited
cleanly. This is a small FP16 kernel result; full Qwen TPS remains unmeasured.

The [research assessment](research/KERNEL_RESEARCH.md) records the source audit,
first experiment, Qwen architecture gaps, and validation gates. WaferLLM has
modifiable inference kernels and simulator launchers; its decoder still needs
Qwen-specific implementation and decoder correctness checks. Kernel gains can
be demonstrated in simulation before validating a full model and hardware TPS.

`tools/inference_benchmark.py` remains available as optional hosted-service
telemetry, under the preserved [API protocol](research/API_PROTOCOL.md).
It is not part of the simulator setup or the kernel optimization score.

For another machine, use the [SDK installation guide](https://sdk.cerebras.ai/installation-guide)
and the VM definition in `research/lima-sdk-x86.yaml`. Adjust its workspace
mount if the checkout path differs, install Lima/QEMU and the x86 guest agent,
then create the VM:

```sh
limactl create --name=cerebras-sdk research/lima-sdk-x86.yaml
limactl start --tty=false cerebras-sdk
```

Extract your SDK into `.sdk/sdk-2.10.1` and verify its bundled checksums.
Alternatively set `CEREBRAS_SDK_DIR` to another SDK directory within the mounted
workspace. `research/sdk_install.json` records this installation's exact SIF
hash and versions. SDK binaries and the private download link are not committed.
Stop this VM with `limactl stop cerebras-sdk` when it is not in use.
