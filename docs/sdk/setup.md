# SDK and execution environment

The local installation uses SDK 2.10.1 and WSE-3 compilation. The selected host
route is Lima VZ, Ubuntu 24.04 ARM64, Rosetta for the unchanged x86 SDK image,
and SingularityCE 4.1.1. `config/sdk-platform.json` selects and identifies this
execution profile; `config/platforms/lima-rosetta.yaml` pins its guest image and
container package. This is a repository-tested host route. Consult the official
SDK guide for Cerebras-supported host configurations.

Exact source revisions are in `config/sources.lock.json`. The SDK SIF digest and
original QEMU installation evidence remain in `config/sdk-install.json`.
New receipts record the actual controller/OS/container version separately from
that historical installation evidence. VM, controller and package mismatches
are rejected before a scored run starts.

The archive's bundled checksums were verified during installation. SDK binaries
and its private download URL are excluded from Git. Obtain your own archive
through the [official SDK installation guide](https://sdk.cerebras.ai/installation-guide).

## Existing checkout

```sh
limactl start --tty=false cerebras-sdk-rosetta
tools/sdk-run python3 research.py doctor
tools/sdk-run python3 research.py doctor --verify-sdk
make sdk-test
make smoke
```

Full SIF hashing can take minutes under QEMU. The fast doctor checks tool
availability and executes compiler help; it does not rehash the image. Repeat
digest verification and numerical smoke after a toolchain change.

`tools/sdk-run` reads the selected VM from `config/sdk-platform.json`, supplies
the SDK path, and exports the VM identity to the controller. The current drivers
request one simulator thread and suppress GUI traces. Per-case process isolation,
timeouts, log caps and clean-exit requirements preserve failures. Keep SDK Python
helpers inside the mounted working directory; the runner copies the complete
trusted helper set into each isolated work directory.

The six-case same-binary host-route diagnostic produced byte-identical numerical
and state arrays. Device-case wall time was 16.8 seconds on Rosetta versus 204.4
seconds on QEMU; twelve additional Rosetta cases had identical elapsed cycles.
Cross-host counts differed by up to four cycles, so the old timing-equivalence
gate failed and its evidence is preserved. Rosetta is a distinct evaluator
track requiring fresh baselines and fresh candidate pairs. The launcher,
selected platform and SDK-install manifest are included in new source capsules.
Host wall time is iteration cost; it is not a kernel-cycle or TPS improvement.

## Another machine

1. For this host route, use an ARM Mac with Lima VZ and Rosetta for Linux;
   see [Lima's multi-architecture guide](https://lima-vm.io/docs/config/multi-arch/).
   The original QEMU route remains in `config/platforms/lima-x86.yaml`.
2. Adjust the workspace mount in `config/platforms/lima-rosetta.yaml` if the clone
   path differs; the configuration mounts only this workspace writable.
3. Create/start the VM, then extract the verified archive within the workspace.

```sh
limactl create --name=cerebras-sdk-rosetta config/platforms/lima-rosetta.yaml
limactl start --tty=false cerebras-sdk-rosetta
```

The default SDK directory is `.sdk/sdk-2.10.1`. `CEREBRAS_SDK_DIR` can select
another directory inside the mounted workspace. On native Linux, put its
wrappers on PATH and run `python3 research.py ...` directly. A different host
needs matching execution metadata and fresh baselines; do not retain this
checkout's VM/OS labels. Update the selected platform or native installation
manifest when changing the execution route.

`CEREBRAS_SDK_VM` overrides routing for diagnostics. Scored runs still require
the VM and controller to match the selected profile; an override cannot silently
reuse baselines from another route.

The compiler wrapper's preflight command is `cslc --help`. Use the SIF build
identity/digest for provenance. Inspect compiled memory with `cs_readelf --msize`.
The runner collects program-object observations automatically.

`limactl stop cerebras-sdk-rosetta` releases the VM's resources when idle. Host wall time
describes experiment iteration cost; it is never the device performance score.
Appliance-mode simulation needs cluster access and is a separate execution route.
