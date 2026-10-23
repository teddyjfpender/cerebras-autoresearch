# SDK and execution environment

The pinned installation uses SDK 2.10.1 and WSE-3 compilation. Exact source
revisions are in `config/sources.lock.json`; the SIF digest and current
execution route are in `config/sdk-install.json`. Historical receipts embed
their own environment manifests, including the original Lima installation.
The manifest's qualification note records the bootstrap check at its creation;
later benchmark qualification is recorded in `results/reports/`.

The archive's bundled checksums were verified during installation. SDK binaries
and its private download URL are excluded from Git. Obtain your own archive
through the [official SDK installation guide](https://sdk.cerebras.ai/installation-guide).

## Existing checkout

```sh
limactl start --tty=false cerebras-sdk
tools/sdk-run python3 research.py doctor
tools/sdk-run python3 research.py doctor --verify-sdk
make sdk-test
make smoke
```

Full SIF hashing can take minutes under emulation. The fast doctor checks tool
availability and executes compiler help; it does not rehash the image. Repeat
digest verification and numerical smoke after a toolchain change.

`tools/sdk-run` supplies the SDK path inside the VM. The current drivers request
one simulator thread and suppress GUI traces. Five-thread shutdown was
intermittently unstable under QEMU during bootstrap. Per-case process isolation,
timeouts, log caps, and clean-exit requirements preserve failures rather than
accepting partial results. This does not establish general emulator reliability.

## Another machine

1. Install Lima/QEMU and the matching x86 guest agent, or use a native x86 Linux
   SDK environment. See the official guide for supported host alternatives.
2. Adjust the workspace mount in `config/platforms/lima-x86.yaml` if the clone
   path differs; the configuration mounts only this workspace writable.
3. Create/start the VM, then extract the verified archive within the workspace.

```sh
limactl create --name=cerebras-sdk config/platforms/lima-x86.yaml
limactl start --tty=false cerebras-sdk
```

The default SDK directory is `.sdk/sdk-2.10.1`. `CEREBRAS_SDK_DIR` can select
another directory inside the mounted workspace. On native Linux, put its
wrappers on PATH and run `python3 research.py ...` directly. Update the SDK
manifest when changing toolchains and create fresh baselines.

The compiler wrapper's preflight command is `cslc --help`. Use the SIF build
identity/digest for provenance. Inspect compiled memory with `cs_readelf --msize`.
The runner collects program-object observations automatically.

`limactl stop cerebras-sdk` releases the VM's resources when idle. QEMU wall time
describes experiment iteration cost; it is never the device performance score.
Appliance-mode simulation needs cluster access and is a separate execution route.

## Unpacked image on native Linux

The October 23 measurements use the verified SDK image's compiler, Python,
libraries and simulator directly on x86 Linux. This is a reproduction route
for this workspace, not a statement of vendor support. It needs no container
privileges, namespace changes, network service or SDK API key. The standard
Singularity route remains available through the original SDK wrappers.

Extract the licensed archive with `tar --no-same-owner`, then verify every
entry in its bundled `sha256sum.txt`. Extract the SIF's SquashFS partition
with `unsquashfs`; this exact image has its filesystem at byte 40960. Keep the
image and extracted binaries unchanged. This installation used unsquashfs
4.6.1. Allow about 11 GB for the extracted root plus the image and archive.

The SDK contains absolute interpreter/toolchain paths. In an otherwise empty
execution environment, `/cb`, `/cbcore` and `/python` must resolve to the
corresponding directories in the extracted root. Do not replace existing
system directories. The wrapper checks those paths and loads the image's
own `.singularity.d/env/10-docker2singularity.sh`.

```sh
export CEREBRAS_SDK_DIR=/path/to/extracted-archive
export CEREBRAS_SDK_ROOTFS=/path/to/unpacked-sif-root
tools/sdk-run python3 research.py doctor --verify-sdk
make sdk-test
```

`tools/sdk-run` selects the unpacked entrypoints only when
`CEREBRAS_SDK_ROOTFS` is explicitly set on Linux. Their source is captured in
each new evaluator capsule. SDK paths and licensed binaries stay outside Git.
