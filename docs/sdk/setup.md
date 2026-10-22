# SDK and execution environment

The local installation uses SDK 2.10.1, WSE-3 compilation, Lima/QEMU x86_64,
Ubuntu 22.04, and SingularityCE 4.0.2. Exact source revisions are in
`config/sources.lock.json`; the SIF digest and original installation evidence
are in `config/sdk-install.json`.

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
