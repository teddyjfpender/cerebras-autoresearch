"""Stable execution-platform identity for fresh, comparable SDK measurements."""

import json
import os
import platform
import subprocess

from .paths import digest


def execution_sdk(root):
    manifest = root / "config/sdk-install.json"
    sdk = json.loads(manifest.read_text())
    selected = root / "config/sdk-platform.json"
    if not selected.exists():
        return sdk
    profile = json.loads(selected.read_text())
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported SDK execution-platform schema")
    actual_vm = os.environ.get("CEREBRAS_SDK_VM", profile["lima_instance"])
    if actual_vm != profile["lima_instance"]:
        raise ValueError("SDK VM differs from the selected execution platform")
    machine, hostname = platform.machine(), platform.node()
    if machine != profile["controller_machine"] or hostname != profile["hostname"]:
        raise ValueError("SDK controller does not match the selected execution platform")
    runtime = subprocess.check_output(["singularity", "--version"], text=True, timeout=10).strip()
    if runtime != profile["container_runtime"]:
        raise ValueError("SDK container runtime differs from the selected platform")
    package, version = profile["container_package"].split("=", 1)
    installed = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${Version}", package], text=True, timeout=10
    ).strip()
    if installed != version:
        raise ValueError("SDK container package differs from the selected platform")
    os_name = platform.freedesktop_os_release().get("PRETTY_NAME", platform.system())
    sdk.update(
        lima_instance=actual_vm,
        guest_os=os_name + " " + machine,
        guest_container_runtime=runtime,
        execution_platform=profile,
        execution_platform_sha256=digest(selected),
        installation_manifest_sha256=digest(manifest),
        controller={
            "machine": machine,
            "hostname": hostname,
            "kernel": platform.release(),
            "python": platform.python_version(),
        },
    )
    if profile["virtualization"] != "qemu":
        sdk.pop("host_qemu_version", None)
    sdk["installation_evidence"] = {
        key: sdk.pop(key) for key in ("example_compatibility_status", "smoke_run") if key in sdk
    }
    return sdk
