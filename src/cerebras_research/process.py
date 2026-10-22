"""Bounded subprocess execution and a single-writer simulator lease."""

import contextlib
import fcntl
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .paths import ROOT


@contextlib.contextmanager
def simulator_lease():
    folder = ROOT / "runs"
    folder.mkdir(exist_ok=True)
    with (folder / ".simulator.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another benchmark owns the simulator lease") from error
        try:
            if shutil.disk_usage(folder).free < 1024**3:
                raise RuntimeError("Less than 1 GiB free; preserve artifacts and free disk space")
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_logged(args, cwd, log_path, timeout=600):
    """Kill the isolated process group on timeout, cancellation, or log flooding."""
    started = time.monotonic()
    with Path(log_path).open("w") as output:
        process = subprocess.Popen(
            args, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    raise RuntimeError(f"Step exceeded {timeout:g}s: {log_path}")
                total = 0
                # The simulator creates/removes temporary subdirectories while
                # running. os.walk tolerates a directory disappearing before
                # descent; Path.rglob on Python 3.11 can abort the experiment.
                for parent, _, files in os.walk(cwd):
                    for name in files:
                        if not name.endswith(".log"):
                            continue
                        path = Path(parent) / name
                        try:
                            size = path.stat().st_size
                        except FileNotFoundError:
                            continue
                        total += size
                        if size > 64 * 1024**2:
                            raise RuntimeError(f"Log exceeded 64 MiB: {path}")
                if total > 256 * 1024**2:
                    raise RuntimeError(f"Run logs exceeded 256 MiB: {cwd}")
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise RuntimeError(f"Step failed ({process.returncode}): {log_path}")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    return {
        "command": args,
        "wall_seconds": time.monotonic() - started,
        "log": str(Path(log_path).relative_to(cwd)),
        "exit_code": process.returncode,
    }
