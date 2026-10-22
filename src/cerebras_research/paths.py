"""Repository paths and stable, portable artifact identities."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("CEREBRAS_RESEARCH_ROOT", Path(__file__).resolve().parents[2])).resolve()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tree_digest(folder):
    files = {str(p.relative_to(folder)): digest(p) for p in sorted(Path(folder).rglob("*.csl"))}
    if not files:
        raise ValueError(f"No CSL sources in {folder}")
    return identity(files)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def contained(folder, relative):
    """Artifact and overlay paths may not escape their declared directory."""
    path = (Path(folder) / relative).resolve()
    if not path.is_relative_to(Path(folder).resolve()):
        raise ValueError(f"Path escapes artifact directory: {relative}")
    return path
