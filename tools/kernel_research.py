#!/usr/bin/env python3
"""Compatibility entry point for the original, archived MeshGEMV experiments."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cerebras_research import legacy

if __name__ == "__main__":
    try:
        raise SystemExit(legacy.main())
    except (RuntimeError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
