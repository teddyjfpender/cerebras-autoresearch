#!/usr/bin/env python3
"""Checkout entry point; installation and third-party Python packages are optional."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cerebras_research.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
