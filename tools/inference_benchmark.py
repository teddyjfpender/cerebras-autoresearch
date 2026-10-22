#!/usr/bin/env python3
"""Compatibility entry point for optional hosted API telemetry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cerebras_research import telemetry

if __name__ == "__main__":
    raise SystemExit(telemetry.main())
