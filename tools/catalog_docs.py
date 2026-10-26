#!/usr/bin/env python3
"""Keep configured profile/suite tables in sync; execution evidence stays separate."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cerebras_research.catalog import load_catalog  # noqa: E402

DOC = ROOT / "docs/benchmarks/catalog.md"


def cell(value):
    return str(value).replace("\n", " ").replace("|", "\\|")


def replace_table(text, name, table):
    start = f"<!-- catalog:{name}:start -->"
    end = f"<!-- catalog:{name}:end -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Require exactly one {name} table marker pair")
    a, b = text.index(start) + len(start), text.index(end)
    if b <= a:
        raise ValueError("Reversed catalog table markers")
    return text[:a] + "\n" + table.rstrip() + "\n" + text[b:]


def render(text):
    catalog = load_catalog()
    lines = [
        "| Profile | Family | Precision | Validation | Primary |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, profile in sorted(catalog.items()):
        values = [
            f"`{name}`",
            profile["family"],
            profile["precision"],
            profile["validation"],
            profile["metric"] or "correctness only",
        ]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    text = replace_table(text, "profiles", "\n".join(lines))
    # Human purposes are retained for older manifests without a description.
    old_purposes = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 5 and parts[1].startswith("`") and parts[2].isdigit():
            old_purposes[parts[1].strip("`")] = parts[3]
    lines = ["| Suite | Profiles | Purpose |", "| --- | ---: | --- |"]
    for path in sorted((ROOT / "benchmarks/suites").glob("*.json")):
        suite = json.loads(path.read_text())
        purpose = (
            suite.get("description")
            or old_purposes.get(path.stem)
            or "Configured collection; see suite manifest"
        )
        lines.append(f"| `{path.stem}` | {len(suite['profiles'])} | {cell(purpose)} |")
    return replace_table(text, "suites", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    original = DOC.read_text()
    updated = render(original)
    if args.write:
        DOC.write_text(updated)
        print("Catalog tables updated from manifests")
    elif updated != original:
        raise SystemExit("Catalog tables are stale; run make docs")
    else:
        print("Catalog tables match configured profiles and suites")


if __name__ == "__main__":
    main()
