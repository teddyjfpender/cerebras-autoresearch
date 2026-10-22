"""Declarative, validated benchmark profiles and suites."""

import json
import math
import re

from .paths import ROOT, contained, identity

CATALOG = ROOT / "benchmarks/manifests/catalog.json"
ADAPTERS = {"meshgemv", "meshgemm", "pipeline", "sdk"}


def load_catalog():
    data = json.loads(CATALOG.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark catalog schema")
    profiles = []
    for relative in data["includes"]:
        manifest = json.loads(contained(CATALOG.parent, relative).read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError(f"Unsupported profile manifest: {relative}")
        profiles.extend(manifest["profiles"])
    names = set()
    for profile in profiles:
        name = profile["id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]+", name) or name in names:
            raise ValueError(f"Invalid or duplicate profile: {name}")
        names.add(name)
        if profile["adapter"] not in ADAPTERS:
            raise ValueError(f"Unknown adapter: {profile['adapter']}")
        if not profile["variants"] or profile["variants"][0] != "baseline":
            raise ValueError(f"Missing baseline for {name}")
        grid = profile["pe_grid"]
        if len(grid) != 2 or any(type(v) is not int or v < 1 for v in grid):
            raise ValueError(f"Invalid PE grid: {name}")
        if not profile["dimensions"] or any(
            type(v) is not int or v < 1 for v in profile["dimensions"].values()
        ):
            raise ValueError(f"Invalid dimensions: {name}")
        if profile["validation"] not in (
            "independent_six_case",
            "upstream_reference",
            "diagnostic_only",
        ):
            raise ValueError(f"Unknown validation tier: {name}")
        for key, lower in (("timeout_seconds", 0), ("work_units", 0), ("regression_limit", 1)):
            value = profile[key]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < lower
                or value == 0
            ):
                raise ValueError(f"Invalid {key}: {name}")
        if profile["adapter"] != "sdk":
            if (
                profile["validation"] != "independent_six_case"
                or profile["metric"] != "max_local_cycles"
            ):
                raise ValueError(
                    f"Independent adapter requires its numerical/timing contract: {name}"
                )
            expected_timing = {"meshgemv": (1, 10), "meshgemm": (0, 1), "pipeline": (1, 4)}[
                profile["adapter"]
            ]
            timing = profile["timing_protocol"]
            if (timing["warmups"], timing["repeats"]) != expected_timing:
                raise ValueError(f"Timing profile disagrees with trusted driver: {name}")
        if profile["adapter"] in ("meshgemv", "meshgemm"):
            p = grid[0]
            dims = profile["dimensions"]
            if grid != [4, 4] or any(v % p for v in dims.values()):
                raise ValueError(f"Dense dimensions must be positive multiples of four: {name}")
        if profile["adapter"] == "pipeline" and grid != [1, 1]:
            raise ValueError(f"Current pipeline adapter requires one PE: {name}")
        if profile["adapter"] == "sdk":
            contained(ROOT / ".deps/sdk-examples", profile["source_path"])
            for key in ("compile", "run"):
                if not profile[key] or not all(isinstance(v, str) for v in profile[key]):
                    raise ValueError(f"Invalid command vector: {name}/{key}")
            if profile["compile"][0] != "cslc" or profile["run"][0] != "run.py":
                raise ValueError(f"Unexpected SDK entry point: {name}")
    for path in (ROOT / "benchmarks/suites").glob("*.json"):
        suite = json.loads(path.read_text())
        if not suite["profiles"] or len(suite["profiles"]) != len(set(suite["profiles"])):
            raise ValueError(f"Empty or duplicate suite members: {path}")
        missing = set(suite["profiles"]) - names
        if missing:
            raise ValueError(f"Unknown suite members: {missing}")
    return {p["id"]: p for p in profiles}


def get_profile(name):
    profiles = load_catalog()
    if name not in profiles:
        raise ValueError(f"Unknown profile {name}; use 'list'")
    return profiles[name]


def get_suite(name):
    path = contained(ROOT / "benchmarks/suites", name + ".json")
    data = json.loads(path.read_text())
    catalog = load_catalog()
    return [catalog[key] for key in data["profiles"]]


def profile_identity(profile):
    return identity(profile)
