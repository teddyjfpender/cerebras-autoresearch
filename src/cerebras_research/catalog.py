"""Declarative, validated benchmark profiles and suites."""

import json
import math
import re

from .paths import ROOT, contained, identity

CATALOG = ROOT / "benchmarks/manifests/catalog.json"
ADAPTERS = {
    "meshgemv",
    "meshgemm",
    "pipeline",
    "gated_mlp",
    "decoder",
    "reusable_gemm",
    "resident_swiglu",
    "resident_swiglu_bf16",
    "recurrent_gdn",
    "recurrent_gdn_native",
    "recurrent_gdn_full",
    "decode_attention",
    "sdk",
}


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
            if profile["validation"] != "independent_six_case" or profile["metric"] != (
                "coordinator_cycles"
                if profile["adapter"] == "decode_attention"
                else "max_local_cycles"
            ):
                raise ValueError(
                    f"Independent adapter requires its numerical/timing contract: {name}"
                )
            expected_timing = {
                "meshgemv": (1, 10),
                "meshgemm": (0, 1),
                "pipeline": (1, 4),
                "gated_mlp": (0, 1),
                "decoder": (0, 1),
                "decode_attention": (0, 1),
                "reusable_gemm": (0, 1),
                "resident_swiglu": (0, 1),
                "resident_swiglu_bf16": (0, 1),
                "recurrent_gdn": (0, 1),
                "recurrent_gdn_native": (0, 1),
                "recurrent_gdn_full": (0, 1),
            }[profile["adapter"]]
            timing = profile["timing_protocol"]
            if (timing["warmups"], timing["repeats"]) != expected_timing:
                raise ValueError(f"Timing profile disagrees with trusted driver: {name}")
        if profile["adapter"] in ("meshgemv", "meshgemm"):
            p = grid[0]
            dims = profile["dimensions"]
            if grid != [4, 4] or any(v % p for v in dims.values()):
                raise ValueError(f"Dense dimensions must be positive multiples of four: {name}")
        if profile["adapter"] == "reusable_gemm":
            dims = profile["dimensions"]
            if (
                grid != [4, 4]
                or set(dims) != {"M", "K", "N", "T"}
                or dims.get("T") != 4
                or any(dims[k] % 4 for k in ("M", "K", "N"))
                or profile["precision"] != "fp16"
                or profile.get("oracle_contract") != "resident_gemm_sequence_fp16_v1"
                or (dims["K"] + 4) * 2**-11 >= 1
            ):
                raise ValueError(f"Invalid resident GEMM sequence contract: {name}")
        if profile["adapter"] == "resident_swiglu":
            dims = profile["dimensions"]
            if (
                grid != [4, 4]
                or set(dims) != {"M", "H", "F", "T"}
                or dims.get("T") != 4
                or any(dims[k] % 4 for k in ("M", "H", "F"))
                or dims["F"] < dims["H"]
                or profile["precision"] != "fp16-fp32"
                or profile.get("oracle_contract") != "resident_swiglu_sequence_fp16_fp32_v1"
                or (dims["F"] + 4) * 2**-11 >= 1
                or profile["work_unit"] != "projection_arithmetic_operations"
                or profile["work_units"] != 6 * dims["T"] * dims["M"] * dims["H"] * dims["F"]
            ):
                raise ValueError(f"Invalid resident SwiGLU sequence contract: {name}")
        if profile["adapter"] == "resident_swiglu_bf16":
            dims = profile["dimensions"]
            if (
                grid != [4, 4]
                or set(dims) != {"M", "H", "F", "T"}
                or dims.get("T") != 4
                or any(dims[k] % 4 for k in ("M", "H", "F"))
                or dims["F"] < dims["H"]
                or (dims["M"] // 4) * (dims["F"] // 4) % 2
                or (dims["H"] // 4) * (dims["F"] // 4) % 2
                or profile["precision"] != "bf16-fp32-accum"
                or profile.get("oracle_contract") != "resident_swiglu_sequence_bf16_fp32_v1"
                or (dims["F"] + 4) * 2**-24 >= 1
                or profile["work_unit"] != "projection_arithmetic_operations"
                or profile["work_units"] != 6 * dims["T"] * dims["M"] * dims["H"] * dims["F"]
            ):
                raise ValueError(f"Invalid BF16 resident chain contract: {name}")
        if profile["adapter"] == "recurrent_gdn":
            dims = profile["dimensions"]
            if (
                grid != [4, 4]
                or set(dims) != {"K", "V", "T", "R"}
                or dims.get("K") not in (16, 32)
                or dims.get("V") != 16
                or dims.get("T") != 4
                or dims.get("R") != 3
                or profile["precision"] != "bf16-fp32-state"
                or profile.get("oracle_contract") != "resident_recurrent_gdn_bf16_fp32_v1"
                or profile["work_unit"] != "state_arithmetic_operations"
                or profile["work_units"]
                != 16 * dims["T"] * dims["R"] * (7 * dims["K"] * dims["V"] + 2 * dims["V"])
            ):
                raise ValueError(f"Invalid recurrent gated delta contract: {name}")
        if profile["adapter"] == "recurrent_gdn_native":
            dims = profile["dimensions"]
            cell = (dims.get("T"), dims.get("VT"), *grid)
            if (
                set(dims) != {"K", "V", "T", "R", "VT"}
                or dims.get("K") != 128
                or dims.get("V") != 128
                or dims.get("R") != 3
                or cell not in ((4, 8, 8, 6), (8, 4, 12, 8))
                or profile["precision"] != "bf16-fp32-state"
                or profile.get("oracle_contract") != "resident_recurrent_gdn_native_head_group_v1"
                or profile["work_unit"] != "state_arithmetic_operations"
                or profile["work_units"] != 3 * dims["T"] * (7 * 128 * 128 + 2 * 128)
            ):
                raise ValueError(f"Invalid native-width recurrent contract: {name}")
        if profile["adapter"] == "recurrent_gdn_full":
            dims = profile["dimensions"]
            if (
                dims != {"G": 16, "R": 3, "K": 128, "V": 128, "T": 2, "VT": 8}
                or grid != [32, 24]
                or profile["precision"] != "bf16-fp32-state"
                or profile.get("oracle_contract") != "resident_recurrent_gdn_full_native_heads_v1"
                or profile["work_unit"] != "state_arithmetic_operations"
                or profile["work_units"] != 16 * 3 * 2 * (7 * 128 * 128 + 2 * 128)
            ):
                raise ValueError(f"Invalid full native recurrent head contract: {name}")
        if profile["adapter"] == "pipeline" and grid != [1, 1]:
            raise ValueError(f"Current pipeline adapter requires one PE: {name}")
        if profile["adapter"] == "decode_attention":
            dims = profile["dimensions"]
            if (
                set(dims) != {"P", "S", "D", "G", "T", "L"}
                or grid != [dims["P"], 1]
                or not 2 <= dims["P"] <= 128
                or dims["P"] & (dims["P"] - 1)
                or dims["D"] not in (64, 128)
                or dims["G"] not in (2, 4)
                or dims["T"] != 4
                or not 0 < dims["L"] <= dims["P"] * dims["S"] - dims["T"]
                or dims["S"] * dims["D"] >= 32768
                or profile["precision"] != "fp32"
                or profile.get("oracle_contract") != "sharded_gqa_decode_fp32_v1"
            ):
                raise ValueError(f"Invalid sharded attention contract: {name}")
            expected_work = (
                4 * dims["G"] * dims["D"] * sum(dims["L"] + step + 1 for step in range(dims["T"]))
            )
            if profile["work_units"] != expected_work:
                raise ValueError(f"Attention work count changed: {name}")
            static_floats = (
                2 * dims["D"] * dims["S"]
                + 2 * dims["T"] * dims["G"] * dims["D"]
                + 2 * dims["T"] * dims["D"]
                + 2 * dims["G"] * dims["S"]
                + dims["P"] * dims["G"]
                + 2 * dims["G"]
                + 2 * dims["G"] * (dims["D"] + 1)
            )
            if 4 * static_floats + 28 > 52 * 1024:
                raise ValueError(
                    f"Attention static buffers leave insufficient code headroom: {name}"
                )
        if profile["adapter"] in ("gated_mlp", "decoder"):
            dims = profile["dimensions"]
            if (
                grid != [1, 1]
                or profile["precision"] != "fp32"
                or profile.get("oracle_contract") != "resident_transformer_fp32_v1"
                or dims.get("T") != 4
                or set(dims)
                != (
                    {"H", "F", "T", "C", "L", "A"}
                    if profile["adapter"] == "decoder"
                    else {"H", "F", "T"}
                )
            ):
                raise ValueError(f"Invalid resident block contract: {name}")
            if profile["adapter"] == "decoder" and (
                dims["L"] + dims["T"] > dims["C"] or dims["H"] % (2 * dims["A"])
            ):
                raise ValueError(f"Invalid decoder cache/head dimensions: {name}")
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
