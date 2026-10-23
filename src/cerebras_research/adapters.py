"""Source preparation, candidate overlays, and explicit compile/run plans."""

import json
import re
import shutil

from . import legacy
from .paths import ROOT, contained, tree_digest, write_json


def baseline_source(profile):
    kind = profile["adapter"]
    if kind == "decode_attention":
        return ROOT / "kernels/attention/sharded-gqa", {"repository": "local"}
    if kind == "pipeline":
        return ROOT / "kernels/composed/matvec-epilogue", {"repository": "local"}
    if kind == "reusable_gemm":
        return ROOT / "kernels/dense/reusable-gemm", {
            "repository": "local",
            "derived_from": "WaferLLM",
            "commit": "fd1c2daae37cd68706c03fc8009887ecee9900f8",
        }
    if kind in ("gated_mlp", "decoder"):
        return ROOT / {
            "gated_mlp": "kernels/composed/gated-mlp",
            "decoder": "kernels/decoder/causal-layer",
        }[kind], {"repository": "local"}
    key = "sdk-examples" if kind == "sdk" else "waferllm"
    spec = json.loads((ROOT / "config/sources.lock.json").read_text())[key]
    checkout = ROOT / ".deps" / key
    legacy.verify_source(checkout, spec)
    suffix = (
        profile["source_path"]
        if kind == "sdk"
        else {"meshgemv": "MeshGEMV", "meshgemm": "MeshGEMM"}[kind]
    )
    return checkout / suffix, spec


def materialize(profile, folder, csl_only=False):
    """Include pinned shared libraries inside the isolated, fingerprinted tree."""
    source, spec = baseline_source(profile)
    layers = [(source, folder)]
    for name in profile.get("shared_libraries", []):
        layers.append(
            (contained(source.parent / "benchmark-libs", name), folder / "benchmark-libs" / name)
        )
    for origin, destination in layers:
        if csl_only:
            for path in origin.rglob("*.csl"):
                target = destination / path.relative_to(origin)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        else:
            shutil.copytree(
                origin,
                destination,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
    # Candidate initialization copies CSL only; retain the pinned repository's
    # notices alongside that code, including locally derived reference kernels.
    notice_roots = [source]
    kind = profile.get("adapter")
    if kind in ("meshgemv", "meshgemm", "sdk"):
        notice_roots.append(ROOT / ".deps" / ("sdk-examples" if kind == "sdk" else "waferllm"))
    for name in ("LICENSE", "NOTICE"):
        options = [
            parent / suffix for parent in notice_roots for suffix in (name + ".upstream", name)
        ]
        notice = next((path for path in options if path.is_file()), None)
        if notice:
            shutil.copy2(notice, folder / (name + ".upstream"))
    # Original SDK examples import sibling benchmark-libs from src/. Relocate
    # those imports (including dynamic library path strings) into this run's
    # own tree, without changing arithmetic.
    if profile.get("shared_libraries"):
        for path in folder.rglob("*.csl"):
            path.write_text(path.read_text().replace("../../benchmark-libs/", "../benchmark-libs/"))
    return spec


def transform(folder, profile, variant):
    if variant == "unroll" and profile["adapter"] == "meshgemv":
        path = folder / "src/meshgemv.csl"
        source = path.read_text()
        marker = "    @map(gemv_static_step, X_dsd);"
        if source.count(marker) != 1:
            raise ValueError("Unroll marker did not match pinned baseline")
        lines = "\n".join(
            f"    @fmach(res_dest_dsr, res_src0_dsr, W_src1_dsr, X_tile[{i}]);"
            for i in range(profile["dimensions"]["M"] // 4)
        )
        path.write_text(source.replace(marker, lines))
    elif variant == "bank-alignment" and profile["adapter"] == "meshgemv":
        path = folder / "src/meshgemv.csl"
        source = path.read_text()
        original = "var W_tile: [Mt*Nt]f16 = @zeros([Mt*Nt]f16);\nvar ptr_W: [*]f16 = &W_tile;\nvar W_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Nt} -> W_tile[i] });"
        replacement = "const weight_storage_t = struct { pad: [2]f16, values: [Mt*Nt]f16, };\nvar W_storage: weight_storage_t align(16) = .{ .pad = @zeros([2]f16), .values = @zeros([Mt*Nt]f16) };\nvar ptr_W: [*]f16 = &W_storage.values;\nvar W_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &W_storage.values, .extent = @as(u16, Nt) });"
        if source.count(original) != 1:
            raise ValueError("Alignment marker did not match pinned baseline")
        path.write_text(
            source.replace(original, replacement).replace(
                "var res_tile: [Nt]f16 =", "var res_tile: [Nt]f16 align(16) ="
            )
        )
    elif variant not in ("baseline", "fused", "candidate"):
        raise ValueError(f"Unsupported variant {variant} for {profile['id']}")


def functions(text):
    masked = re.sub(r"//[^\n]*|/\*.*?\*/", lambda m: " " * len(m[0]), text, flags=re.S)
    found = {}
    for match in re.finditer(r"\bfn\s+(\w+)\s*\([^)]*\)[^{]*\{", masked):
        depth, end = 1, match.end()
        while depth and end < len(masked):
            depth += (masked[end] == "{") - (masked[end] == "}")
            end += 1
        found[match[1]] = text[match.start() : end]
    return found


def verify_timing_contract(original, edited, kind):
    if kind == "decode_attention":
        # Freeze layout and all data declarations, timers, causal masking, cache
        # updates, collectives and output stores. Only local arithmetic may vary.
        begin = "// BEGIN LOCAL COMPUTE: candidate edits are confined to this region."
        end = "// END LOCAL COMPUTE"
        if begin not in original:
            if edited != original:
                raise ValueError("Candidate changed protected attention layout")
            return
        for text in (original, edited):
            if text.count(begin) != 1 or text.count(end) != 1:
                raise ValueError("Candidate changed protected attention markers")
        before, after = original.split(begin)
        _, tail = after.split(end)
        candidate_before, candidate_after = edited.split(begin)
        _, candidate_tail = candidate_after.split(end)
        if before != candidate_before or tail != candidate_tail:
            raise ValueError("Candidate changed protected attention control")
        return
    names = (
        {"compute", "snapshot"}
        if kind in ("pipeline", "gated_mlp", "decoder")
        else {
            "sequence_host",
            "save_frame",
            "init_task",
            "exit",
            "f_memcpy_timestamps",
            "f_reference_timestamps",
            "meshgemv_host",
            "meshgemv_entry",
            "meshgemm_host",
            "meshgemm_entry",
        }
    )
    # Ignore trailing whitespace inherited from upstream examples; every
    # nontrailing character in protected function bodies remains fixed.
    before, after = (
        functions("\n".join(line.rstrip() for line in source.splitlines()))
        for source in (original, edited)
    )
    for name in names & before.keys():
        if before[name] != after.get(name):
            raise ValueError(f"Candidate changed protected timing/control function: {name}")


def overlay(folder, candidate, profile, baseline_hash):
    manifest = json.loads((candidate / "candidate.json").read_text())
    if (
        manifest["adapter"] != profile["adapter"]
        or manifest["baseline_tree_sha256"] != baseline_hash
    ):
        raise ValueError("Candidate belongs to a different source baseline/adapter")
    if manifest.get("shape_specific") and manifest["initial_profile"] != profile["id"]:
        raise ValueError("This candidate is specific to its initial shape")
    existing = {str(p.relative_to(folder)) for p in folder.rglob("*.csl")}
    for path in candidate.rglob("*.csl"):
        relative = str(path.relative_to(candidate))
        if profile["adapter"] == "decode_attention" and relative not in existing:
            raise ValueError("Attention candidates cannot add unreviewed CSL modules")
        target = contained(folder, relative)
        if relative in existing and profile["adapter"] != "sdk":
            verify_timing_contract(target.read_text(), path.read_text(), profile["adapter"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def candidate_init(profile, name, start_variant="baseline"):
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]+", name):
        raise ValueError("Candidate name must use lowercase letters, digits, dot, dash, underscore")
    if start_variant not in profile["variants"] or start_variant == "candidate":
        raise ValueError("Unsupported starting variant")
    folder = ROOT / "experiments/candidates" / name
    folder.mkdir(parents=True, exist_ok=False)
    spec = materialize(profile, folder, csl_only=True)
    baseline_hash = tree_digest(folder)
    transform(folder, profile, start_variant)
    write_json(
        folder / "candidate.json",
        {
            "adapter": profile["adapter"],
            "initial_profile": profile["id"],
            "source": spec,
            "baseline_tree_sha256": baseline_hash,
            "starting_variant": start_variant,
            "shape_specific": start_variant == "unroll",
            "approach": profile["adapter"] + "." + start_variant,
            "hypothesis": "TODO: describe the change, bottleneck, and expected resource tradeoff.",
            "falsification": "TODO: state the observation that would reject this hypothesis.",
        },
    )
    return folder


def prepare(profile, folder, variant, candidate=None):
    if variant not in profile["variants"]:
        raise ValueError(f"Unsupported variant for {profile['id']}: {variant}")
    spec = materialize(profile, folder)
    baseline_hash = tree_digest(folder)
    if candidate:
        overlay(folder, candidate, profile, baseline_hash)
    else:
        transform(folder, profile, variant)
    kind, dims = profile["adapter"], profile["dimensions"]
    if kind == "sdk":
        compile_args = list(profile["compile"])
    else:
        if kind == "meshgemv":
            params = f"P:4,Mt:{dims['M'] // 4},Nt:{dims['N'] // 4},pe_num_group:2,root_1st_phase:1,root_2nd_phase:3"
        elif kind in ("meshgemm", "reusable_gemm"):
            params = f"P:4,Mt:{dims['M'] // 4},Kt:{dims['K'] // 4},Nt:{dims['N'] // 4}"
            if kind == "reusable_gemm":
                params += f",T:{dims['T']}"
        elif kind in ("gated_mlp", "decoder", "decode_attention"):
            params = ",".join(f"{key}:{value}" for key, value in dims.items())
        else:
            mode = (
                json.loads((candidate / "candidate.json").read_text())["starting_variant"]
                if candidate
                else variant
            )
            params = f"K:{dims['K']},N:{dims['N']},FUSED:{int(mode == 'fused')}"
        width, height = profile["pe_grid"]
        compile_args = [
            "cslc",
            "./src/layout.csl",
            "--arch=wse3",
            f"--fabric-dims={width + 7},{height + 2}",
            "--fabric-offsets=4,1",
            "--params=" + params,
            "-o",
            "out",
            "--memcpy",
            "--channels=1",
        ]
    for path in (ROOT / "benchmarks/drivers").glob("*.py"):
        destination = folder / (
            "benchmark_" + path.name if path.name == "pipeline.py" else path.name
        )
        if destination.exists():
            raise ValueError(f"Trusted driver name collides with upstream file: {destination.name}")
        shutil.copy2(path, destination)
    return compile_args, spec, baseline_hash
