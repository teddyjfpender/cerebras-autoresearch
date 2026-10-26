"""Make vector activation fit the footprint gate through proved scratch lifetimes."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cerebras_research.adapters import candidate_init, functions, verify_timing_contract  # noqa: E402
from cerebras_research.catalog import get_profile  # noqa: E402
from cerebras_research.paths import digest, identity, tree_digest  # noqa: E402
from cerebras_research.provenance import evaluator_files  # noqa: E402
from cerebras_research.scoring import load_validated  # noqa: E402


def main():
    parent = ROOT / "experiments/candidates/bf16-swiglu-vector-silu6-reuse"
    scalar = ROOT / "experiments/candidates/bf16-swiglu-projection-silu6-reuse"
    manifest = json.loads((parent / "candidate.json").read_text())
    baseline_path = ROOT / manifest["qualification_receipt"]
    baseline = load_validated(baseline_path)
    if baseline["provenance"]["evaluator_sha256"] != identity(evaluator_files(ROOT)):
        raise ValueError("Require matching full baseline evaluator")
    source = (parent / "src/meshgemm.csl").read_text()
    original = "var silu_square: [Mt*Ft]f32;\nvar silu_square_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Mt*Ft} -> silu_square[i] });"
    assert source.count(original) == 1
    source = source.replace(
        original,
        "// W1 send/receive has completed before nonlinear_product is called.\n"
        "// Down-stage receive overwrites all logical W1 words before reuse.\n"
        "var silu_square_dsd = @get_dsd(mem1d_dsd, .{ .base_address = @ptrcast([*]f32, &W_1_tile), .extent = Ht*Ft/2 });",
    )
    source = source.replace("W_1_tile: [Kt*Nt]fp =", "W_1_tile: [Kt*Nt]fp align(4) =")
    source = source.replace(
        "// A single additional square tile has no projection or communication alias.",
        "// The dead W1 receive buffer holds squares only when 2*Mt<=Ht.",
    )
    original_fn = functions(source)["nonlinear_product"]
    vector_fn = original_fn.replace("fn nonlinear_product()", "fn nonlinear_vector_product()", 1)
    vector_fn = vector_fn.replace(
        "    var gate =",
        "    silu_square_dsd = @set_dsd_length(silu_square_dsd, @as(u16, Mt*Ft));\n    var gate =",
        1,
    )
    scalar_fn = functions((scalar / "src/meshgemm.csl").read_text())["nonlinear_product"]
    scalar_fn = scalar_fn.replace("fn nonlinear_product()", "fn nonlinear_scalar_product()", 1)
    dispatcher = """fn nonlinear_product() void {
    if ((Ft >= 32) and (2*Mt <= Ht)) { nonlinear_vector_product(); }
    else { nonlinear_scalar_product(); }
}"""
    source = source.replace(original_fn, vector_fn + "\n\n" + scalar_fn + "\n\n" + dispatcher)
    canonical = (ROOT / "kernels/composed/resident-swiglu-bf16/src/meshgemm.csl").read_text()
    verify_timing_contract(canonical, source, "resident_swiglu_bf16")
    name = "bf16-swiglu-vector-scratch-dispatch"
    folder = candidate_init(get_profile(baseline["profile"]["id"]), name)
    path = folder / "candidate.json"
    result = json.loads(path.read_text())
    hypothesis = "Use FP32 vector degree-six SiLU for Ft>=32 and 2*Mt<=Ht, reinterpreting the 4-byte-aligned dead W1 receive buffer as square scratch. Its last send/receive completes before the stage callback; next down-stage reception overwrites every logical W1 word before use. This removes the additional square allocation. Smaller tiles use the scalar composition to bound code/descriptor overhead. Every stage, term and output/state store remains fixed."
    result.update(
        shape_specific=False,
        approach="resident_swiglu_bf16.vector-silu6-receive-scratch-dispatch",
        hypothesis=hypothesis,
        falsification=manifest["falsification"]
        + " Reject any alias used while communication is live or any scratch extent exceeding W1; compare complete working-state histories at both seeds.",
        author="unknown",
        model="unknown",
        agent_harness="Codex",
        qualification_receipt=manifest["qualification_receipt"],
        qualification_receipt_sha256=digest(baseline_path),
        parent_source_sha256=tree_digest(parent),
        parent_manifest_sha256=digest(parent / "candidate.json"),
        scalar_source_sha256=tree_digest(scalar),
        primitive_math_probe_sha256=manifest["primitive_math_probe_sha256"],
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    (folder / "src/meshgemm.csl").write_text(source)
    (folder / "README.md").write_text(
        "# " + name + "\n\n" + hypothesis + "\n\n" + result["falsification"] + "\n"
    )
    print(folder)


if __name__ == "__main__":
    main()
