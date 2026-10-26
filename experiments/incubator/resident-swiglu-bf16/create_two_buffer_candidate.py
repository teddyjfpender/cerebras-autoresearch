"""Fold row accumulation and activation into two live FP32 tiles; no baseline edit."""

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
    parent = ROOT / "experiments/candidates/bf16-swiglu-vector-scratch-tail4-runtime"
    manifest = json.loads((parent / "candidate.json").read_text())
    baseline_path = ROOT / manifest["qualification_receipt"]
    baseline = load_validated(baseline_path)
    if baseline["provenance"]["evaluator_sha256"] != identity(evaluator_files(ROOT)):
        raise ValueError("Require matching full baseline evaluator")
    canonical = (ROOT / "kernels/composed/resident-swiglu-bf16/src/meshgemm.csl").read_text()
    source = (parent / "src/meshgemm.csl").read_text()
    for line in (
        "var row_accum: [Mt*Ft]f32;\n",
        "var row_accum_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Mt*Ft} -> row_accum[i] });\n",
    ):
        assert source.count(line) == 1
        source = source.replace(line, "")
    source = source.replace("|i|{Ft} -> row_accum[i]", "|i|{Ft} -> res_tile[i]")
    pack = """fn pack_outputs() void {
    // One row is already both row-major and column-major; no transform is needed.
    if (Mt == 1) { return; }
    // Communication has completed. W1 holds one full FP32 tile when 2*Mt<=Ht.
    var scratch = @set_dsd_base_addr(res_dsd, @ptrcast([*]f32, &W_1_tile));
    var packed_target = @set_dsd_base_addr(packed_output_dsd, @ptrcast([*]f32, &W_1_tile));
    var row: i16 = 0;
    while (row < Mt) : (row += 1) {
        var input = @increment_dsd_offset(row_output_dsd, row*Ft, f32);
        var destination = @increment_dsd_offset(packed_target, row, f32);
        @fmovs(destination, input);
    }
    @fmovs(res_dsd, scratch);
}"""
    source = source.replace(functions(source)["pack_outputs"], pack)
    source = source.replace("and (Ft > Ht))", "and (Ft > Ht) and (2*Mt <= Ht))")
    zero = """        if ((stage < 2) and (Ft >= 32) and (Ft > Ht) and (2*Mt <= Ht)) { @fmovs(row_accum_dsd, 0.0); }
        else { @fmovs(res_dsd, 0.0); }"""
    assert source.count(zero) == 1
    source = source.replace(zero, "        @fmovs(res_dsd, 0.0);")
    original = functions(source)["nonlinear_vector_product"]
    updated = original.replace("row_accum_dsd", "res_dsd").replace("row_accum[i]", "res_tile[i]")
    old_finish = """    @fmuls(res_dsd, res_dsd, 0.5);
    @fadds(res_dsd, res_dsd, res_dsd);"""
    assert updated.count(old_finish) == 1
    # BF16 G times exact BF16 0.5 accumulates in FP32. This avoids a third tile.
    updated = updated.replace(old_finish, "    @fmachs(res_dsd, res_dsd, gate, @as(fp, 0.5));")
    old_product = """    @fh2s(res_dsd, up);
    @fmuls(res_dsd, res_dsd, res_dsd);"""
    assert updated.count(old_product) == 1
    updated = updated.replace(
        old_product,
        "    @fh2s(silu_square_dsd, up);\n    @fmuls(res_dsd, res_dsd, silu_square_dsd);",
    )
    source = source.replace(original, updated)
    source = source.replace(
        "// Reuse res_tile for FP32 operands and row_accum for polynomial output.",
        "// res_tile holds G, then polynomial output; W1 holds squares, then FP32 U.",
    )
    source = source.replace(
        "// Contiguous expansion accumulators with a timed full column-major pack.",
        "// Accumulate expansion rows directly in res_tile; transpose via dead W1.",
    )
    if "row_accum" in source:
        raise ValueError("Separate row accumulator still exists")
    verify_timing_contract(canonical, source, "resident_swiglu_bf16")
    name = "bf16-swiglu-two-buffer-tail4"
    folder = candidate_init(get_profile(baseline["profile"]["id"]), name)
    path = folder / "candidate.json"
    result = json.loads(path.read_text())
    hypothesis = "Remove the separate FP32 row accumulator: accumulate expansions in res_tile, transpose via dead W1 after all transfers complete, and copy back before protected BF16 stage stores. For Mt=1 the two layouts coincide. Vector activation reuses res_tile for polynomial output and W1 for squares, then FP32 U; mixed BF16 G*0.5 with FP32 FMA restores the linear SiLU term without another tile. Guard both uses by 2*Mt<=Ht. This trades a timed extra pack pass for lower program memory, preserving every term, precision boundary, frame and state store."
    result.update(
        shape_specific=False,
        approach="resident_swiglu_bf16.two-live-fp32-tiles",
        hypothesis=hypothesis,
        falsification=manifest["falsification"]
        + " Reject any transpose-before-transfer completion, scratch extent beyond W1, missing copy-back, incorrect identity-layout case or missing original G in the mixed FMA. Require complete four-shape full/fresh memory and worst-cell gates before widening the claim.",
        author="unknown",
        model="unknown",
        agent_harness="Codex",
        qualification_receipt=manifest["qualification_receipt"],
        qualification_receipt_sha256=digest(baseline_path),
        parent_source_sha256=tree_digest(parent),
        parent_manifest_sha256=digest(parent / "candidate.json"),
        primitive_math_probe_sha256=manifest["primitive_math_probe_sha256"],
        heldout_status="Follow-up hypothesis after the held-out baselines; distinct from the original predeclared frozen candidate.",
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    (folder / "src/meshgemm.csl").write_text(source)
    (folder / "README.md").write_text(
        "# " + name + "\n\n" + hypothesis + "\n\n" + result["falsification"] + "\n"
    )
    print(folder)


if __name__ == "__main__":
    main()
