"""Create lifetime-reused FP32 vector SiLU from the frozen scalar hypothesis."""

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
    parent = ROOT / "experiments/candidates/bf16-swiglu-projection-silu6-reuse"
    parent_manifest = json.loads((parent / "candidate.json").read_text())
    baseline_path = ROOT / parent_manifest["qualification_receipt"]
    baseline = load_validated(baseline_path)
    if baseline["provenance"]["evaluator_sha256"] != identity(evaluator_files(ROOT)):
        raise ValueError("Require a matching evaluator")
    profile = get_profile(baseline["profile"]["id"])
    canonical = (ROOT / "kernels/composed/resident-swiglu-bf16/src/meshgemm.csl").read_text()
    source = (parent / "src/meshgemm.csl").read_text()
    probe = json.loads((ROOT / "runs/maintenance/bf16-silu-primitive.json").read_text())
    row = next(r for r in probe["results"] if r["degree"] == 6)
    if not row["primitive_gate_passed"]:
        raise ValueError("Require the separate BF16 primitive screen")
    coefficients = row["coefficients"]
    original = functions(source)["nonlinear_product"]
    vector = """// U has already been rounded into U_HISTORY; projection scratch is dead.
// Reuse res_tile for FP32 operands and row_accum for polynomial output.
// A single additional square tile has no projection or communication alias.
var silu_square: [Mt*Ft]f32;
var silu_square_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Mt*Ft} -> silu_square[i] });
fn nonlinear_product() void {
    var gate = @increment_dsd_offset(g_history_dsd, frame*Mt*Ft, fp);
    var up = @increment_dsd_offset(u_history_dsd, frame*Mt*Ft, fp);
    var target = @increment_dsd_offset(v_history_dsd, frame*Mt*Ft, fp);
    @fh2s(res_dsd, gate);
    // Clamp before squaring; all out-of-range gates use stable exp below.
    @fmaxs(res_dsd, res_dsd, -4.0);
    @fmuls(res_dsd, res_dsd, -1.0);
    @fmaxs(res_dsd, res_dsd, -4.0);
    @fmuls(res_dsd, res_dsd, -1.0);
    @fmuls(silu_square_dsd, res_dsd, res_dsd);
"""
    vector += f"    @fmovs(row_accum_dsd, {coefficients[-1]!r});\n"
    for coefficient in coefficients[-2::-1]:
        vector += "    @fmuls(row_accum_dsd, row_accum_dsd, silu_square_dsd);\n"
        vector += f"    @fadds(row_accum_dsd, row_accum_dsd, {coefficient!r});\n"
    vector += """    @fmuls(row_accum_dsd, row_accum_dsd, silu_square_dsd);
    @fmuls(res_dsd, res_dsd, 0.5);
    @fadds(row_accum_dsd, row_accum_dsd, res_dsd);
    var i: i16 = 0;
    while (i < Mt*Ft) : (i += 1) {
        const g = @as(f32, G_HISTORY[frame*Mt*Ft+i]);
        if ((g < -4.0) or (g > 4.0)) { row_accum[i] = silu_polynomial(g); }
    }
    @fh2s(res_dsd, up);
    @fmuls(row_accum_dsd, row_accum_dsd, res_dsd);
    @fs2h(target, row_accum_dsd);
}
"""
    source = source.replace(original, vector)
    verify_timing_contract(canonical, source, profile["adapter"])
    name = "bf16-swiglu-vector-silu6-reuse"
    folder = candidate_init(profile, name)
    path = folder / "candidate.json"
    manifest = json.loads(path.read_text())
    hypothesis = "Vectorize FP32 degree-six SiLU while reusing dead projection scratch: res_tile holds clamped G then quantized U; row_accum holds the polynomial; one extra FP32 tile holds G squared. U is saved before reuse, down accumulation clears res, all BF16 stage stores and fallback exp remain timed. Measure cycle gain versus scalar composition without exceeding memory gates."
    manifest.update(
        shape_specific=False,
        approach="resident_swiglu_bf16.vector-silu6-lifetime-reuse",
        hypothesis=hypothesis,
        falsification=parent_manifest["falsification"],
        author="unknown",
        model="unknown",
        agent_harness="Codex",
        qualification_receipt=parent_manifest["qualification_receipt"],
        qualification_receipt_sha256=digest(baseline_path),
        parent_source_sha256=tree_digest(parent),
        parent_manifest_sha256=digest(parent / "candidate.json"),
        primitive_math_probe_sha256=digest(ROOT / "runs/maintenance/bf16-silu-primitive.json"),
    )
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    (folder / "src/meshgemm.csl").write_text(source)
    (folder / "README.md").write_text(
        "# " + name + "\n\n" + hypothesis + "\n\n" + manifest["falsification"] + "\n"
    )
    print(folder)


if __name__ == "__main__":
    main()
