"""Create fixed-contract hypotheses only after a matching full chain baseline qualifies."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cerebras_research.adapters import candidate_init, functions  # noqa: E402
from cerebras_research.catalog import get_profile  # noqa: E402
from cerebras_research.paths import digest, identity, tree_digest  # noqa: E402
from cerebras_research.provenance import evaluator_files  # noqa: E402
from cerebras_research.scoring import load_validated  # noqa: E402

NAMES = [
    "swiglu-projection-dispatch",
    "swiglu-receive-buffer-reuse",
    "swiglu-silu-degree9",
    "swiglu-projection-silu-reuse",
]


def projections(source):
    original = functions(source)["mm_compute"]
    a = original.index("        for (@range(i16, active_kt)) |k| {")
    b = original.index("\n        step += 1;", a)
    columns = """        out_dsd = @set_dsd_base_addr(out_dsd, ptr_out);
        @load_to_dsr(out_dest_dsr, out_dsd, .{ .save_address = true });
        @load_to_dsr(out_src0_dsr, out_dsd, .{ .save_address = true });
        for (@range(i16, active_kt)) |k| {
            @set_dsr_base_addr(out_dest_dsr, res_tile);
            @set_dsr_base_addr(out_src0_dsr, res_tile);
            @load_to_dsr(X_src1_dsr, X_dsd, .{ .save_address = false });
            @map(gemv_static_step_X, W_dsd);
            X_dsd = @increment_dsd_offset(X_dsd, Mt, f16);
            W_dsd = @increment_dsd_offset(W_dsd, active_nt, f16);
        }
"""
    row_compute = (
        """        if ((stage < 2) and (Ft >= 32) and (Ft > Ht)) {
            var weights = @set_dsd_base_addr(row_weight_dsd, ptr_W_send);
            var k: i16 = 0;
            while (k < Ht) : (k += 1) {
                @load_to_dsr(X_src1_dsr, weights, .{ .save_address = false });
                var row: i16 = 0;
                while (row < Mt) : (row += 1) {
                    var output = @increment_dsd_offset(row_output_dsd, row*Ft, f16);
                    @load_to_dsr(out_dest_dsr, output, .{ .save_address = false });
                    @load_to_dsr(out_src0_dsr, output, .{ .save_address = false });
                    @fmach(out_dest_dsr, out_src0_dsr, X_src1_dsr, ptr_X_send[k*Mt+row]);
                }
                weights = @increment_dsd_offset(weights, Ft, f16);
            }
        } else {
"""
        + columns
        + "        }\n"
    )
    edited = original[:a] + row_compute + original[b:]
    needle = "    } else {\n        step = 0;"
    assert needle in edited
    edited = edited.replace(
        needle,
        "    } else {\n        if ((stage < 2) and (Ft >= 32) and (Ft > Ht)) { pack_outputs(); }\n        step = 0;",
    )
    source = source.replace(original, edited)
    extra = """// Contiguous expansion accumulators with a timed full column-major pack.
var row_accum: [Mt*Ft]f16;
var row_accum_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Mt*Ft} -> row_accum[i] });
var row_output_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Ft} -> row_accum[i] });
var row_weight_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Ft} -> W_0_tile[i] });
var packed_output_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Ft} -> res_tile[i*Mt] });
fn pack_outputs() void {
    var row: i16 = 0;
    while (row < Mt) : (row += 1) {
        var source = @increment_dsd_offset(row_output_dsd, row*Ft, f16);
        var destination = @increment_dsd_offset(packed_output_dsd, row, f16);
        @fmovh(destination, source);
    }
}

"""
    source = source.replace(
        "// ------------------------- Timer", extra + "// ------------------------- Timer", 1
    )
    needle = "        @fmovh(res_dsd, 0.0);"
    assert source.count(needle) == 1
    return source.replace(
        needle,
        "        if ((stage < 2) and (Ft >= 32) and (Ft > Ht)) { @fmovh(row_accum_dsd, 0.0); }\n        else { @fmovh(res_dsd, 0.0); }",
    )


def receive_reuse(source):
    for name in ("working_x0_dsd", "working_x1_dsd", "working_w1_dsd"):
        line = f"    @fmovh({name}, 0.0);\n"
        assert source.count(line) == 1
        source = source.replace(line, "")
    return source


def activation(source, coefficients):
    helper = """// Degree-nine polynomial for SiLU's even component on |g|<=4.
// Every finite FP16 gate in that range passed the prescribed host math screen.
// Hardware/full-chain correctness and timing remain subject to normal gates.
fn silu_polynomial(g: f32) f32 {
    if ((g < -4.0) or (g > 4.0)) {
        if (g >= 0.0) { return g/(1.0+math.exp(-g)); }
        const e = math.exp(g);
        return g*e/(1.0+e);
    }
    const square: f32 = g*g;
"""
    helper += f"    var p: f32 = {coefficients[-1]!r};\n"
    helper += "".join(f"    p = p*square + {c!r};\n" for c in coefficients[-2::-1])
    helper += "    return 0.5*g + square*p;\n}\n\n"
    fn = functions(source)["nonlinear_product"]
    old = """        var activated: f32 = 0.0;
        if (g >= 0.0) { activated = g/(1.0 + math.exp(-g)); }
        else { const e = math.exp(g); activated = g*e/(1.0+e); }"""
    assert old in fn
    new = fn.replace(old, "        const activated: f32 = silu_polynomial(g);")
    return source.replace(fn, helper + new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualified_baseline", type=Path)
    args = parser.parse_args()
    baseline = load_validated(args.qualified_baseline)
    profile = get_profile(baseline["profile"]["id"])
    if profile["adapter"] != "resident_swiglu" or baseline["variant"] != "baseline":
        raise ValueError("Require a complete resident SwiGLU baseline")
    if baseline["provenance"]["evaluator_sha256"] != identity(evaluator_files(ROOT)):
        raise ValueError("Baseline is from a different evaluator; rebaseline first")
    if baseline["source_sha256"] != tree_digest(ROOT / "kernels/composed/resident-swiglu"):
        raise ValueError("Baseline CSL changed since qualification")
    for name in NAMES:
        if (ROOT / "experiments/candidates" / name).exists():
            raise ValueError("Refuse to overwrite a candidate")
    probe_path = ROOT / "runs/maintenance/swiglu-silu-primitive.json"
    probe = json.loads(probe_path.read_text())
    selected = [r for r in probe["results"] if r["degree"] == 9]
    if len(selected) != 1 or not selected[0]["primitive_gate_passed"]:
        raise ValueError("Require the prescribed exhaustive degree-nine math screen")
    base = (ROOT / "kernels/composed/resident-swiglu/src/meshgemm.csl").read_text()
    hypotheses = [
        (
            NAMES[0],
            projections(base),
            "resident_swiglu.projection-layout-dispatch",
            "Port the confirmed expansion row-vector/weight-descriptor layout and contraction array-base resets into every appropriate chain projection, preserving all H/F FP16 terms/order, timed output packing and stage/frame state.",
        ),
        (
            NAMES[1],
            receive_reuse(base),
            "resident_swiglu.receive-buffer-reuse",
            "Remove redundant X0/X1/W1 clearing: every logical X0 element is copied from the frame or computed V, and full even-sized receives overwrite X1/W1 before use. Padded words may retain prior values but never enter arithmetic. Canonical weight copies and complete state/output evidence remain fixed.",
        ),
        (
            NAMES[2],
            activation(base, selected[0]["coefficients"]),
            "resident_swiglu.fp32-silu-even-polynomial",
            "Evaluate SiLU using a degree-nine FP32 polynomial in g squared for |g|<=4, retaining overflow-safe SDK exp outside. Exhaustive finite FP16 gates passed a prescribed0.00025 host relative-error screen; this is not hardware proof. Preserve FP32 activation/product, FP16 V storage and the original full-chain bound.",
        ),
        (
            NAMES[3],
            activation(receive_reuse(projections(base)), selected[0]["coefficients"]),
            "resident_swiglu.projection-silu-buffer-composition",
            "Combine projection layout/descriptor dispatch, complete receive-buffer overwrite invariants and bounded FP32 SiLU polynomial. Every projection term, distinct frame, conversion, nonlinear/residual operation, intermediate and state store remains timed and verified under the unchanged contract.",
        ),
    ]
    for name, source, approach, hypothesis in hypotheses:
        folder = candidate_init(profile, name)
        manifest_path = folder / "candidate.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(
            shape_specific=False,
            approach=approach,
            hypothesis=hypothesis,
            falsification="Reject missing/reordered arithmetic, logical reads from stale padding, any six-case intermediate/output/state or shutdown failure, protected-control changes, speedup below1.05x or memory above1.10x. Full fresh paired seeds are mandatory. Cross-cell claims require complete suite and worst-cell guards; never relax the fixed bounds.",
            author="unknown",
            model="unknown",
            agent_harness="Codex",
            qualification_receipt=str(args.qualified_baseline),
            qualification_receipt_sha256=digest(args.qualified_baseline),
            primitive_math_probe_sha256=digest(probe_path),
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (folder / "src/meshgemm.csl").write_text(source)
        (folder / "README.md").write_text(
            "# " + name + "\n\n" + hypothesis + "\n\n" + manifest["falsification"] + "\n"
        )
        print(folder)


if __name__ == "__main__":
    main()
