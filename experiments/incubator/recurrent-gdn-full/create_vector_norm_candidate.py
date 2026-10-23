"""Remove the native normalization row loop using FP32 sixteen-partial sums."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cerebras_research.adapters import candidate_init, verify_timing_contract  # noqa: E402
from cerebras_research.catalog import get_profile  # noqa: E402
from cerebras_research.paths import tree_digest  # noqa: E402

HELPER = """var norm_parts_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |j|{16} -> VECTOR_SCRATCH[j] });

fn squared_norm(frame: i16, query: bool) f32 {
    // Current-frame history is dead until protected save_frame. Only K words
    // are used here; the complete state slot is subsequently overwritten.
    var products = @set_dsd_length(product_template_dsd, K);
    products = @increment_dsd_offset(products, frame*K*V, f32);
    const dst = @get_dsr(dsr_dest, 0);
    const src0 = @get_dsr(dsr_src0, 0);
    const src1 = @get_dsr(dsr_src1, 0);
    @load_to_dsr(dst, products);
    if (query) {
        @load_to_dsr(src0, q_work_dsd);
        @load_to_dsr(src1, q_work_dsd);
    } else {
        @load_to_dsr(src0, k_work_dsd);
        @load_to_dsr(src1, k_work_dsd);
    }
    @fmuls(dst, src0, src1);
    @fmovs(norm_parts_dsd, @as(f32,0.0));
    const sums = @get_dsd(mem4d_dsd, .{ .tensor_access = |i,j|{K/16,16} -> VECTOR_SCRATCH[j] });
    @load_to_dsr_xdsr_sr(dst, @get_xdsr(0), .{}, sums);
    @load_to_dsr_xdsr_sr(src0, @get_xdsr(1), .{}, sums);
    @load_to_dsr(src1, products);
    @fadds(dst, src0, src1);
    return ((((VECTOR_SCRATCH[0]+VECTOR_SCRATCH[1])+(VECTOR_SCRATCH[2]+VECTOR_SCRATCH[3]))+((VECTOR_SCRATCH[4]+VECTOR_SCRATCH[5])+(VECTOR_SCRATCH[6]+VECTOR_SCRATCH[7])))+(((VECTOR_SCRATCH[8]+VECTOR_SCRATCH[9])+(VECTOR_SCRATCH[10]+VECTOR_SCRATCH[11]))+((VECTOR_SCRATCH[12]+VECTOR_SCRATCH[13])+(VECTOR_SCRATCH[14]+VECTOR_SCRATCH[15]))));
}

"""


def main():
    parent = ROOT / "experiments/candidates/full-gdn-bulk-extended-dsr"
    folder = candidate_init(
        get_profile("recurrent-gdn-full.k128-v128-h16-t2"), "full-gdn-bulk-vector-norm"
    )
    source = (parent / "src/pe.csl").read_text()
    s = source.replace("fn normalize_frame(", HELPER + "fn normalize_frame(", 1)
    a = s.index("    var qs:", s.index("fn normalize_frame("))
    b = s.index("    const qi:", a)
    s = (
        s[:a]
        + "    const qs: f32 = squared_norm(frame, true);\n    const ks: f32 = squared_norm(frame, false);\n"
        + s[b:]
    )
    verify_timing_contract(source, s, "recurrent_gdn_full")
    (folder / "src/pe.csl").write_text(s)
    p = folder / "candidate.json"
    data = json.loads(p.read_text())
    old = json.loads((parent / "candidate.json").read_text())
    for k in ("author", "model", "agent_harness", "falsification"):
        data[k] = old[k]
    data.update(
        approach="full_gdn.bulk_state_and_vector_norm",
        hypothesis="Evaluate every raw FP32 Q/K square using vector multiply and sixteen independent ascending partial sums, then the original sqrt/epsilon/scaling. Combine with complete bulk FP32 state arithmetic. Reuse only the dead current-frame state-history slot as product scratch; original normalization/state error envelopes, BF16 boundaries, every token/group/store and fixed timing remain.",
        parent_candidate=parent.name,
        parent_tree_sha256=tree_digest(parent),
    )
    p.write_text(json.dumps(data, indent=2) + "\n")
    (folder / "README.md").write_text(
        "# Bulk state and vector norm\n\nEvery normalization square and sum is FP32. Sixteen independent partial sums and\na four-level FP32 tree replace the scalar128-row loop. The original fixed\nnormalization and propagated state bounds must accept all rounding differences.\nAll current-frame history scratch is overwritten before protected storage.\n"
    )
    print(folder)


if __name__ == "__main__":
    main()
