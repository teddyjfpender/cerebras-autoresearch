"""Bulk mem4d state arithmetic using dead current-frame history as FP32 scratch.

The repeat-reduction idiom is used in the pinned SDK25-point stencil. Four
partial V-vectors give distance between repeated accumulator addresses. Every
K*V product is executed; protected full-state stores overwrite temporary history
only after its last use. No timing/input/oracle/precision contract is changed.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cerebras_research.adapters import candidate_init, verify_timing_contract  # noqa: E402
from cerebras_research.catalog import get_profile  # noqa: E402
from cerebras_research.paths import identity, tree_digest  # noqa: E402
from cerebras_research.provenance import evaluator_files  # noqa: E402
from cerebras_research.scoring import load_validated  # noqa: E402

EXTRA = """
comptime { @comptime_assert(R == 1); @comptime_assert(K % 4 == 0); }
var all_scratch_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |j|{4*V} -> VECTOR_SCRATCH[j] });
var scratch1_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |j|{V} -> VECTOR_SCRATCH[V+j] });
var scratch2_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |j|{V} -> VECTOR_SCRATCH[2*V+j] });
var scratch3_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |j|{V} -> VECTOR_SCRATCH[3*V+j] });
var sum_repeat_dsd = @get_dsd(mem4d_dsd, .{ .tensor_access = |i,j|{K/4,4*V} -> VECTOR_SCRATCH[j] });
var key_broadcast_dsd = @get_dsd(mem4d_dsd, .{ .tensor_access = |i,j|{K,V} -> K_WORK[i] });
var query_broadcast_dsd = @get_dsd(mem4d_dsd, .{ .tensor_access = |i,j|{K,V} -> Q_WORK[i] });
var delta_broadcast_dsd = @get_dsd(mem4d_dsd, .{ .tensor_access = |i,j|{K,V} -> DELTA[j] });
var product_template_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |j|{K*V} -> STATE_HISTORY[j] });

fn reduce_four_partials() void {
    @fadds(scratch_dsd, scratch_dsd, scratch1_dsd);
    @fadds(scratch2_dsd, scratch2_dsd, scratch3_dsd);
    @fadds(scratch_dsd, scratch_dsd, scratch2_dsd);
}

"""
BODY = """    // The current frame's history has no reader until protected save_frame.
    // Past histories and canonical inputs are never used as scratch.
    var product = @increment_dsd_offset(product_template_dsd, frame*K*V, f32);
    const decay: f32 = decay_exp(G_CANON[frame]);
    A_WORK[0] = decay;
    @fmuls(state_dsd, state_dsd, decay);

    // Every memory-projection product, followed by four FP32 partial sums.
    @fmuls(product, state_dsd, key_broadcast_dsd);
    @fmovs(all_scratch_dsd, @as(f32,0.0));
    @fadds(sum_repeat_dsd, sum_repeat_dsd, product);
    reduce_four_partials();
    var value = @increment_dsd_offset(v_input_dsd, frame*V, fp);
    @fh2s(delta_dsd, value);
    @fmuls(scratch_dsd, scratch_dsd, @as(f32,-1.0));
    @fadds(delta_dsd, delta_dsd, scratch_dsd);
    @fmuls(delta_dsd, delta_dsd, BETA_CANON[frame]);

    // The outer product and state addition stay FP32, with the fixed gamma2 bound.
    @fmuls(product, key_broadcast_dsd, delta_broadcast_dsd);
    @fadds(state_dsd, state_dsd, product);

    @fmuls(product, state_dsd, query_broadcast_dsd);
    @fmovs(all_scratch_dsd, @as(f32,0.0));
    @fadds(sum_repeat_dsd, sum_repeat_dsd, product);
    reduce_four_partials();
    @fs2h(y_work_dsd, scratch_dsd);
    // product is now dead; save_frame overwrites the entire slot with STATE.
}

"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--parent", default="native-gdn-vector-norm")
    parser.add_argument("--name", default="native-gdn-bulk-state-dsd")
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    qualified = load_validated(baseline)
    profile = get_profile(qualified["profile"]["id"])
    if (
        profile["adapter"] != "recurrent_gdn_native"
        or qualified["variant"] != "baseline"
        or qualified["provenance"]["evaluator_sha256"] != identity(evaluator_files(ROOT))
    ):
        raise ValueError("Current full native baseline required")
    parent = ROOT / "experiments/candidates" / args.parent
    metadata = json.loads((parent / "candidate.json").read_text())
    if metadata["baseline_tree_sha256"] != qualified["baseline_tree_sha256"]:
        raise ValueError("Parent belongs to a different vector baseline")
    folder = candidate_init(profile, args.name)
    for path in parent.rglob("*.csl"):
        target = folder / path.relative_to(parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    path = folder / "src/pe.csl"
    original = path.read_text()
    source = original.replace("var VECTOR_SCRATCH: [V]f32;", "var VECTOR_SCRATCH: [4*V]f32;")
    source = source.replace("fn decay_exp(", EXTRA + "fn decay_exp(", 1)
    start = source.index("    var head: i16 = 0;", source.index("fn one_frame("))
    end = source.index("// Protected:", start)
    source = source[:start] + BODY + source[end:]
    verify_timing_contract(original, source, "recurrent_gdn_native")
    path.write_text(source)
    p = folder / "candidate.json"
    data = json.loads(p.read_text())
    pattern = ROOT / ".deps/sdk-examples/benchmarks/25-pt-stencil/task.csl"
    data.update(
        approach="native_gdn.bulk_mem4d_fp32_state",
        hypothesis="Replace all three K-row state loops with bulk FP32 vector products, repeat-DSD four-partial reductions and pointwise outer update. Reuse only the dead current-frame STATE_HISTORY slot as product scratch; protected save_frame still writes every final state. Execute every product with FP32 MUL/ADD under the original gamma2K/gamma2 bounds; original BF16 output conversion, exact canonical/replica checks and all frames remain.",
        falsification="Reject incorrect mem4d flattening/reduction, any omitted K*V term/frame/state store, reuse of a past history or live/canonical buffer, scratch extent outside the current frame, changed precision or timing, fixed local/full-chain error failure, shutdown/core/timer failure, speedup below1.05x or memory above1.10x. Require full six-case fresh pairs and exact suite coverage for widening claims.",
        author="unknown",
        model="unknown",
        agent_harness="Codex",
        shape_specific=False,
        parent_candidate=args.parent,
        parent_source_sha256=tree_digest(parent),
        qualification_receipt=str(baseline.relative_to(ROOT)),
        qualification_receipt_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        sdk_pattern_source="sdk-examples/benchmarks/25-pt-stencil/task.csl:reduceBuffer",
        sdk_pattern_file_sha256=hashlib.sha256(pattern.read_bytes()).hexdigest(),
    )
    p.write_text(json.dumps(data, indent=2) + "\n")
    (folder / "README.md").write_text("""# Bulk FP32 state dataflow

The repeat-DSD accumulator pattern is present in Cerebras SDK25-point stencil's
reduceBuffer. This candidate evaluates all K*V products using FP32 multiply/add,
then four partial sums. It removes scalar row-loop/descriptor setup overhead.
The unchanged local/full-chain oracle must accept every rounding difference.

| Phase | Current-frame history slot | State and evidence |
| --- | --- | --- |
| Entry | Dead; previous frames untouched | Live previous STATE, immutable S0/QKV/g/beta |
| Memory projection | All decayed-state times key products | Every K term reduced through4 partial V-vectors |
| Outer update | All key times delta products | Every FP32 STATE element updated |
| Readout | All updated-state times query products | Complete BF16 output, original conversion |
| Protected save_frame | Entire slot overwritten with STATE | All prefix/delta/output/state snapshots preserved |

All operations are blocking. No future work escapes the protected timer and no
warmup/cache output is used. R=1 is a physical-column-shard constraint; logical
value heads remain complete in the fixed native layout. No performance result
exists until an actual SDK attempt, full/fresh validation and artifact replay.
""")
    print(folder)


if __name__ == "__main__":
    main()
