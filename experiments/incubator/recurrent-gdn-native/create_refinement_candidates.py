"""Preregister native refinements against a qualified FP32 vector reference.

Each mode changes one dataflow idea. Protected sequence/history functions,
precision, all input/state terms, error bounds, PE budgets and timing stay fixed.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cerebras_research.adapters import baseline_source  # noqa: E402
from cerebras_research.catalog import get_profile  # noqa: E402
from cerebras_research.paths import identity, tree_digest  # noqa: E402
from cerebras_research.provenance import evaluator_files  # noqa: E402
from cerebras_research.scoring import load_validated  # noqa: E402


def fuse_readout(source):
    start = source.index(
        "        row = @set_dsd_length(head_state, V);", source.index("@fmuls(delta, delta,")
    )
    end = source.index("        var output = ", start)
    replacement = """        // DELTA is complete; memory-projection scratch is now dead.
        @fmovs(scratch_dsd, @as(f32, 0.0));
        row = @set_dsd_length(head_state, V);
        i = 0;
        while (i < K) : (i += 1) {
            @fmacs(row, row, delta, K_WORK[i]);
            // Read each updated row immediately, in ascending K order.
            @fmacs(scratch_dsd, scratch_dsd, row, Q_WORK[i]);
            row = @increment_dsd_offset(row, V, f32);
        }
"""
    return source[:start] + replacement + source[end:]


def vector_norm(source):
    globals = """var q_input_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |j|{K} -> Q_CANON[j] });
var k_input_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |j|{K} -> K_CANON[j] });
"""
    source = source.replace("var Q_WORK: [K]f32;", globals + "\nvar Q_WORK: [K]f32;")
    prefix = """    // Exact BF16->FP32 conversion once; both the norm and scaling reuse it.
    var q_input = @increment_dsd_offset(q_input_dsd, frame*K, fp);
    var k_input = @increment_dsd_offset(k_input_dsd, frame*K, fp);
    @fh2s(q_work_dsd, q_input);
    @fh2s(k_work_dsd, k_input);
"""
    source = source.replace(
        "fn one_frame(frame: i16) void {\n", "fn one_frame(frame: i16) void {\n" + prefix
    )
    source = source.replace("@as(f32, Q_CANON[frame*K+i])", "Q_WORK[i]").replace(
        "@as(f32, K_CANON[frame*K+i])", "K_WORK[i]"
    )
    start = source.index("    i = 0;", source.index("    const scale: f32 ="))
    end = source.index("    var head:", start)
    replacement = """    // Preserve the two FP32 query scaling steps and the key scaling step.
    @fmuls(q_work_dsd, q_work_dsd, qi);
    @fmuls(q_work_dsd, q_work_dsd, scale);
    @fmuls(k_work_dsd, k_work_dsd, ki);
"""
    return source[:start] + replacement + source[end:]


def partial4(source):
    source = source.replace(
        "// Accumulate all K terms, in the original ascending row order.",
        "// Evaluate every K product in ascending rows; use four FP32 partial sums.",
    )
    source = source.replace(
        "var VECTOR_SCRATCH: [V]f32;",
        "var VECTOR_SCRATCH: [4*V]f32;\ncomptime { @comptime_assert(K % 4 == 0); }",
    )
    extra = """var all_scratch_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |j|{4*V} -> VECTOR_SCRATCH[j] });
"""
    for lane in (1, 2, 3):
        extra += f"var scratch{lane}_dsd = @get_dsd(mem1d_dsd, .{{ .tensor_access = |j|{{V}} -> VECTOR_SCRATCH[{lane}*V+j] }});\n"
    source = source.replace("var Q_WORK: [K]f32;", extra + "\nvar Q_WORK: [K]f32;")
    source = source.replace(
        "@fmovs(scratch_dsd, @as(f32, 0.0));", "@fmovs(all_scratch_dsd, @as(f32, 0.0));"
    )
    for operand in ("K_WORK", "Q_WORK"):
        original = f"""        while (i < K) : (i += 1) {{
            @fmacs(scratch_dsd, scratch_dsd, row, {operand}[i]);
            row = @increment_dsd_offset(row, V, f32);
        }}"""
        lines = ["        while (i < K) : (i += 4) {"]
        for lane in range(4):
            target = "scratch_dsd" if lane == 0 else f"scratch{lane}_dsd"
            index = "i" if lane == 0 else f"i+{lane}"
            lines += [
                f"            @fmacs({target}, {target}, row, {operand}[{index}]);",
                "            row = @increment_dsd_offset(row, V, f32);",
            ]
        lines += [
            "        }",
            "        @fadds(scratch_dsd, scratch_dsd, scratch1_dsd);",
            "        @fadds(scratch2_dsd, scratch2_dsd, scratch3_dsd);",
            "        @fadds(scratch_dsd, scratch_dsd, scratch2_dsd);",
        ]
        if source.count(original) != 1:
            raise ValueError("Expected one complete dot loop")
        source = source.replace(original, "\n".join(lines))
    return source


MODES = {
    "fuse-readout": (
        fuse_readout,
        "native_gdn.fused_state_update_readout",
        "Immediately accumulate the BF16-output readout after each FP32 state row is updated. Preserve ascending K order and every full state store, while removing one separate row-walk loop.",
    ),
    "vector-norm": (
        vector_norm,
        "native_gdn.convert_once_vector_normalization",
        "Convert each canonical BF16 query/key once into FP32 work arrays, reuse exact converted values in the unchanged scalar squared-norm reductions, then apply the original query/key scaling with vector FP32 multiplies. Remove redundant conversions and scalar scaling loops.",
    ),
    "partial4": (
        partial4,
        "native_gdn.four_partial_fp32_dots",
        "Interleave four independent FP32 V-lane partial accumulators for memory projection and readout, then a three-add tree. Every K product is still evaluated; changed FP32 association must pass the unchanged local/full-chain bounds. Extra scratch is only3*V FP32 values, with code/descriptor growth subject to the original memory gate.",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--name")
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    relative = str(baseline.relative_to(ROOT))
    qualified = load_validated(baseline)
    profile = get_profile(qualified["profile"]["id"])
    origin, _ = baseline_source(profile)
    if (
        profile["adapter"] != "recurrent_gdn_native"
        or qualified["variant"] != "baseline"
        or qualified["source_sha256"] != tree_digest(origin)
        or qualified["provenance"]["evaluator_sha256"] != identity(evaluator_files(ROOT))
    ):
        raise ValueError("Need a current full native vector-reference baseline")
    name = args.name or "native-gdn-" + args.mode
    subprocess.run(
        [sys.executable, str(ROOT / "research.py"), "candidate-init", profile["id"], name],
        cwd=ROOT,
        check=True,
    )
    folder = ROOT / "experiments/candidates" / name
    source = folder / "src/pe.csl"
    change, approach, hypothesis = MODES[args.mode]
    source.write_text(change(source.read_text()))
    manifest = folder / "candidate.json"
    data = json.loads(manifest.read_text())
    data.update(
        approach=approach,
        hypothesis=hypothesis,
        falsification="Reject any omitted K/state/normalization term, changed precision/control/timed history, canonical or replicated-prefix drift, fixed local/full-chain error failure, shutdown/core/timer mismatch, speedup below1.05x or program-memory growth above1.10x. Require all six families, fresh paired seed, both complete native profiles and worst-cell guards before a suite claim.",
        author="unknown",
        model="unknown",
        agent_harness="Codex",
        shape_specific=False,
        qualification_receipt=relative,
        qualification_receipt_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
    )
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    (folder / "README.md").write_text(
        "# Native FP32 recurrent refinement\n\n"
        + hypothesis
        + "\n\nThe starting reference already uses FP32 vector state arithmetic. Every token,\nfull state, intermediate, canonical input and protected timer/history function\nis preserved. See candidate.json for the preregistered falsification gates.\n"
    )
    print(folder)


if __name__ == "__main__":
    main()
