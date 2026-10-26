"""Create a candidate only after the scaled recurrent contract is registered.

DSDs walk every FP32 state row. The only additional allocation is one V-element
FP32 scratch vector, reused for memory projection and readout. No trusted driver,
input, bound, timing function or state/history store is changed.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAME = "recurrent-gdn-vector-state"

VECTOR = """    var head: i16 = 0;
    while (head < R) : (head += 1) {
        const decay: f32 = decay_exp(G_CANON[frame*R+head]);
        A_WORK[head] = decay;
        var head_state = @set_dsd_length(state_dsd, K*V);
        head_state = @increment_dsd_offset(head_state, head*K*V, f32);
        @fmuls(head_state, head_state, decay);

        // Accumulate all K terms, in the original ascending row order.
        @fmovs(scratch_dsd, @as(f32, 0.0));
        var row = @set_dsd_length(head_state, V);
        i = 0;
        while (i < K) : (i += 1) {
            @fmacs(scratch_dsd, scratch_dsd, row, K_WORK[i]);
            row = @increment_dsd_offset(row, V, f32);
        }
        var delta = @set_dsd_length(delta_dsd, V);
        delta = @increment_dsd_offset(delta, head*V, f32);
        var value = @increment_dsd_offset(v_input_dsd, (frame*R+head)*V, fp);
        @fh2s(delta, value);
        @fmuls(scratch_dsd, scratch_dsd, @as(f32, -1.0));
        @fadds(delta, delta, scratch_dsd);
        @fmuls(delta, delta, BETA_CANON[frame*R+head]);

        row = @set_dsd_length(head_state, V);
        i = 0;
        while (i < K) : (i += 1) {
            @fmacs(row, row, delta, K_WORK[i]);
            row = @increment_dsd_offset(row, V, f32);
        }
        // The same scratch storage is dead after DELTA is complete.
        @fmovs(scratch_dsd, @as(f32, 0.0));
        row = @set_dsd_length(head_state, V);
        i = 0;
        while (i < K) : (i += 1) {
            @fmacs(scratch_dsd, scratch_dsd, row, Q_WORK[i]);
            row = @increment_dsd_offset(row, V, f32);
        }
        var output = @set_dsd_length(y_work_dsd, V);
        output = @increment_dsd_offset(output, head*V, fp);
        @fs2h(output, scratch_dsd);
    }
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--profile", default="recurrent-gdn.k16-v16")
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from cerebras_research.adapters import baseline_source
    from cerebras_research.catalog import get_profile
    from cerebras_research.paths import identity, tree_digest
    from cerebras_research.provenance import evaluator_files
    from cerebras_research.scoring import load_validated

    baseline = args.baseline.resolve()
    relative = str(baseline.relative_to(ROOT))
    qualified = load_validated(baseline)
    profile = get_profile(args.profile)
    source, _ = baseline_source(profile)
    if (
        profile["adapter"] not in ("recurrent_gdn", "recurrent_gdn_native")
        or qualified["profile"]["id"] != args.profile
        or qualified["variant"] != "baseline"
        or qualified["source_sha256"] != tree_digest(source)
        or qualified["provenance"]["evaluator_sha256"] != identity(evaluator_files(ROOT))
    ):
        raise ValueError("Need a current matched full baseline before candidate creation")
    if "var VECTOR_SCRATCH" in (source / "src/pe.csl").read_text():
        raise ValueError("Baseline already has vector state; use a refinement candidate")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "research.py"),
            "candidate-init",
            args.profile,
            args.name,
        ],
        cwd=ROOT,
        check=True,
    )
    folder = ROOT / "experiments/candidates" / args.name
    path = folder / "src/pe.csl"
    source = path.read_text()
    start = source.index("    var head: i16 = 0;", source.index("fn one_frame("))
    end = source.index("// Protected:", start)
    source = source[:start] + VECTOR + "\n" + source[end:]
    marker = "var Q_WORK: [K]f32;"
    scratch = """var VECTOR_SCRATCH: [V]f32;
var scratch_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{V} -> VECTOR_SCRATCH[i] });
var v_input_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{V} -> V_CANON[i] });
"""
    source = source.replace(marker, scratch + "\n" + marker)
    path.write_text(source)
    metadata = folder / "candidate.json"
    data = json.loads(metadata.read_text())
    data.update(
        approach="recurrent_gdn.fp32_vector_state_row_walk",
        hypothesis="Contiguous V-lane FP32 vector decay, memory projection, outer update and readout remove scalar address/loop overhead; one V-element scratch vector fits the fixed memory gate. Every ascending K term and all protected normalization/exp/state-history/timing work remain.",
        falsification="Reject if any of the six initial/fresh families fails the unchanged local/full-chain FP32/BF16 bounds or immutable/core/timer checks, if any projection term/state store is absent, if memory grows over 10 percent, or if matched full-case speedup is below 1.05x. A suite claim also requires both shapes and worst-cell guards.",
        author="OpenAI Codex",
        model="unknown",
        agent_harness="Codex",
        shape_specific=False,
        qualification_receipt=relative,
        qualification_receipt_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
    )
    metadata.write_text(json.dumps(data, indent=2) + "\n")
    (folder / "README.md").write_text("""# FP32 vector recurrent state candidate

One V-element scratch vector is first the memory projection and then the
readout accumulator. Every K term is accumulated in ascending row order; the
outer update writes every state element. Normalization, exp, immutable inputs,
initial-state copy, four token frames, complete delta/state/output histories,
final state and protected timing functions remain unchanged.

The candidate uses FP32 FMA vector instructions with the fixed original error
contract. It is an optimization of the repository's independent scalar CSL
reference, not a production Cerebras inference implementation. Require matched
six-case validation, fresh paired seed and complete two-shape coverage before a
cross-shape retention claim. See candidate.json for falsification gates.
""")
    print(folder)


if __name__ == "__main__":
    main()
