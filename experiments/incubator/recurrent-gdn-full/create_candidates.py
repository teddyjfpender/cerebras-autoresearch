"""Freeze native normalization and fused readout transfers for the full-head contract."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cerebras_research.adapters import candidate_init, verify_timing_contract  # noqa: E402
from cerebras_research.catalog import get_profile  # noqa: E402
from cerebras_research.paths import tree_digest  # noqa: E402


def main():
    profile = get_profile("recurrent-gdn-full.k128-v128-h16-t2")
    parent = ROOT / "experiments/candidates/native-gdn-vector-norm"
    fused = ROOT / "experiments/candidates/native-gdn-fuse-readout/src/pe.csl"
    native = (parent / "src/pe.csl").read_text()
    source = native.replace("(T == 4 and V == 8) or (T == 8 and V == 4)", "T == 2 and V == 8")
    # Only the two complete state-update/readout loops differ. Their ascending-K
    # FMA order per output and every full-state save remain unchanged.
    start = source.index(
        "        row = @set_dsd_length(head_state, V);", source.index("@fmuls(delta, delta")
    )
    end = source.index("        var output", start)
    fs = fused.read_text()
    a = fs.index("        // DELTA is complete")
    b = fs.index("        var output", a)
    variants = {
        "full-gdn-vector-norm": source,
        "full-gdn-norm-fused-readout": source[:start] + fs[a:b] + source[end:],
    }
    baseline = (ROOT / "kernels/attention/recurrent-gdn-full/src/pe.csl").read_text()
    for name, text in variants.items():
        folder = candidate_init(profile, name)
        verify_timing_contract(baseline, text, "recurrent_gdn_full")
        (folder / "src/pe.csl").write_text(text)
        p = folder / "candidate.json"
        data = json.loads(p.read_text())
        data.update(
            approach="full_gdn.normalization_reuse" + ("_fused_readout" if "fused" in name else ""),
            hypothesis="Transfer exact BF16-to-FP32 conversion reuse and vector normalization scaling to all16 key/48 value heads. "
            + (
                "Fuse each updated-state row with its ascending-K readout, preserving every FMA and state snapshot."
                if "fused" in name
                else "Keep the original state-update/readout loops and ascending-K arithmetic."
            ),
            falsification="Reject any missing group, value head, K/V term, token, state/history store or immutable replica; changed precision, timer or fixed bounds; shutdown failure; speedup below1.05x or memory growth above1.10x. Require all six families and a fresh matched seed; prior one-group results do not qualify this workload.",
            author="unknown",
            model="unknown",
            agent_harness="Codex",
            parent_candidate=parent.name,
            parent_tree_sha256=tree_digest(parent),
            fused_source_sha256=hashlib.sha256(fused.read_bytes()).hexdigest(),
            transfer_scope="Native operator only. No model TPS result.",
        )
        p.write_text(json.dumps(data, indent=2) + "\n")
        (folder / "README.md").write_text(
            "# Full native recurrent transfer\n\nFrozen before any full-head device result. All16 key heads,48 value heads,\nnative K128/V128 and both complete stateful tokens remain required.\nThe normalized operands and original FMA ordering are preserved.\n"
        )
        print(folder)


if __name__ == "__main__":
    main()
