"""Freeze the native persistent-DSR unroll refinements for full native heads."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cerebras_research.adapters import candidate_init, verify_timing_contract  # noqa: E402
from cerebras_research.catalog import get_profile  # noqa: E402
from cerebras_research.paths import tree_digest  # noqa: E402


def main():
    reference = (ROOT / "kernels/attention/recurrent-gdn-full/src/pe.csl").read_text()
    for factor in (4, 8):
        parent = ROOT / f"experiments/candidates/native-gdn-pr1-dsr-unroll{factor}"
        source = (
            (parent / "src/pe.csl")
            .read_text()
            .replace("(T == 4 and V == 8) or (T == 8 and V == 4)", "T == 2 and V == 8")
        )
        name = f"full-gdn-pr1-dsr-unroll{factor}"
        folder = candidate_init(get_profile("recurrent-gdn-full.k128-v128-h16-t2"), name)
        verify_timing_contract(reference, source, "recurrent_gdn_full")
        (folder / "src/pe.csl").write_text(source)
        p = folder / "candidate.json"
        data = json.loads(p.read_text())
        old = json.loads((parent / "candidate.json").read_text())
        for key in (
            "hypothesis",
            "falsification",
            "author",
            "model",
            "agent_harness",
            "reference_pr",
            "reference_commit",
            "reference_source_sha256",
            "unroll_factor",
        ):
            data[key] = old[key]
        data.update(
            approach=f"full_gdn.pr1_dsr_bank_unroll{factor}",
            parent_candidate=parent.name,
            parent_tree_sha256=tree_digest(parent),
            scope="Full16 key/48 value heads,K128/V128,T2 on the fixed768-PE contract. Native-source compatibility supplies no full-head execution proof; new full/fresh controls and complete state/timer replay required.",
        )
        p.write_text(json.dumps(data, indent=2) + "\n")
        (folder / "README.md").write_text(
            "# Full native-head DSR unroll transfer\n\nAll16 groups and48 value heads remain. This transfers only bounded loop\nunrolling and persistent DSR row addresses with a bank-offset accumulator.\nEvery ascending FP32 FMA, normalized operand, history and timer stays fixed.\nThe full-head runtime must be qualified before any retained result.\n"
        )
        print(folder)


if __name__ == "__main__":
    main()
