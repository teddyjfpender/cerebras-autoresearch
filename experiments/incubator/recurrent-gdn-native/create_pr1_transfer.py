"""Qualify PR1-derived bank/DSR reuse on the established native T4 operator."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cerebras_research.adapters import candidate_init, verify_timing_contract  # noqa: E402
from cerebras_research.catalog import get_profile  # noqa: E402
from cerebras_research.paths import tree_digest  # noqa: E402


def main():
    profile = get_profile("recurrent-gdn-native.k128-v128-t4")
    baseline = (ROOT / "kernels/attention/recurrent-gdn-native/src/pe.csl").read_text()
    for source_name, name in [
        ("full-gdn-bank-offset", "native-gdn-pr1-bank-offset"),
        ("full-gdn-dsr-bank-offset", "native-gdn-pr1-dsr-bank-offset"),
    ]:
        parent = ROOT / "experiments/candidates" / source_name
        source = (
            (parent / "src/pe.csl")
            .read_text()
            .replace("T == 2 and V == 8", "(T == 4 and V == 8) or (T == 8 and V == 4)")
        )
        folder = candidate_init(profile, name)
        verify_timing_contract(baseline, source, "recurrent_gdn_native")
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
        ):
            data[key] = old[key]
        data.update(
            approach="native_gdn.pr1_bank_offset" + ("_dsr_reuse" if "dsr" in name else ""),
            parent_candidate=source_name,
            parent_tree_sha256=tree_digest(parent),
            scope="Current experiment targets the fixed one-group native T4 profile only. Source compatibility is not T8 or full-head execution proof. New matched evaluator baselines and fresh six-case pairs required.",
        )
        p.write_text(json.dumps(data, indent=2) + "\n")
        (folder / "README.md").write_text(
            "# Native-width PR1 transfer\n\nThis applies the frozen bank-offset/DSR row-reuse idea to the existing native\nK128/V128, three-value-head T4 contract. Every ascending FP32 FMA, normalized\noperand, full state and protected timer/store remains. Claims require new\nmatched full six-case and fresh pairs. T8/full-head results are separate.\n"
        )
        print(folder)


if __name__ == "__main__":
    main()
