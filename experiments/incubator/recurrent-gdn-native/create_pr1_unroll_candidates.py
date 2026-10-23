"""Bounded unrolling of persistent ascending-K FP32 FMA row walks."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cerebras_research.adapters import candidate_init, verify_timing_contract  # noqa: E402
from cerebras_research.catalog import get_profile  # noqa: E402
from cerebras_research.paths import tree_digest  # noqa: E402


def main():
    parent = ROOT / "experiments/candidates/native-gdn-pr1-dsr-bank-offset"
    original = (parent / "src/pe.csl").read_text()
    for factor in (4, 8):
        source = original
        for scalar in ("K_WORK", "Q_WORK"):
            old = (
                """        while (i < K) : (i += 1) {
            @fmacs(compute_dest, compute_src0, compute_src1, """
                + scalar
                + """[i]);
        }"""
            )
            count = source.count(old)
            if count != (2 if scalar == "K_WORK" else 1):
                raise ValueError("Parent FMA walks changed")
            lines = [f"        while (i < K) : (i += {factor}) {{"]
            for j in range(factor):
                index = "i" if j == 0 else f"i+{j}"
                lines.append(
                    f"            @fmacs(compute_dest, compute_src0, compute_src1, {scalar}[{index}]);"
                )
            lines.append("        }")
            source = source.replace(old, "\n".join(lines))
        source = source.replace(
            "const compute_dest =",
            f"comptime {{ @comptime_assert(K % {factor} == 0); }}\nconst compute_dest =",
            1,
        )
        name = f"native-gdn-pr1-dsr-unroll{factor}"
        folder = candidate_init(get_profile("recurrent-gdn-native.k128-v128-t4"), name)
        verify_timing_contract(original, source, "recurrent_gdn_native")
        (folder / "src/pe.csl").write_text(source)
        path = folder / "candidate.json"
        data = json.loads(path.read_text())
        old = json.loads((parent / "candidate.json").read_text())
        for key in (
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
            approach=f"native_gdn.pr1_dsr_bank_unroll{factor}",
            hypothesis=f"Unroll each persistent DSR FMA walk by{factor}. Keep every K term in the identical ascending order with unchanged FP32 FMA, normalization/exp and complete state/history. Reduce loop branch/index overhead while preserving pointer progress and bank offset.",
            parent_candidate=parent.name,
            parent_tree_sha256=tree_digest(parent),
            unroll_factor=factor,
            scope="Fixed native T4 contract. No T8/full-head gain is established by source compatibility. Fresh matched six-case pairs required.",
        )
        path.write_text(json.dumps(data, indent=2) + "\n")
        (folder / "README.md").write_text(
            f"# Persistent DSR walk unroll{factor}\n\nThe three K-row loops execute exactly128 FP32 FMAs apiece, in unchanged\nascending order. Dedicated DSRs advance rows automatically; fixed accumulator\nand delta addresses reset after every vector instruction. Only loop overhead\nis reduced. All full states, timing, precision and original bounds remain.\n"
        )
        print(folder)


if __name__ == "__main__":
    main()
