"""Create bounded-exp tail alternatives after the exhaustive BF16 host screen."""

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
    parent = ROOT / "experiments/candidates/bf16-swiglu-vector-scratch-dispatch"
    manifest = json.loads((parent / "candidate.json").read_text())
    baseline_path = ROOT / manifest["qualification_receipt"]
    baseline = load_validated(baseline_path)
    if baseline["provenance"]["evaluator_sha256"] != identity(evaluator_files(ROOT)):
        raise ValueError("Require a matched full baseline")
    canonical = (ROOT / "kernels/composed/resident-swiglu-bf16/src/meshgemm.csl").read_text()
    source = (parent / "src/meshgemm.csl").read_text()
    probe_path = ROOT / "runs/maintenance/bf16-silu-tail-primitive.json"
    probe = json.loads(probe_path.read_text())
    for degree in (4, 5):
        name = f"bf16-swiglu-vector-scratch-tail{degree}"
        if (ROOT / "experiments/candidates" / name).exists():
            raise ValueError("Refuse to overwrite a candidate")
    for degree in (4, 5):
        row = next(r for r in probe["results"] if r["degree"] == degree)
        if not row["primitive_gate_passed"]:
            raise ValueError("Require the prescribed exhaustive tail screen")
        coefficients = row["coefficients"]
        helper = """// FP32 Taylor exp(-abs(g)/16), then four squares, only for 4<abs(g)<=8.
// Native overflow-safe exp remains the fallback outside the screened interval.
fn silu_bounded_tail(g: f32) f32 {
    var magnitude: f32 = g;
    if (g < 0.0) { magnitude = -g; }
    const x: f32 = -magnitude/16.0;
"""
        helper += f"    var e: f32 = {coefficients[-1]!r};\n"
        helper += "".join(f"    e = e*x + {c!r};\n" for c in coefficients[-2::-1])
        helper += "    e = e*e;\n" * 4
        helper += "    if (g >= 0.0) { return g/(1.0+e); }\n    return g*e/(1.0+e);\n}\n\n"
        original = functions(source)["silu_polynomial"]
        needle = "    if ((g < -4.0) or (g > 4.0)) {\n"
        assert original.count(needle) == 1
        edited = original.replace(
            needle,
            needle + "        if ((g >= -8.0) and (g <= 8.0)) { return silu_bounded_tail(g); }\n",
        )
        updated = source.replace(original, helper + edited)
        verify_timing_contract(canonical, updated, "resident_swiglu_bf16")
        name = f"bf16-swiglu-vector-scratch-tail{degree}"
        folder = candidate_init(get_profile(baseline["profile"]["id"]), name)
        path = folder / "candidate.json"
        result = json.loads(path.read_text())
        hypothesis = f"Extend the vector/scratch dispatcher with a degree-{degree} FP32 Taylor exp tail for 4<abs(g)<=8, four repeated squares and stable SiLU division. Keep degree-six central SiLU and native exp outside abs(g)>8. An exhaustive separate BF16 gate screen passed with one quarter of the unchanged local allowance reserved; require full device/fresh validation. This targets costly positive-tail exp fallbacks without weakening precision or allocating new scratch."
        result.update(
            shape_specific=False,
            approach=f"resident_swiglu_bf16.vector-silu6-scratch-tail{degree}",
            hypothesis=hypothesis,
            falsification=manifest["falsification"],
            author="unknown",
            model="unknown",
            agent_harness="Codex",
            qualification_receipt=manifest["qualification_receipt"],
            qualification_receipt_sha256=digest(baseline_path),
            parent_source_sha256=tree_digest(parent),
            parent_manifest_sha256=digest(parent / "candidate.json"),
            primitive_math_probe_sha256=digest(probe_path),
        )
        path.write_text(json.dumps(result, indent=2) + "\n")
        (folder / "src/meshgemm.csl").write_text(updated)
        (folder / "README.md").write_text(
            "# " + name + "\n\n" + hypothesis + "\n\n" + result["falsification"] + "\n"
        )
        print(folder)


if __name__ == "__main__":
    main()
