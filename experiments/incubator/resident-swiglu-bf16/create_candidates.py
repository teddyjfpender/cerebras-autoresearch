"""Create BF16 hypotheses only from a qualified, current full baseline."""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cerebras_research.adapters import candidate_init, verify_timing_contract  # noqa: E402
from cerebras_research.catalog import get_profile  # noqa: E402
from cerebras_research.paths import digest, identity, tree_digest  # noqa: E402
from cerebras_research.provenance import evaluator_files  # noqa: E402
from cerebras_research.scoring import load_validated  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "fp16_hypotheses", ROOT / "experiments/incubator/resident-swiglu-chain/create_candidates.py"
)
fp16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp16)


def projections(base):
    # Reuse the measured algorithm, with BF16 input offsets and FP32 output
    # descriptors/packing. Neither accumulation nor stage rounding is reduced.
    source = fp16.projections(base.replace("@fmovs(res_dsd, 0.0)", "@fmovh(res_dsd, 0.0)"))
    source = re.sub(r"\bf16\b", "fp", source).replace("@fmach(", "@fmachs(")
    source = source.replace("row_accum: [Mt*Ft]fp", "row_accum: [Mt*Ft]f32")
    source = source.replace("row_output_dsd, row*Ft, fp", "row_output_dsd, row*Ft, f32")
    source = source.replace("packed_output_dsd, row, fp", "packed_output_dsd, row, f32")
    source = source.replace("@fmovh(destination, source)", "@fmovs(destination, source)")
    source = source.replace("@fmovh(row_accum_dsd, 0.0)", "@fmovs(row_accum_dsd, 0.0)")
    return source.replace("@fmovh(res_dsd, 0.0)", "@fmovs(res_dsd, 0.0)")


def activation(base, coefficients, degree):
    source = fp16.activation(base, coefficients)
    return source.replace("Degree-nine", f"Degree-{degree}").replace(
        "Every finite FP16 gate in that range passed the prescribed host math screen.",
        "Every finite BF16 gate in that range passed the separate host math screen.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualified_baseline", type=Path)
    args = parser.parse_args()
    baseline = load_validated(args.qualified_baseline)
    profile = get_profile(baseline["profile"]["id"])
    if profile["adapter"] != "resident_swiglu_bf16" or baseline["variant"] != "baseline":
        raise ValueError("Require a complete BF16 resident SwiGLU baseline")
    if baseline["provenance"]["evaluator_sha256"] != identity(evaluator_files(ROOT)):
        raise ValueError("Require the same evaluator as the qualified baseline")
    canonical = ROOT / "kernels/composed/resident-swiglu-bf16"
    if baseline["source_sha256"] != tree_digest(canonical):
        raise ValueError("Baseline source changed since qualification")
    probe_path = ROOT / "runs/maintenance/bf16-silu-primitive.json"
    probe = json.loads(probe_path.read_text())
    coefficients = {}
    for degree in (6, 9):
        row = next(r for r in probe["results"] if r["degree"] == degree)
        if not row["primitive_gate_passed"]:
            raise ValueError("Require the prescribed BF16 primitive screen")
        coefficients[degree] = row["coefficients"]
    base = (canonical / "src/meshgemm.csl").read_text()
    hypotheses = [
        (
            "bf16-swiglu-projection-dispatch",
            projections(base),
            "resident_swiglu_bf16.projection-layout-dispatch",
            "Use contiguous FP32 row accumulators and hoisted BF16 weight descriptors for wide expansions, with timed FP32 packing and array-base resets on shorter projections. Preserve all mixed-FMA terms/order, BF16 stage rounding and complete state.",
        ),
        (
            "bf16-swiglu-silu-degree9",
            activation(base, coefficients[9], 9),
            "resident_swiglu_bf16.fp32-silu-degree9",
            "Use a degree-nine FP32 polynomial in g squared on |g|<=4 and stable SDK exp outside; preserve quantized BF16 G/U operands and V storage. The separate exhaustive BF16 host screen is preliminary; the fixed full-chain oracle remains mandatory.",
        ),
        (
            "bf16-swiglu-silu-degree6",
            activation(base, coefficients[6], 6),
            "resident_swiglu_bf16.fp32-silu-degree6",
            "Reduce the FP32 SiLU polynomial to degree six using the separate exhaustive BF16 gate screen, reserving one quarter of the original 4*u_b local allowance after normal product rounding. Retain stable exp outside |g|<=4, FP32 arithmetic and every BF16 stage boundary; no bounds change.",
        ),
        (
            "bf16-swiglu-projection-silu6-reuse",
            activation(fp16.receive_reuse(projections(base)), coefficients[6], 6),
            "resident_swiglu_bf16.projection-silu6-buffer-composition",
            "Combine FP32 row/column projection dispatch, bounded FP32 degree-six SiLU and complete receive-buffer overwrite invariants. Every logical X0 word is copied and received X1/W1 words are overwritten before use; stale padding never enters arithmetic. All mixed FMA terms, stage rounding, outputs and state remain timed.",
        ),
    ]
    for name, _, _, _ in hypotheses:
        if (ROOT / "experiments/candidates" / name).exists():
            raise ValueError("Refuse to overwrite a candidate")
    for name, source, approach, hypothesis in hypotheses:
        verify_timing_contract(base, source, profile["adapter"])
        if "const fp = @fp16();" not in source or "@fpfp()" in source:
            raise ValueError("BF16 format selection was altered")
        folder = candidate_init(profile, name)
        path = folder / "candidate.json"
        manifest = json.loads(path.read_text())
        manifest.update(
            shape_specific=False,
            approach=approach,
            hypothesis=hypothesis,
            falsification="Reject skipped/reordered projection terms, BF16 accumulation, missing stage quantization, stale logical reads, numerical/state/shutdown failures, protected-control changes, speedup below 1.05x or memory above 1.10x. Require six families, fresh matched seed and complete suite guards for cross-shape claims.",
            author="unknown",
            model="unknown",
            agent_harness="Codex",
            qualification_receipt=str(args.qualified_baseline),
            qualification_receipt_sha256=digest(args.qualified_baseline),
            primitive_math_probe_sha256=digest(probe_path),
        )
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        (folder / "src/meshgemm.csl").write_text(source)
        (folder / "README.md").write_text(
            "# " + name + "\n\n" + hypothesis + "\n\n" + manifest["falsification"] + "\n"
        )
        print(folder)


if __name__ == "__main__":
    main()
