"""Create actual candidate workspaces only after the reviewed adapter is live."""

import json
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / "src"))
from cerebras_research.adapters import candidate_init, functions, verify_timing_contract  # noqa: E402
from cerebras_research.catalog import get_profile  # noqa: E402
from cerebras_research.paths import write_json  # noqa: E402

profile = get_profile("reusable-gemm.tiny-sequence")
if profile["adapter"] != "reusable_gemm":
    raise SystemExit("Reusable adapter not integrated")
baseline = root / "kernels/dense/reusable-gemm/src/meshgemm.csl"
original = baseline.read_text()
cold = (root / "experiments/candidates/gemm-row-major-map-k-base/src/meshgemm.csl").read_text()
if "&ptr_X_send[row]" in cold:
    raise SystemExit("Cold mapping still has the rejected scalar pointer base")
start = cold.index("// Modified for research: contiguous output-column vectors")
end = cold.index("// ------------------------- Timer", start)
helpers = cold[start:end]
start = cold.index("// DSR source 1 now streams")
end = cold.index("// ---------------------- Entry Point", start)
compute = cold[start:end]
old_functions = functions(original)
left = old_functions["left_matrix_shift_callback"]
assert left.count("@fmovh(res_dsd, 0.0);") == 1
left = left.replace(
    "@fmovh(res_dsd, 0.0);", "@fmovh(res_dsd, 0.0);\n        @fmovh(row_accum_dsd, 0.0);"
)
replaced = original.replace(old_functions["gemv_static_step_X"], "")
replaced = replaced.replace(old_functions["mm_compute"], compute)
replaced = replaced.replace(old_functions["left_matrix_shift_callback"], left)
marker = "// ------------------------- Timer"
assert replaced.count(marker) == 1
replaced = replaced.replace(marker, helpers + "\n" + marker)
for mapped, reuse in ((False, True), (True, False), (True, True)):
    name = ("reusable-gemm-map-k" if mapped else "reusable-gemm") + ("-reuse" if reuse else "")
    edited = replaced if mapped else original
    if reuse:
        edited = edited.replace(
            "@fmovh(working_w0_dsd, canonical_w_dsd);",
            "if (frame == 0) { @fmovh(working_w0_dsd, canonical_w_dsd); }",
        )
        edited = edited.replace("    @fmovh(working_x1_dsd, 0.0);\n", "")
        edited = edited.replace("    @fmovh(working_w1_dsd, 0.0);\n", "")
    verify_timing_contract(original, edited, "reusable_gemm")
    folder = candidate_init(profile, name)
    (folder / "src/meshgemm.csl").write_text(edited)
    manifest = json.loads((folder / "candidate.json").read_text())
    manifest.update(
        approach=(
            "reusable-gemm.contiguous-columns-map-k" if mapped else "reusable-gemm.baseline-compute"
        )
        + ("-weights-reuse" if reuse else ""),
        hypothesis=(
            "Accumulate contiguous output-column vectors, mapping over each complete strided K row with a stationary output descriptor and advancing weight descriptor. Transpose all outputs back to the fixed ABI. "
            if mapped
            else "Keep the original complete FP16 matrix computation and communication. "
        )
        + (
            "Only reload canonical weights on frame zero and omit working-buffer clears overwritten before use; the full P=4 communication cycle must return W0 exactly before every next frame."
            if reuse
            else "Preserve every baseline working-buffer reset, canonical weight reload and counter reset to isolate the compute-layout change."
        ),
        falsification="Reject a compile/exit error, any six-case output or exact canonical/per-frame W0 state failure, changed precision/timing/PE contract, speedup below 1.05x, memory growth over 1.10x, or failed fresh paired seed. No cold-GEMM or model-TPS comparison.",
        shape_specific=False,
        agent_harness="Codex",
    )
    write_json(folder / "candidate.json", manifest)
    (folder / "README.md").write_text(
        "# "
        + name
        + "\n\n"
        + manifest["hypothesis"]
        + "\n\n"
        + manifest["falsification"]
        + "\n\nProtected sequence/timer/snapshot functions remain byte-equivalent to the registered reference. Initially unmeasured; retain only through the standard challenge. See LICENSE.upstream for the pinned WaferLLM source.\n"
    )
    print(name, flush=True)
