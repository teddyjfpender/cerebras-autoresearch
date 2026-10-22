"""Create a separate bank-layout hypothesis after the reusable adapter is live."""

import json
import re
import shutil
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / "src"))
from cerebras_research.adapters import candidate_init, verify_timing_contract  # noqa: E402
from cerebras_research.catalog import get_profile  # noqa: E402
from cerebras_research.paths import write_json  # noqa: E402

profile = get_profile("reusable-gemm.tiny-sequence")
parent = root / "experiments/candidates/reusable-gemm-map-k-reuse"
manifest = json.loads((parent / "candidate.json").read_text())
original = (root / "kernels/dense/reusable-gemm/src/meshgemm.csl").read_text()
source = (parent / "src/meshgemm.csl").read_text()
weight = "var W_0_tile: [Kt*Nt]f16 = @zeros([Kt*Nt]f16);"
rows = "var row_accum: [Mt*Nt]f16;"
assert source.count(weight) == 1 and source.count(rows) == 1
storage = """// Distinct mod-4 bank groups for contiguous weight and output streams.
comptime { @comptime_assert(Nt % 4 == 0); }
const weight_rows_t = struct {
    weights: [Kt*Nt]f16,
    pad: [2]f16,
    rows: [Mt*Nt]f16,
};
var weight_rows: weight_rows_t align(16) = .{
    .weights = @zeros([Kt*Nt]f16),
    .pad = @zeros([2]f16),
    .rows = @zeros([Mt*Nt]f16),
};"""
source = source.replace(weight, storage).replace(rows, "")
source = re.sub(r"\bW_0_tile\b", "weight_rows.weights", source)
source = re.sub(r"\brow_accum\b", "weight_rows.rows", source)
source = source.replace("var W_1_tile: [Kt*Nt]f16 =", "var W_1_tile: [Kt*Nt]f16 align(16) =")
verify_timing_contract(original, source, "reusable_gemm")
folder = candidate_init(profile, "reusable-gemm-map-k-reuse-bank")
for p in parent.rglob("*.csl"):
    destination = folder / p.relative_to(parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, destination)
(folder / "src/meshgemm.csl").write_text(source)
metadata = json.loads((folder / "candidate.json").read_text())
metadata.update(
    approach="reusable-gemm.contiguous-columns-map-k-reuse-bank-layout",
    hypothesis="The inspected cold vector-pack ELF places W0, W1 and row_accum in the same mod-4 bank group on PE (4,1). Keep the mapped compute and the same per-frame weight-reuse hypothesis, but co-locate W0 and row accumulators in an aligned struct separated by two FP16 words, and align W1. For the registered Nt multiples of four this aims to separate the weight and accumulator stream groups. Inspect compiler layout and measure all outputs/state; static addresses do not prove dynamic stalls.",
    falsification="Reject compile/exit errors, any six-case output or exact per-frame W0/canonical-state failure, changed precision/timer/PE contract, speedup under 1.05x, memory over 1.10x, or failed fresh pairing. Compare with the otherwise identical mapped-reuse candidate. Report any gain as a measured layout change unless instruction traces establish its cause.",
    shape_specific=False,
    agent_harness="Codex",
)
write_json(folder / "candidate.json", metadata)
(folder / "README.md").write_text(
    "# Bank separation for reusable mapped GEMM\n\nUnmeasured. "
    + metadata["hypothesis"]
    + "\n\n"
    + metadata["falsification"]
    + "\n\nRequires Nt divisible by four, enforced at compile time. Protected sequence, snapshots and timer functions remain unchanged. See LICENSE.upstream for the pinned source.\n"
)
print(folder)
