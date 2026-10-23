"""Transfer PR1 row-descriptor reuse and bank-offset accumulation to native GDN."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cerebras_research.adapters import candidate_init, verify_timing_contract  # noqa: E402
from cerebras_research.catalog import get_profile  # noqa: E402
from cerebras_research.paths import tree_digest  # noqa: E402

REFERENCE = "https://github.com/teddyjfpender/cerebras-autoresearch/pull/1"
REF_COMMIT = "5e818e10b7740daaa0d5e2962f7010100385e0e1"


def replace_loop(source, scalar, update=False):
    old = (
        """        i = 0;
        while (i < K) : (i += 1) {
            @fmacs("""
        + ("row, row, delta" if update else "scratch_dsd, scratch_dsd, row")
        + """, """
        + scalar
        + """[i]);
            row = @increment_dsd_offset(row, V, f32);
        }"""
    )
    if update:
        loads = """        @load_to_dsr(compute_dest, row, .{ .save_address = true });
        @load_to_dsr(compute_src0, row, .{ .save_address = true });
        @load_to_dsr(compute_src1, delta, .{ .save_address = false });
"""
    else:
        loads = """        @load_to_dsr(compute_dest, scratch_dsd, .{ .save_address = false });
        @load_to_dsr(compute_src0, scratch_dsd, .{ .save_address = false });
        @load_to_dsr(compute_src1, row, .{ .save_address = true });
"""
    new = (
        loads
        + """        i = 0;
        while (i < K) : (i += 1) {
            @fmacs(compute_dest, compute_src0, compute_src1, """
        + scalar
        + """[i]);
        }"""
    )
    if old not in source:
        raise ValueError("Parent row-loop structure changed")
    return source.replace(old, new, 1)


def main():
    parent = ROOT / "experiments/candidates/full-gdn-vector-norm"
    original = (parent / "src/pe.csl").read_text()
    bank = original.replace(
        "var VECTOR_SCRATCH: [V]f32;", "var VECTOR_SCRATCH: [V+1]f32 align(16);"
    ).replace("-> VECTOR_SCRATCH[i]", "-> VECTOR_SCRATCH[i+1]")
    dsr = bank.replace(
        "fn one_frame(",
        """// DSR banks and save_address pattern follow PR1's validated WSE3 kernel.
const compute_dest = @get_dsr(dsr_dest, 2);
const compute_src0 = @get_dsr(dsr_src0, 2);
const compute_src1 = @get_dsr(dsr_src1, 2);

fn one_frame(""",
        1,
    )
    dsr = replace_loop(dsr, "K_WORK")
    dsr = replace_loop(dsr, "K_WORK", update=True)
    dsr = replace_loop(dsr, "Q_WORK")
    for name, source in [("full-gdn-bank-offset", bank), ("full-gdn-dsr-bank-offset", dsr)]:
        folder = candidate_init(get_profile("recurrent-gdn-full.k128-v128-h16-t2"), name)
        verify_timing_contract(original, source, "recurrent_gdn_full")
        (folder / "src/pe.csl").write_text(source)
        path = folder / "candidate.json"
        data = json.loads(path.read_text())
        data.update(
            approach="full_gdn.bank_offset" + ("_dsr_row_reuse" if "dsr" in name else ""),
            hypothesis="Offset the aligned FP32 projection/readout accumulator by one word to separate SRAM banks from aligned state rows. "
            + (
                "Preload dedicated DSR bank2: persistent row addresses advance automatically, accumulator/delta addresses reset. Remove per-K descriptor reload/increment while preserving every ascending-K FMA."
                if "dsr" in name
                else "Keep all original row loops and operations to isolate the bank-offset effect."
            ),
            falsification="Reject any changed normalization/exp/FMA ordering, skipped K/V term/group/token/history, descriptor overrun or bank side effect, fixed numerical/state/precision/timing failure, shutdown failure, speedup below1.05x or memory above1.10x. Full six-case matched and fresh pairs required; PR1 GQA gains do not establish recurrent gains.",
            author="unknown",
            model="unknown",
            agent_harness="Codex",
            parent_candidate=parent.name,
            parent_tree_sha256=tree_digest(parent),
            reference_pr=REFERENCE,
            reference_commit=REF_COMMIT,
            reference_source_sha256=hashlib.sha256(
                (ROOT / "runs/references/pr-1/attention-newton.csl").read_bytes()
            ).hexdigest(),
        )
        path.write_text(json.dumps(data, indent=2) + "\n")
        (folder / "README.md").write_text(
            "# PR1-derived native recurrence dataflow\n\nThe reference is PR1 at "
            + REF_COMMIT
            + ". Its32-PE GQA result is a separate workload. This candidate transfers bank-offset accumulation and, where named, persistent DSR row addresses. Original ascending-K FP32 FMAs, normalized operands, every full state and all protected timing remain fixed. New matched device validation is required.\n"
        )
        print(folder)


if __name__ == "__main__":
    main()
