"""Full-head I/O bringup with immutable pointers to writable canonical buffers."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cerebras_research.adapters import candidate_init, verify_timing_contract  # noqa: E402
from cerebras_research.catalog import get_profile  # noqa: E402


def main():
    folder = candidate_init(
        get_profile("recurrent-gdn-full.k128-v128-h16-t2"), "full-gdn-constant-input-exports"
    )
    p = folder / "src/pe.csl"
    original = p.read_text()
    source = original
    layout = folder / "src/layout.csl"
    ls = layout.read_text()
    for name, dtype in [
        ("Q_CANON", "fp"),
        ("K_CANON", "fp"),
        ("V_CANON", "fp"),
        ("G_CANON", "f32"),
        ("BETA_CANON", "f32"),
        ("S0", "f32"),
    ]:
        before = f"var ptr_{name}: [*]{dtype} = &{name};"
        if before not in source:
            raise ValueError("Canonical export declaration changed")
        source = source.replace(before, before.replace("var ", "const ", 1))
        before = f'@export_name("{name}", [*]{dtype}, true);'
        if before not in ls:
            raise ValueError("Canonical layout export changed")
        ls = ls.replace(before, before.replace("true", "false"))
    verify_timing_contract(original, source, "recurrent_gdn_full")
    p.write_text(source)
    layout.write_text(ls)
    p = folder / "candidate.json"
    data = json.loads(p.read_text())
    data.update(
        approach="full_gdn.constant_input_export_bringup",
        hypothesis="Declare exported canonical-buffer pointers immutable while retaining writable BF16/FP32 arrays and exact initial inputs. This tests whether static address resolution improves the full-head setup path. The complete baseline arithmetic, every protected timer/store, PE budget and precision remain identical.",
        falsification="Reject any canonical input-bit difference, lost token/head/term/state history, fixed numerical/precision/timing failure, raw timer/core mismatch or shutdown failure. A bounded positive screen supplies bringup evidence only; it cannot establish a retained speedup or full qualification without matched six-case and fresh paired controls.",
        author="unknown",
        model="unknown",
        agent_harness="Codex",
        scope="Full native16 key/48 value heads,K128/V128,T2,768 PEs; setup-only hypothesis, no model TPS.",
        precision_changed=False,
        timing_changed=False,
    )
    p.write_text(json.dumps(data, indent=2) + "\n")
    (folder / "README.md").write_text(
        "# Immutable input-pointer bringup\n\nThe exported pointers become const; their BF16/FP32 buffers remain writable.\nNo timed arithmetic, input value, state store, precision or protected timestamp\nis changed. This targets setup address resolution. A positive-case device\nscreen is provisional and is not a complete baseline or retained improvement.\n"
    )
    print(folder)


if __name__ == "__main__":
    main()
