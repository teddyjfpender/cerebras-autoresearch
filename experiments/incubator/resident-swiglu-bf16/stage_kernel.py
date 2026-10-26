"""Stage a distinct BF16-storage/FP32-accumulation baseline; never edit live kernels."""

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    source = ROOT / "kernels/composed/resident-swiglu"
    target = HERE / "kernel"
    target.mkdir(exist_ok=True)
    shutil.copytree(source / "src", target / "src", dirs_exist_ok=True)
    shutil.copy2(source / "LICENSE.upstream", target / "LICENSE.upstream")
    for path in (target / "src").rglob("*.csl"):
        text = path.read_text()
        if re.search(r"\bf16\b", text):
            text = "const fp = @fp16();\n" + re.sub(r"\bf16\b", "fp", text)
        path.write_text(text)
    path = target / "src/meshgemm.csl"
    text = path.read_text()
    text = text.replace(
        "const fp = @fp16();", "const fp = @fp16();\ncomptime { @comptime_assert(fp == bf16); }", 1
    )
    for old, new in [
        ("var res_tile: [Mt*Ft]fp", "var res_tile: [Mt*Ft]f32"),
        ("var ptr_res: [*]fp", "var ptr_res: [*]f32"),
        ("@zeros([Mt*Ft]fp)", "@zeros([Mt*Ft]f32)"),
        ("var ptr_out: [*]fp", "var ptr_out: [*]f32"),
        ("var Y_HISTORY: [T*Mt*Ht]f32", "var Y_HISTORY: [T*Mt*Ht]fp"),
        ("const ptr_Y_HISTORY: [*]f32", "const ptr_Y_HISTORY: [*]fp"),
        ("@fmach(out_dest_dsr", "@fmachs(out_dest_dsr"),
        ("@fmovh(res_dsd, 0.0)", "@fmovs(res_dsd, 0.0)"),
        ("@fmovh(target, full_output_dsd)", "@fs2h(target, full_output_dsd)"),
        ("@fmovh(target, down_output_dsd)", "@fs2h(target, down_output_dsd)"),
        ("const u = @as(f32, res_tile[i]);", "const u = @as(f32, U_HISTORY[frame*Mt*Ft+i]);"),
        (
            "Y_HISTORY[frame*Mt*Ht+i] = @as(f32, res_tile[i]) + @as(f32, X_FRAMES[frame*Mt*Ht+i]);",
            "Y_HISTORY[frame*Mt*Ht+i] = @as(fp, @as(f32, D_HISTORY[frame*Mt*Ht+i]) + @as(f32, X_FRAMES[frame*Mt*Ht+i]));",
        ),
    ]:
        if old not in text:
            raise ValueError("Qualified source template changed: " + old)
        text = text.replace(old, new)
    assert "@fmach(" not in text and "@fmachs(" in text
    path.write_text(text)
    path = target / "src/layout.csl"
    text = (
        path.read_text()
        .replace('@export_name("res", [*]fp, true)', '@export_name("res", [*]f32, true)')
        .replace(
            '@export_name("Y_HISTORY", [*]f32, false)', '@export_name("Y_HISTORY", [*]fp, false)'
        )
    )
    path.write_text(text)
    (target / "README.md").write_text(
        "# Staged BF16 resident FFN baseline\n\nUnregistered precision contract: native BF16 storage and operands with FP32\nprojection accumulators, FP32 SiLU/product/residual, then BF16 stage/output\nrounding. Every G/U/V/D/Y stage and all canonical/boundary weight state remain\nchecked. Derived from the independently qualified FP16 chain, with distinct\nprecision, oracle, compiler format flag and baselines required.\n"
    )
    print(json.dumps(dict(staged=str(target), device_qualified=False)))


if __name__ == "__main__":
    main()
