"""Create new runtime-alias candidates after the archived comptime-cast failure."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from cerebras_research.adapters import candidate_init, verify_timing_contract
from cerebras_research.catalog import get_profile
from cerebras_research.paths import digest, tree_digest


def main():
    root = Path(__file__).resolve().parents[3]
    names = []
    for old, new in [
        ("bf16-swiglu-vector-scratch-dispatch", "bf16-swiglu-vector-scratch-runtime"),
        ("bf16-swiglu-vector-scratch-tail4", "bf16-swiglu-vector-scratch-tail4-runtime"),
        ("bf16-swiglu-vector-scratch-tail5", "bf16-swiglu-vector-scratch-tail5-runtime"),
    ]:
        parent = root / "experiments/candidates" / old
        manifest = json.loads((parent / "candidate.json").read_text())
        source = (parent / "src/meshgemm.csl").read_text()
        oldline = "var silu_square_dsd = @get_dsd(mem1d_dsd, .{ .base_address = @ptrcast([*]f32, &W_1_tile), .extent = Ht*Ft/2 });"
        assert source.count(oldline) == 1
        source = source.replace(
            oldline,
            "var silu_square_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Mt*Ft} -> res_tile[i] });",
        )
        needle = "fn nonlinear_vector_product() void {\n"
        assert source.count(needle) == 1
        source = source.replace(
            needle,
            needle
            + "    silu_square_dsd = @set_dsd_base_addr(silu_square_dsd, @ptrcast([*]f32, &W_1_tile));\n",
        )
        baseline = (root / "kernels/composed/resident-swiglu-bf16/src/meshgemm.csl").read_text()
        verify_timing_contract(baseline, source, "resident_swiglu_bf16")
        folder = candidate_init(get_profile(manifest["initial_profile"]), new)
        path = folder / "candidate.json"
        m = json.loads(path.read_text())
        for key in (
            "shape_specific",
            "hypothesis",
            "falsification",
            "author",
            "model",
            "agent_harness",
            "qualification_receipt",
            "qualification_receipt_sha256",
            "primitive_math_probe_sha256",
        ):
            m[key] = manifest[key]
        m.update(
            approach=manifest["approach"] + ".runtime-dsd-alias",
            hypothesis=manifest["hypothesis"]
            + " Set the aligned W1-to-FP32 scratch alias at runtime: this SDK rejects comptime pointer casts. Preserve the same lifetime/capacity predicate and all requested work.",
            parent_source_sha256=tree_digest(parent),
            parent_manifest_sha256=digest(parent / "candidate.json"),
        )
        path.write_text(json.dumps(m, indent=2) + "\n")
        (folder / "src/meshgemm.csl").write_text(source)
        (folder / "README.md").write_text(
            "# " + new + "\n\n" + m["hypothesis"] + "\n\n" + m["falsification"] + "\n"
        )
        names.append(new)
        print(folder)
    plan = json.loads(
        (
            root / "experiments/campaigns/resident-swiglu-bf16/optimization-scratch-and-tail.json"
        ).read_text()
    )
    plan["name"] = "bf16-scaled-runtime-scratch-and-tail"
    plan["cells"] = [plan["cells"][1]]
    plan["cells"][0]["candidates"] = [dict(variant="candidate", candidate=n) for n in names]
    out = (
        root / "experiments/campaigns/resident-swiglu-bf16/optimization-runtime-scratch-scaled.json"
    )
    assert not out.exists()
    out.write_text(json.dumps(plan, indent=2) + "\n")
    p = root / "experiments/incubator/resident-swiglu-bf16/heldout-runtime-freeze.json"
    assert not p.exists()
    p.write_text(
        json.dumps(
            dict(
                schema_version=1,
                status="proposed_not_registered_or_executed",
                reason="Runtime-only pointer cast required after the SDK rejected the original comptime alias; no held-out device runs have occurred.",
                candidate_manifests=[
                    dict(
                        candidate=n,
                        source_sha256=tree_digest(root / "experiments/candidates" / n),
                        manifest_sha256=digest(
                            root / "experiments/candidates" / n / "candidate.json"
                        ),
                    )
                    for n in names
                ],
                profiles=json.loads(
                    (
                        root
                        / "experiments/incubator/resident-swiglu-bf16/heldout-profiles-proposal.json"
                    ).read_text()
                )["profiles"],
                device_execution=False,
                qwen_tps_result=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
