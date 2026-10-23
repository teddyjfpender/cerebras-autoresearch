"""Unscored CPU reference: observe the pinned upstream fallback without body edits.

uv run --no-project --python 3.12 --with torch==2.8.0 --with numpy==2.2.6 \
  experiments/incubator/recurrent-gdn/upstream_reference_probe.py
No SDK program, timer, promotion or full-model execution occurs here.
"""

import argparse
import ast
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
sys.path.insert(0, str(ROOT / "runs/staging/recurrent-gdn-evaluator/benchmarks/drivers"))
sys.path.insert(0, str(Path(__file__).parent))
import oracle  # noqa: E402
from common import CASES  # noqa: E402
from resident_swiglu_bf16_oracle import decode, encode  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference():
    metadata = json.loads((ROOT / "experiments/incubator/recurrent-gdn/reference.json").read_text())
    source = ROOT / "runs/references/qwen3_5/modeling_qwen3_5.py"
    if sha(source) != metadata["sha256"]:
        raise ValueError("Pinned upstream source changed")
    tree = ast.parse(source.read_text())
    names = ("l2norm", "torch_recurrent_gated_delta_rule")
    functions = [
        next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        for name in names
    ]
    for function in functions:
        function.decorator_list = []  # Bypass dispatch wrappers only, never rewrite the math body.
    loop = next(node for node in functions[1].body if isinstance(node, ast.For))
    output = next(
        node
        for node in loop.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "core_attn_out"
            for target in node.targets
        )
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace[names[1]], output.lineno, metadata


def bits(tensor):
    array = tensor.detach().cpu().float().numpy()
    return encode(array) if tensor.dtype == torch.bfloat16 else array


def run_case(function, observe_line, tokens, case, raw_directory):
    data = oracle.inputs(tokens, 20261038, case)
    q, key = (
        np.repeat(decode(data[name])[:, None, :], 3, axis=1)[None].copy()
        for name in ("Q_CANON", "K_CANON")
    )
    value = decode(data["V_CANON"])[None].copy()
    g, beta = (data[name][None].copy() for name in ("G_CANON", "BETA_CANON"))
    initial = data["S0"][None].copy()
    tensors = [torch.from_numpy(a).clone().to(torch.bfloat16) for a in (q, key, value)]
    tensors += [torch.from_numpy(a).clone() for a in (g, beta, initial)]
    before = [bits(tensor).tobytes() for tensor in tensors]
    observed = []

    def trace(frame, event, arg):
        if frame.f_code is not function.__code__:
            return None
        if event == "line" and frame.f_lineno == observe_line:
            observed.append(
                {
                    name: bits(frame.f_locals[name]).copy()
                    for name in ("query", "key", "decay_t", "delta", "last_recurrent_state")
                }
            )
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        output, final = function(
            *tensors[:3],
            g=tensors[3],
            beta=tensors[4],
            initial_state=tensors[5],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    finally:
        sys.settrace(previous)
    if len(observed) != tokens or any(bits(t).tobytes() != b for t, b in zip(tensors, before)):
        raise ValueError("Incomplete upstream observation or mutated input")
    prefixes = {}
    for name, target in (("query", "Q_HISTORY"), ("key", "K_HISTORY")):
        a = observed[-1][name][0]
        if any(a[0].tobytes() != a[h].tobytes() for h in (1, 2)):
            raise ValueError("Repeated key-head normalization differs")
        prefixes[target] = a[0].copy()
    actual = dict(
        **prefixes,
        A_HISTORY=np.stack([a["decay_t"].reshape(3) for a in observed]),
        DELTA_HISTORY=np.stack([a["delta"].reshape(3, 128) for a in observed]),
        STATE_HISTORY=np.stack([a["last_recurrent_state"].reshape(3, 128, 128) for a in observed]),
        Y_HISTORY=bits(output)[0].copy(),
        FINAL_STATE=bits(final)[0].copy(),
    )
    arrays, error, bound = oracle.evidence(
        tokens, data, actual, {n: a.copy() for n, a in data.items()}
    )
    if not arrays["exact_state_match"] or not np.all(error <= bound):
        raise ValueError(
            f"Native upstream violates fixed bound: T{tokens} {case}, {float((error / bound).max())}"
        )
    artifact = raw_directory / f"t{tokens}-{case}.npz"
    if artifact.exists():
        raise ValueError("Refuse raw fixture overwrite")
    np.savez_compressed(artifact, **arrays, error_bound=bound)
    return dict(
        K=128,
        V=128,
        value_heads=3,
        key_heads=1,
        tokens=tokens,
        case=case,
        seed=20261038,
        artifact=str(artifact.relative_to(ROOT)),
        artifact_sha256=sha(artifact),
        passed=True,
        max_error_fraction=float((error / bound).max()),
        all_immutable_inputs_and_final_state_exact=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/maintenance/recurrent-gdn-native-upstream-reference.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Refuse to overwrite upstream reference fixture")
    raw_directory = output.with_suffix(".raw")
    raw_directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    function, line, metadata = load_reference()
    records = [run_case(function, line, t, case, raw_directory) for t in (4, 8) for case in CASES]
    report = dict(
        schema_version=1,
        status="upstream_cpu_reference_checked",
        reference=metadata,
        script_sha256=sha(__file__),
        oracle_sha256=sha(oracle.__file__),
        shared_error_model_sha256=sha(
            ROOT / "runs/staging/recurrent-gdn-evaluator/benchmarks/drivers/recurrent_gdn_oracle.py"
        ),
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        platform=platform.platform(),
        math_body_modified=False,
        dispatch_decorators_removed=True,
        observation="Read-only Python line trace at the output assignment; captures actual normalized operands, decay, delta and every updated state.",
        records=records,
        fixture_count=len(records),
        sdk_execution=False,
        performance_measurement=False,
        qwen_tps_result=False,
        scope="Pinned recurrent fallback function only; no full Transformers model, projections, convolution or output normalization executed.",
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), status=report["status"], fixtures=len(records))))


if __name__ == "__main__":
    main()
