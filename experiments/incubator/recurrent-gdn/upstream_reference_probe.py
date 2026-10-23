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
sys.path.insert(0, str(Path(__file__).parent))
import oracle  # noqa: E402
from common import CASES  # noqa: E402
from resident_swiglu_bf16_oracle import decode, encode  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference():
    metadata = json.loads(Path(__file__).with_name("reference.json").read_text())
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


def run_case(function, observe_line, k, case, raw_directory):
    data = oracle.inputs(k, 16, 20261035, case)
    q, key = (
        decode(data[name]).reshape(16, 4, k).transpose(1, 0, 2) for name in ("Q_CANON", "K_CANON")
    )
    q, key = (np.repeat(value, 3, axis=1)[None].copy() for value in (q, key))
    value = (
        decode(data["V_CANON"])
        .reshape(16, 4, 3, 16)
        .transpose(1, 0, 2, 3)
        .reshape(1, 4, 48, 16)
        .copy()
    )
    g, beta = (
        data[name].reshape(16, 4, 3).transpose(1, 0, 2).reshape(1, 4, 48).copy()
        for name in ("G_CANON", "BETA_CANON")
    )
    initial = data["S0"].reshape(1, 48, k, 16).copy()
    tensors = [torch.from_numpy(a).clone().to(torch.bfloat16) for a in (q, key, value)]
    tensors += [torch.from_numpy(a).clone() for a in (g, beta, initial)]
    before = [bits(tensor).tobytes() for tensor in tensors]
    observed = []

    def trace(frame, event, arg):
        if frame.f_code is not function.__code__:
            return None
        if event == "line" and frame.f_lineno == observe_line:
            local = frame.f_locals
            observed.append(
                {
                    name: bits(local[name]).copy()
                    for name in ("query", "key", "decay_t", "delta", "last_recurrent_state")
                }
            )
        return trace

    previous_trace = sys.gettrace()
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
        sys.settrace(previous_trace)
    if len(observed) != 4 or any(bits(t).tobytes() != value for t, value in zip(tensors, before)):
        raise ValueError("Incomplete observation or immutable upstream input changed")
    qn, kn = [], []
    for name, target in (("query", qn), ("key", kn)):
        array = observed[-1][name][0].reshape(16, 3, 4, k)
        if any(array[:, 0].tobytes() != array[:, head].tobytes() for head in (1, 2)):
            raise ValueError("Repeated key-head normalized values differ")
        target.append(array[:, 0].reshape(4, 4, 4, k).copy())
    actual = dict(
        Q_HISTORY=qn[0],
        K_HISTORY=kn[0],
        A_HISTORY=np.stack([a["decay_t"].reshape(4, 4, 3) for a in observed], axis=2),
        DELTA_HISTORY=np.stack([a["delta"].reshape(4, 4, 3, 16) for a in observed], axis=2),
        STATE_HISTORY=np.stack(
            [a["last_recurrent_state"].reshape(4, 4, 3, k, 16) for a in observed], axis=2
        ),
        Y_HISTORY=bits(output)[0]
        .reshape(4, 16, 3, 16)
        .transpose(1, 0, 2, 3)
        .reshape(4, 4, 4, 3, 16)
        .copy(),
        FINAL_STATE=bits(final).reshape(4, 4, 3, k, 16).copy(),
    )
    arrays, error, bound = oracle.evidence(
        k, 16, data, actual, {n: a.copy() for n, a in data.items()}
    )
    if not arrays["exact_state_match"] or not np.all(error <= bound):
        raise ValueError(
            f"Upstream reference violated fixed contract: K{k} {case}, fraction {float((error / bound).max())}"
        )
    sections = {}
    offsets = arrays["check_offsets"]
    for name, start, end in zip(arrays["check_labels"], offsets[:-1], offsets[1:]):
        region = slice(int(start), int(end))
        sections[str(name)] = float((error[region] / bound[region]).max())
    digest = hashlib.sha256()
    for name, value in data.items():
        digest.update(
            name.encode() + str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()
        )
    artifact = raw_directory / f"k{k}-{case}.npz"
    if artifact.exists():
        raise ValueError("Refuse to overwrite a raw upstream fixture")
    np.savez_compressed(artifact, **arrays, error_bound=bound)
    return dict(
        K=k,
        V=16,
        case=case,
        seed=20261035,
        input_sha256=digest.hexdigest(),
        artifact=str(artifact.relative_to(ROOT)),
        artifact_sha256=sha(artifact),
        passed=True,
        max_error_fraction=float((error / bound).max()),
        per_check_max_error_fraction=sections,
        all_immutable_inputs_and_final_state_exact=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/maintenance/recurrent-gdn-upstream-reference.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Refuse to overwrite upstream reference fixture")
    raw_directory = output.with_suffix(".raw")
    raw_directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    function, line, metadata = load_reference()
    records = [run_case(function, line, k, case, raw_directory) for k in (16, 32) for case in CASES]
    report = dict(
        schema_version=1,
        status="upstream_cpu_reference_checked",
        reference=metadata,
        script_sha256=sha(__file__),
        oracle_sha256=sha(oracle.__file__),
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
