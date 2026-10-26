"""Replay the upstream CPU fixture in SDK NumPy; no device program or score."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/drivers"))
sys.path.insert(0, str(Path(__file__).parent))
import oracle  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "runs/maintenance/recurrent-gdn-upstream-reference-v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/maintenance/recurrent-gdn-upstream-sdk-replay.json",
    )
    args = parser.parse_args()
    args.reference = args.reference.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError("Refuse to overwrite a reference replay")
    report = json.loads(args.reference.read_text())
    if (
        sha(oracle.__file__) != report["oracle_sha256"]
        or report["status"] != "upstream_cpu_reference_checked"
    ):
        raise ValueError("Oracle source or upstream fixture changed")
    records = []
    for item in report["records"]:
        path = ROOT / item["artifact"]
        if sha(path) != item["artifact_sha256"]:
            raise ValueError("Upstream raw arrays changed")
        data = oracle.inputs(item["K"], item["V"], item["seed"], item["case"])
        digest = hashlib.sha256()
        for name, value in data.items():
            digest.update(
                name.encode()
                + str(value.dtype).encode()
                + str(value.shape).encode()
                + value.tobytes()
            )
        if digest.hexdigest() != item["input_sha256"]:
            raise ValueError("SDK and CPU reference generated different fixture inputs")
        with np.load(path, allow_pickle=False) as raw:
            for name, value in data.items():
                saved = raw["input_" + name]
                if (
                    saved.dtype != value.dtype
                    or saved.shape != value.shape
                    or saved.tobytes() != value.tobytes()
                ):
                    raise ValueError("Raw upstream fixture input differs from SDK generator")
            names = (
                "Q_HISTORY",
                "K_HISTORY",
                "A_HISTORY",
                "DELTA_HISTORY",
                "STATE_HISTORY",
                "Y_HISTORY",
                "FINAL_STATE",
            )
            arrays, error, bound = oracle.evidence(
                item["K"],
                item["V"],
                data,
                {n: raw[n] for n in names},
                {n: raw["canonical_" + n] for n in data},
            )
            for name in ("actual", "check_offsets", "check_labels"):
                np.testing.assert_array_equal(arrays[name], raw[name])
            np.testing.assert_allclose(arrays["expected"], raw["expected"], rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(bound, raw["error_bound"], rtol=2e-12, atol=0)
            if not arrays["exact_state_match"] or not np.all(error <= bound):
                raise ValueError("Upstream tensors fail the fixed contract in SDK NumPy")
        records.append(
            dict(
                K=item["K"],
                V=item["V"],
                case=item["case"],
                seed=item["seed"],
                input_sha256=item["input_sha256"],
                artifact_sha256=item["artifact_sha256"],
                input_bits_match=True,
                passed=True,
                max_error_fraction=float((error / bound).max()),
            )
        )
    output = dict(
        schema_version=1,
        status="upstream_raw_tensors_replayed_in_sdk_numpy",
        reference=dict(path=str(args.reference.relative_to(ROOT)), sha256=sha(args.reference)),
        script_sha256=sha(__file__),
        oracle_sha256=sha(oracle.__file__),
        numpy_version=np.__version__,
        records=records,
        fixture_count=len(records),
        sdk_device_execution=False,
        performance_measurement=False,
        qwen_tps_result=False,
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), status=output["status"], fixtures=len(records))))


if __name__ == "__main__":
    main()
