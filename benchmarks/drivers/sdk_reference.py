"""Run audited upstream host checks with bounded, single-thread simulation.

These are reference/smoke benchmarks. They do not enter the independent-oracle
promotion league. Keep the upstream host driver immutable during kernel edits.
"""

import argparse
import json
import runpy
import sys
from pathlib import Path

import cerebras.sdk.runtime.sdkruntimepybind as sdk
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--validation", choices=("upstream_reference", "diagnostic_only"), required=True
    )
    parser.add_argument("script")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    base = Path.cwd()
    captures = []
    original = sdk.SdkRuntime

    class Runtime:
        def __init__(self, *values, **kwargs):
            kwargs.update(simfab_numthreads=1, suppress_simfab_trace=True)
            self.runtime = original(*values, **kwargs)
            self.arrays = []

        def __getattr__(self, name):
            return getattr(self.runtime, name)

        def memcpy_d2h(self, target, *values, **kwargs):
            result = self.runtime.memcpy_d2h(target, *values, **kwargs)
            self.arrays.append(target)
            return result

        def stop(self):
            self.runtime.stop()
            captures.extend(np.asarray(a).copy() for a in self.arrays)

    sdk.SdkRuntime = Runtime
    seed = np.random.seed
    seed(args.seed)
    np.random.seed = lambda *_, **__: seed(args.seed)
    sys.argv = [args.script, *args.args]
    runpy.run_path(args.script, run_name="__main__")
    np.savez_compressed(
        base / "upstream-captures.npz", **{f"d2h_{i}": a for i, a in enumerate(captures)}
    )
    (base / "upstream-validation.json").write_text(
        json.dumps(
            {
                "status": "upstream_reference_passed"
                if args.validation == "upstream_reference"
                else "diagnostic_completed",
                "seed": args.seed,
                "oracle": "pinned upstream CPU reference and assertions"
                if args.validation == "upstream_reference"
                else "none; transfer timing diagnostic",
                "simulator_threads": 1,
                "captured_arrays": len(captures),
                "independent_adversarial_oracle": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
