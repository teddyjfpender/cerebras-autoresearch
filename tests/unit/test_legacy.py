"""Acceptance-gate regression checks; no SDK or network required."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from cerebras_research.legacy import CASES, compare, run_logged


class ComparisonGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = {
            "status": "validated_kernel_run",
            "variant": "baseline",
            "P": 4,
            "M": 64,
            "N": 64,
            "group_num": 2,
            "seed": 7,
            "source": {"commit": "pinned"},
            "verifier_sha256": "same",
            "commands": [["same compiler and workload"]],
            "sdk_install": {"sdk_release": "2.10.1", "sdk_build": "pinned", "sif_sha256": "same"},
            "upstream_max_simulated_cycles": 100,
            "validation_result": {
                "status": "all_cases_passed",
                "seed": 7,
                "reference": "FP64",
                "contract": "fixed",
                "warmups": 1,
                "repeats": 10,
                "timing_status": "simulated",
                "cases": [
                    {
                        "case": name,
                        "max_fraction_of_error_bound": 0.1,
                        "upstream_max_cycles": 100,
                        "artifact": name + ".npz",
                        "artifact_sha256": hashlib.sha256(b"raw").hexdigest(),
                    }
                    for name in sorted(CASES)
                ],
            },
        }
        self.candidate = copy.deepcopy(self.base)
        self.candidate["variant"] = "unroll"
        self.candidate["upstream_max_simulated_cycles"] = 80
        for case in self.candidate["validation_result"]["cases"]:
            case["upstream_max_cycles"] = 80

    def save(self, name, data):
        folder = self.root / name
        (folder / "out_4_16_16_2").mkdir(parents=True, exist_ok=True)
        for case in data["validation_result"]["cases"]:
            (folder / "out_4_16_16_2" / case["artifact"]).write_bytes(b"raw")
        path = folder / "result.json"
        path.write_text(json.dumps(data))
        return path

    def pair(self):
        return self.save("base", self.base), self.save("candidate", self.candidate)

    def test_gain_requires_confirmation(self):
        result = compare(*self.pair())
        self.assertEqual(result["decision"], "needs_fresh_confirmation")
        self.assertEqual(result["speedup"], 1.25)

    def test_reject_changed_workload_or_precision_contract(self):
        for key, value in (("M", 80), ("seed", 8), ("verifier_sha256", "different")):
            with self.subTest(key=key):
                candidate = copy.deepcopy(self.candidate)
                candidate[key] = value
                with self.assertRaises(ValueError):
                    compare(self.save("base", self.base), self.save("candidate", candidate))

    def test_reject_screen_missing_case_or_numerical_failure(self):
        for change in ("screen", "case", "error"):
            with self.subTest(change=change):
                candidate = copy.deepcopy(self.candidate)
                if change == "screen":
                    candidate["screen_only"] = True
                elif change == "case":
                    candidate["validation_result"]["cases"].pop()
                else:
                    candidate["validation_result"]["cases"][0]["max_fraction_of_error_bound"] = 1.1
                with self.assertRaises(ValueError):
                    compare(self.save("base", self.base), self.save("candidate", candidate))

    def test_reject_corrupted_artifact(self):
        base, candidate = self.pair()
        (candidate.parent / "out_4_16_16_2/positive.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "raw artifact"):
            compare(base, candidate)

    def test_no_gain_is_rejected(self):
        self.candidate["upstream_max_simulated_cycles"] = 100
        for case in self.candidate["validation_result"]["cases"]:
            case["upstream_max_cycles"] = 100
        self.assertEqual(compare(*self.pair())["decision"], "rejected")

    def test_log_flood_is_stopped_and_rejected(self):
        out = self.root / "out_test"
        out.mkdir()
        script = (
            "from pathlib import Path; import time; "
            "f=Path('out_test/sim.log').open('wb'); "
            "f.truncate(65*1024*1024); f.close(); time.sleep(60)"
        )
        with self.assertRaisesRegex(RuntimeError, "Log exceeded"):
            run_logged([sys.executable, "-c", script], self.root, self.root / "step.log", timeout=5)


if __name__ == "__main__":
    unittest.main()
