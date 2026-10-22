"""Keep API comparisons tied to the same workload."""

import json
import tempfile
import unittest
from pathlib import Path

from cerebras_research import telemetry as benchmark


def record(label, server_tps, concurrency=1):
    return {
        "status": "complete",
        "label": label,
        "model": benchmark.MODEL,
        "endpoint": benchmark.ENDPOINT,
        "prompts_sha256": "fixed-prompts",
        "request_count": 5,
        "concurrency": concurrency,
        "max_completion_tokens": 256,
        "reasoning_effort": "none",
        "temperature": 0,
        "seed": 7,
        "server_decode_tps": server_tps,
        "client_burst_output_tps": server_tps / 2,
        "responses": [{"system_fingerprint": "fp1"} for _ in range(5)],
    }


class CompareTests(unittest.TestCase):
    def compare_records(self, baseline, candidate):
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / "baseline.json"
            b = Path(directory) / "candidate.json"
            a.write_text(json.dumps(baseline))
            b.write_text(json.dumps(candidate))
            return benchmark.compare(a, b)

    def test_accepts_fifteen_percent_client_improvement(self):
        result = self.compare_records(record("baseline", 1000), record("candidate", 1200, 2))
        self.assertTrue(result["client_speedup_gate"])
        self.assertFalse(result["model_decode_tps_gain_claim_supported"])

    def test_rejects_changed_workload(self):
        candidate = record("candidate", 1200, 2)
        candidate["prompts_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "prompts_sha256"):
            self.compare_records(record("baseline", 1000), candidate)

    def test_rejects_too_few_requests_for_claim(self):
        baseline, candidate = record("baseline", 1000), record("candidate", 1200, 2)
        baseline["request_count"] = candidate["request_count"] = 2
        result = self.compare_records(baseline, candidate)
        self.assertFalse(result["client_speedup_gate"])


if __name__ == "__main__":
    unittest.main()
