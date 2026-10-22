"""Reused suite evidence must preserve full coverage and one measurement context."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cerebras_research.paths import digest, identity, write_json
from cerebras_research.process import simulator_lease
from tests.support.records import make_record
from tools import assemble_suite


class SuiteAssemblyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        context = patch("tools.assemble_suite.ROOT", self.root)
        context.start()
        self.addCleanup(context.stop)
        self.profiles = ["gated-mlp.h16-f32", "gated-mlp.h32-f64"]
        self.receipts = [
            make_record(self.root, "run-" + str(i), profile_id=profile)
            for i, profile in enumerate(self.profiles)
        ]
        self.manifest = self.root / "manifest.json"

    def build(self, paths=None):
        write_json(
            self.manifest,
            {
                "schema_version": 1,
                "receipts": [
                    {"path": str(path.relative_to(self.root)), "sha256": digest(path)}
                    for path in (self.receipts if paths is None else paths)
                ],
            },
        )
        return assemble_suite.build_suite("gated-mlp", self.manifest)

    def test_complete_import_is_ordered_by_suite_and_is_not_new_execution(self):
        result = self.build(list(reversed(self.receipts)))
        self.assertEqual(result["profiles"], self.profiles)
        self.assertEqual(list(result["runs"]), self.profiles)
        self.assertEqual(result["imported_run_count"], 2)
        self.assertEqual(result["executed_attempt_count"], 0)
        self.assertEqual(result["execution_mode"], "assembled_existing_full_runs")
        self.assertNotIn("steps", result)

    def test_metadata_writes_have_their_own_bounded_space_reserve(self):
        with patch(
            "tools.assemble_suite.shutil.disk_usage",
            return_value=SimpleNamespace(free=128 * 1024**2),
        ):
            with assemble_suite.metadata_lease():
                pass
        with (
            patch(
                "tools.assemble_suite.shutil.disk_usage",
                return_value=SimpleNamespace(free=32 * 1024**2),
            ),
            self.assertRaisesRegex(RuntimeError, "64 MiB"),
        ):
            with assemble_suite.metadata_lease():
                self.fail("Metadata writer entered below its reserve")
        self.assertFalse((self.root / "runs/v2").exists())

    def test_metadata_and_sdk_writers_still_share_one_lock(self):
        with (
            patch("cerebras_research.process.ROOT", self.root),
            patch(
                "tools.assemble_suite.shutil.disk_usage",
                return_value=SimpleNamespace(free=2 * 1024**3),
            ),
            assemble_suite.metadata_lease(),
        ):
            with self.assertRaisesRegex(RuntimeError, "Another benchmark"):
                with simulator_lease():
                    self.fail("SDK writer overlapped the metadata writer")

    def test_missing_duplicate_and_outside_profiles_do_not_establish_coverage(self):
        for paths, error in (
            (self.receipts[:1], "incomplete"),
            ([self.receipts[0]] * 2, "Duplicate"),
            (
                self.receipts
                + [make_record(self.root, "outside", profile_id="decoder.h16-f32-ctx4")],
                "named suite",
            ),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                self.build(paths)

    def test_changed_receipt_or_raw_artifact_is_rejected(self):
        self.build()
        path = self.receipts[0]
        before = path.read_text()
        data = json.loads(before)
        data["note"] = "changed after pointer capture"
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            assemble_suite.build_suite("gated-mlp", self.manifest)
        path.write_text(before)
        raw = path.parent / "work/out/positive.npz"
        raw.write_bytes(b"changed numerical evidence")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.build()

    def test_screen_cannot_be_imported_as_a_complete_measurement(self):
        path = self.receipts[0]
        data = json.loads(path.read_text())
        data.update(status="screen_passed", screen_only=True)
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "full independent"):
            self.build()

    def test_distinct_profile_helpers_are_preserved_for_later_paired_comparison(self):
        path = self.receipts[1]
        data = json.loads(path.read_text())
        helper = path.parent / "work/driver.py"
        helper.write_text("# profile-specific helper")
        data["harness_files"]["driver.py"] = digest(helper)
        data["harness_sha256"] = identity(data["harness_files"])
        data["artifacts"]["work/driver.py"] = digest(helper)
        write_json(path, data)
        result = self.build()
        self.assertNotEqual(
            *(result["implementations"][name]["harness_sha256"] for name in self.profiles)
        )

    def test_mixed_seed_variant_and_sdk_are_rejected(self):
        for index, changes in enumerate((dict(seed=8), dict(variant="candidate"), dict(sdk=True))):
            with self.subTest(changes=changes):
                override = dict(changes)
                different_sdk = override.pop("sdk", False)
                path = make_record(
                    self.root, "changed-" + str(index), profile_id=self.profiles[1], **override
                )
                if different_sdk:
                    data = json.loads(path.read_text())
                    data["sdk"] = {"sif_sha256": "another-sdk"}
                    write_json(path, data)
                with self.assertRaisesRegex(ValueError, "mixes"):
                    self.build([self.receipts[0], path])

    def test_old_profile_contract_cannot_claim_current_named_suite(self):
        path = self.receipts[0]
        data = json.loads(path.read_text())
        data["profile"]["dimensions"]["T"] += 1
        data["profile_sha256"] = identity(data["profile"])
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "named suite contract"):
            self.build()
