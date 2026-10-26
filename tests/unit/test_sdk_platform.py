"""Execution-platform changes need new evaluator-bound baselines."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cerebras_research.paths import digest, identity, write_json
from cerebras_research.provenance import evaluator_files
from cerebras_research.sdk import execution_sdk


class PlatformTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = {
            "sif_sha256": "fixed-image",
            "guest_os": "historical",
            "host_qemu_version": "old",
        }
        write_json(self.root / "config/sdk-install.json", self.base)
        self.profile = {
            "schema_version": 1,
            "id": "fixture-arm-platform",
            "lima_instance": "fixture",
            "hostname": "lima-fixture",
            "controller_machine": "aarch64",
            "virtualization": "vz",
            "container_runtime": "singularity-ce version 4.1.1",
            "container_package": "singularity-container=4.1.1+fixture",
        }
        self.selected = self.root / "config/sdk-platform.json"

    def configured(self):
        write_json(self.selected, self.profile)
        for target, value in (
            ("platform.machine", "aarch64"),
            ("platform.node", "lima-fixture"),
            ("platform.release", "fixture-kernel"),
            ("platform.python_version", "3.12.0"),
            ("platform.freedesktop_os_release", {"PRETTY_NAME": "Fixture Linux"}),
        ):
            context = patch("cerebras_research.sdk." + target, return_value=value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.dict("os.environ", {"CEREBRAS_SDK_VM": "fixture"})
        context.start()
        self.addCleanup(context.stop)

    def test_historical_fixture_without_selected_platform_is_unchanged(self):
        self.assertEqual(execution_sdk(self.root), self.base)

    def test_actual_runtime_and_controller_are_recorded_without_rewriting_installation(self):
        self.configured()
        original = (self.root / "config/sdk-install.json").read_bytes()
        with patch(
            "cerebras_research.sdk.subprocess.check_output",
            side_effect=[self.profile["container_runtime"], "4.1.1+fixture"],
        ):
            result = execution_sdk(self.root)
        self.assertEqual(result["sif_sha256"], "fixed-image")
        self.assertEqual(result["execution_platform"], self.profile)
        self.assertEqual(result["execution_platform_sha256"], digest(self.selected))
        self.assertEqual(result["controller"]["machine"], "aarch64")
        self.assertEqual(result["guest_os"], "Fixture Linux aarch64")
        self.assertNotIn("host_qemu_version", result)
        self.assertEqual((self.root / "config/sdk-install.json").read_bytes(), original)

    def test_wrong_vm_or_controller_is_rejected_before_sdk_work(self):
        self.configured()
        with patch.dict("os.environ", {"CEREBRAS_SDK_VM": "other"}):
            with self.assertRaisesRegex(ValueError, "SDK VM"):
                execution_sdk(self.root)
        with patch("cerebras_research.sdk.platform.machine", return_value="x86_64"):
            with self.assertRaisesRegex(ValueError, "controller"):
                execution_sdk(self.root)

    def test_runtime_or_package_change_requires_an_explicit_platform_update(self):
        self.configured()
        for values in (
            ["different runtime"],
            [self.profile["container_runtime"], "different package"],
        ):
            with (
                self.subTest(values=values),
                patch("cerebras_research.sdk.subprocess.check_output", side_effect=values),
                self.assertRaisesRegex(ValueError, "differs"),
            ):
                execution_sdk(self.root)

    def test_platform_and_launcher_changes_produce_distinct_evaluator_tracks(self):
        self.configured()
        launcher = self.root / "tools/sdk-run"
        launcher.parent.mkdir()
        launcher.write_text("# launcher one\n")
        first = identity(evaluator_files(self.root))
        self.profile["id"] = "other-platform"
        write_json(self.selected, self.profile)
        second = identity(evaluator_files(self.root))
        self.assertNotEqual(first, second)
        launcher.write_text("# launcher two\n")
        self.assertNotEqual(second, identity(evaluator_files(self.root)))
        self.assertEqual(json.loads(self.selected.read_text())["id"], "other-platform")
