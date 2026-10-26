"""The advertised aggregate must not silently omit newly registered workloads."""

import unittest

from cerebras_research.catalog import get_suite, load_catalog


class CatalogCoverageTests(unittest.TestCase):
    def test_all_suite_contains_each_configured_profile_exactly_once(self):
        catalog = load_catalog()
        names = [profile["id"] for profile in get_suite("all")]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), set(catalog))
