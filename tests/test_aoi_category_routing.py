import unittest

from src.core.anomaly_memory_integration import (
    canonical_category_key,
    is_adhesive_category,
    routes_for_category,
)
from src.core.experts.shift_expert import ShiftExpert
from src.utils.text_normalizer import normalize_aoi_text


class AOICategoryNormalizationTests(unittest.TestCase):
    def test_new_portuguese_categories_are_returned_without_debug_suffix(self):
        self.assertEqual(normalize_aoi_text("INVERTIDO")[0], "INVERTIDO")
        self.assertEqual(normalize_aoi_text("FALTANDO")[0], "FALTANDO")
        self.assertEqual(
            normalize_aoi_text("MUITO ADESIVO")[0],
            "MUITO ADESIVO",
        )

    def test_old_unknown_debug_text_is_recovered(self):
        category, value = normalize_aoi_text(
            "Unknown - Testou: [25S, INVERTIDO]"
        )
        self.assertEqual(category, "INVERTIDO")
        self.assertEqual(value, "INVERTIDO")
        self.assertNotIn("Testou", category)

    def test_legacy_english_categories_map_to_portuguese(self):
        self.assertEqual(normalize_aoi_text("Reverse")[0], "INVERTIDO")
        self.assertEqual(normalize_aoi_text("Up Side Down")[0], "INVERTIDO")
        self.assertEqual(normalize_aoi_text("Missing")[0], "FALTANDO")
        self.assertEqual(
            normalize_aoi_text("Much Adhesive")[0],
            "MUITO ADESIVO",
        )

    def test_unknown_category_no_longer_exposes_tested_tokens(self):
        category, _ = normalize_aoi_text("25S QUALQUER COISA")
        self.assertEqual(category, "Unknown")
        self.assertNotIn("Testou", category)


class AdhesiveRoutingTests(unittest.TestCase):
    def test_adhesive_routes_include_shift(self):
        for category in ("MUITO ADESIVO", "Much Adhesive"):
            with self.subTest(category=category):
                self.assertTrue(is_adhesive_category(category))
                self.assertIn("shift", routes_for_category(category))
                self.assertTrue(
                    ShiftExpert._is_adhesive_context({"category": category})
                )

    def test_non_adhesive_routes_hide_shift(self):
        for category in ("INVERTIDO", "FALTANDO", "Reverse", "Missing"):
            with self.subTest(category=category):
                self.assertFalse(is_adhesive_category(category))
                self.assertNotIn("shift", routes_for_category(category))
                self.assertFalse(
                    ShiftExpert._is_adhesive_context(
                        {
                            "category": category,
                            "value": "Much Adhesive",
                        }
                    )
                )

    def test_old_and_new_categories_share_knn_keys(self):
        self.assertEqual(
            canonical_category_key("Much Adhesive"),
            canonical_category_key("MUITO ADESIVO"),
        )
        self.assertEqual(
            canonical_category_key("Missing"),
            canonical_category_key("FALTANDO"),
        )
        self.assertEqual(
            canonical_category_key("Reverse"),
            canonical_category_key("INVERTIDO"),
        )


if __name__ == "__main__":
    unittest.main()
