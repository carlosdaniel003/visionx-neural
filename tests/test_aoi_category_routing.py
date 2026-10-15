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


class CategoryOnlyGatekeeperTests(unittest.TestCase):
    def test_deslocado_preserves_shift_tolerance_without_value(self):
        from unittest.mock import patch
        import numpy as np
        from src.core.shift_gatekeeper import ShiftGatekeeper

        image = np.zeros((120, 120, 3), dtype=np.uint8)
        box = {"w": 100, "h": 100}
        with patch(
            "src.core.shift_gatekeeper.cv2.phaseCorrelate",
            return_value=((6.0, 0.0), 1.0),
        ):
            keeper = ShiftGatekeeper()
            standard = keeper.check_global_shift(
                image, image, box, {"category": "FALTANDO", "value": ""}
            )
            displaced = keeper.check_global_shift(
                image, image, box, {"category": "DESLOCADO", "value": ""}
            )
            legacy = keeper.check_global_shift(
                image, image, box, {"value": "Shifted"}
            )
        self.assertFalse(standard["is_critical_shift"])
        self.assertTrue(displaced["is_critical_shift"])
        self.assertEqual(
            displaced["is_critical_shift"], legacy["is_critical_shift"]
        )

    def test_silk_tolerance_and_reverse_epicenter_use_category(self):
        from unittest.mock import patch
        import numpy as np
        from src.core.silkscreen_gatekeeper import SilkscreenGatekeeper

        image = np.zeros((120, 120, 3), dtype=np.uint8)
        contour = np.array(
            [[[5, 5]], [[45, 5]], [[45, 45]], [[5, 45]]],
            dtype=np.int32,
        )
        with patch(
            "src.core.silkscreen_gatekeeper.cv2.phaseCorrelate",
            return_value=((0.0, 0.0), 1.0),
        ), patch(
            "src.core.silkscreen_gatekeeper.cv2.countNonZero",
            return_value=400,
        ), patch(
            "src.core.silkscreen_gatekeeper.cv2.findContours",
            return_value=([contour], None),
        ):
            keeper = SilkscreenGatekeeper()
            normal = keeper.check_silkscreen_anomaly(
                image, image, {"category": "FALTANDO", "value": ""}
            )
            shifted = keeper.check_silkscreen_anomaly(
                image, image, {"category": "DESLOCADO", "value": ""}
            )
            inverted = keeper.check_silkscreen_anomaly(
                image, image,
                {"category": "INVERTIDO", "value": ""},
                [(1000, 1000, 20, 20)],
            )
            legacy = keeper.check_silkscreen_anomaly(
                image, image,
                {"value": "REVERSE"},
                [(1000, 1000, 20, 20)],
            )
        self.assertTrue(normal["is_critical_silk"])
        self.assertFalse(shifted["is_critical_silk"])
        self.assertFalse(inverted["is_critical_silk"])
        self.assertEqual(
            inverted["is_critical_silk"], legacy["is_critical_silk"]
        )


if __name__ == "__main__":
    unittest.main()
