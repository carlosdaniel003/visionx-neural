import unittest

import cv2
import numpy as np

from src.core.experts.shift_expert import ShiftExpert


class AdhesiveFlowExpertTests(unittest.TestCase):
    @staticmethod
    def _make_scene() -> np.ndarray:
        image = np.zeros((120, 100, 3), dtype=np.uint8)
        image[:] = (35, 70, 100)

        # Padding de cobre.
        image[60:110, 15:85] = (20, 105, 205)

        # Laterais metálicas e corpo do resistor.
        image[20:85, 38:62] = (190, 190, 195)
        image[30:75, 42:58] = (60, 60, 70)

        # Quantidade esperada de adesivo sob o componente.
        image[72:86, 44:58] = (25, 35, 85)
        return image

    def test_motor_is_inactive_outside_adhesive_categories(self):
        reference = self._make_scene()
        result = ShiftExpert().analyze(
            reference,
            reference.copy(),
            aoi_info={"category": "Missing"},
            aoi_epicenters=[(0, 0, 100, 120)],
        )

        self.assertFalse(result["shift_active"])
        self.assertFalse(result["is_defect"])
        self.assertEqual(result["comparison_mode"], "adhesive_flow")

    def test_identical_adhesive_distribution_is_stable(self):
        reference = self._make_scene()
        result = ShiftExpert().analyze(
            reference,
            reference.copy(),
            aoi_info={"category": "Much Adhesive"},
            aoi_epicenters=[(0, 0, 100, 120)],
        )

        self.assertTrue(result["shift_active"])
        self.assertFalse(result["is_defect"])
        self.assertLess(result["adhesive_score"], 0.02)
        self.assertEqual(result["excess_coverage"], 0.0)
        self.assertEqual(result["padding_overlap"], 0.0)
        self.assertEqual(result["adhesive_direction"], "ESTÁVEL")

    def test_excess_adhesive_over_padding_is_detected(self):
        reference = self._make_scene()
        test = reference.copy()
        test[75:108, 42:75] = (25, 35, 85)

        result = ShiftExpert().analyze(
            reference,
            test,
            aoi_info={"category": "Much Adhesive"},
            aoi_epicenters=[(0, 0, 100, 120)],
        )

        self.assertTrue(result["is_defect"])
        self.assertGreater(result["adhesive_score"], result["tolerance"])
        self.assertGreater(result["excess_coverage"], 0.04)
        self.assertGreater(result["padding_overlap"], 0.03)
        self.assertGreater(result["area_growth_ratio"], 0.50)
        self.assertGreater(result["lower_leakage_ratio"], 0.70)
        self.assertIn("BAIXO", result["adhesive_direction"])
        self.assertIsNotNone(result["bounding_box"])

    def test_stable_copper_metal_and_component_are_not_marked_as_excess(self):
        reference = self._make_scene()
        result = ShiftExpert().analyze(
            reference,
            reference.copy(),
            aoi_info={"value": "ADESIVO"},
            aoi_epicenters=[(0, 0, 100, 120)],
        )

        self.assertEqual(cv2.countNonZero(result["excess_mask"]), 0)
        self.assertEqual(cv2.countNonZero(result["padding_overlap_mask"]), 0)
        self.assertLess(result["adhesive_score"], result["tolerance"])

    def test_views_preserve_the_exact_roi_dimensions(self):
        reference = self._make_scene()
        test = reference.copy()
        test[80:105, 50:80] = (20, 30, 100)

        result = ShiftExpert().analyze(
            reference,
            test,
            aoi_info={"category": "Much Adhesive"},
            aoi_epicenters=[(10, 12, 78, 92)],
        )

        self.assertEqual(result["roi_box"], (10, 12, 78, 92))
        self.assertEqual(result["reference_view"].shape[:2], (92, 78))
        self.assertEqual(result["test_view"].shape[:2], (92, 78))
        self.assertEqual(result["flow_view"].shape[:2], (92, 78))


if __name__ == "__main__":
    unittest.main()
