import unittest

import cv2
import numpy as np

from src.core.experts.missing_component_expert import MissingComponentExpert


def component_scene(missing=False):
    image = np.full((120, 160, 3), (55, 95, 155), dtype=np.uint8)
    cv2.rectangle(image, (25, 47), (135, 73), (65, 150, 220), -1)
    cv2.rectangle(image, (28, 51), (48, 69), (205, 205, 205), -1)
    cv2.rectangle(image, (112, 51), (132, 69), (205, 205, 205), -1)
    if not missing:
        cv2.rectangle(image, (48, 38), (112, 82), (35, 38, 42), -1)
        cv2.rectangle(image, (52, 42), (108, 78), (55, 58, 62), 2)
        cv2.putText(
            image,
            "102",
            (61, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
    return image


class MissingComponentExpertTests(unittest.TestCase):
    def setUp(self):
        self.expert = MissingComponentExpert()
        self.reference = component_scene(missing=False)
        self.roi = [(18, 28, 124, 64)]

    def test_motor_is_inactive_outside_missing_category(self):
        result = self.expert.analyze(
            self.reference,
            component_scene(missing=True),
            aoi_info={"category": "INVERTIDO"},
            aoi_epicenters=self.roi,
        )
        self.assertFalse(result["missing_active"])
        self.assertFalse(result["missing_is_defect"])

    def test_identical_component_is_preserved(self):
        result = self.expert.analyze(
            self.reference,
            self.reference.copy(),
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=self.roi,
        )
        self.assertTrue(result["missing_active"])
        self.assertFalse(result["missing_is_defect"])
        self.assertLess(result["missing_score"], result["missing_tolerance"])
        self.assertGreater(result["missing_presence_retention"], 0.75)

    def test_removed_component_is_detected(self):
        result = self.expert.analyze(
            self.reference,
            component_scene(missing=True),
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=self.roi,
        )
        self.assertTrue(result["missing_active"])
        self.assertTrue(result["missing_is_defect"])
        self.assertGreater(result["missing_score"], result["missing_tolerance"])
        self.assertGreater(result["missing_structure_loss"], 0.35)
        self.assertGreater(result["missing_coverage"], 0.20)

    def test_views_and_masks_preserve_roi_dimensions(self):
        result = self.expert.analyze(
            self.reference,
            component_scene(missing=True),
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=self.roi,
        )
        expected_shape = (64, 124)
        self.assertEqual(result["component_missing_mask"].shape, expected_shape)
        self.assertEqual(result["missing_reference_view"].shape[:2], expected_shape)
        self.assertEqual(result["missing_test_view"].shape[:2], expected_shape)
        self.assertEqual(result["missing_reconstruction_view"].shape[:2], expected_shape)


if __name__ == "__main__":
    unittest.main()
