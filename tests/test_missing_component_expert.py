import unittest

import cv2
import numpy as np

from src.core.experts.missing_component_expert import MissingComponentExpert


def dark_body_scene(intrusion=False, illumination_shift=0):
    image = np.full((150, 210, 3), (46, 88, 132), dtype=np.int16)
    cv2.rectangle(image, (50, 28), (160, 125), (30, 34, 39), -1)
    cv2.rectangle(image, (56, 34), (154, 119), (40, 44, 49), 2)
    cv2.line(image, (68, 40), (68, 112), (34, 38, 43), 2)
    if intrusion:
        cv2.rectangle(image, (103, 42), (151, 96), (212, 216, 220), -1)
        cv2.rectangle(image, (108, 47), (146, 91), (175, 150, 115), 3)
    image = np.clip(image + illumination_shift, 0, 255).astype(np.uint8)
    return image


def red_pad_scene(intrusion=False):
    image = np.full((130, 220, 3), (52, 98, 148), dtype=np.uint8)
    cv2.rectangle(image, (92, 26), (176, 110), (45, 70, 190), -1)
    cv2.rectangle(image, (92, 26), (108, 110), (220, 220, 212), -1)
    cv2.rectangle(image, (108, 26), (176, 110), (52, 82, 205), -1)
    if intrusion:
        cv2.rectangle(image, (90, 40), (134, 104), (27, 31, 36), -1)
        cv2.rectangle(image, (126, 42), (151, 100), (205, 205, 198), -1)
    return image


class MissingComponentExpertTests(unittest.TestCase):
    def setUp(self):
        self.expert = MissingComponentExpert()
        self.dark_reference = dark_body_scene(False)
        self.dark_roi = [(78, 36, 78, 78)]
        self.pad_reference = red_pad_scene(False)
        self.pad_roi = [(88, 22, 92, 92)]

    def test_motor_is_inactive_outside_missing_category(self):
        result = self.expert.analyze(
            self.dark_reference,
            dark_body_scene(True),
            aoi_info={"category": "INVERTIDO"},
            aoi_epicenters=self.dark_roi,
        )
        self.assertFalse(result["missing_active"])
        self.assertFalse(result["missing_is_defect"])

    def test_identical_patch_is_conforming(self):
        result = self.expert.analyze(
            self.dark_reference,
            self.dark_reference.copy(),
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=self.dark_roi,
        )
        self.assertTrue(result["missing_active"])
        self.assertEqual(result["missing_expectation_mode"], "patch")
        self.assertFalse(result["missing_is_defect"])
        self.assertEqual(result["missing_classification"], "ROI CONFORME")
        self.assertLess(result["missing_changed_coverage"], 0.05)
        self.assertGreater(result["missing_direct_similarity"], 0.90)

    def test_white_terminal_intruding_into_dark_patch_is_localized(self):
        result = self.expert.analyze(
            self.dark_reference,
            dark_body_scene(True),
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=self.dark_roi,
        )
        self.assertTrue(result["missing_is_defect"])
        self.assertGreater(result["missing_score"], result["missing_tolerance"])
        self.assertGreater(result["missing_changed_coverage"], 0.10)
        self.assertLess(result["missing_changed_coverage"], 0.82)
        self.assertGreater(result["missing_residual_mean"], 0.20)
        self.assertIn(
            result["missing_classification"],
            {
                "CONTEÚDO INESPERADO NA ROI",
                "DIVERGÊNCIA PARCIAL NA ROI",
                "QUEBRA DA EXPECTATIVA VISUAL",
                "DESLOCAMENTO PROVÁVEL",
            },
        )

    def test_dark_component_intruding_into_red_pad_patch_is_detected(self):
        result = self.expert.analyze(
            self.pad_reference,
            red_pad_scene(True),
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=self.pad_roi,
        )
        self.assertTrue(result["missing_is_defect"])
        self.assertGreater(result["missing_changed_coverage"], 0.12)
        self.assertGreater(result["missing_residual_p90"], 0.30)
        self.assertLess(result["missing_direct_similarity"], 0.82)

    def test_global_illumination_change_is_normalized_by_external_context(self):
        result = self.expert.analyze(
            self.dark_reference,
            dark_body_scene(False, illumination_shift=12),
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=self.dark_roi,
        )
        self.assertFalse(result["missing_is_defect"])
        self.assertLess(result["missing_score"], result["missing_tolerance"])
        self.assertLess(result["missing_changed_coverage"], 0.12)

    def test_binary_mask_does_not_fill_the_entire_roi(self):
        result = self.expert.analyze(
            self.dark_reference,
            dark_body_scene(True),
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=self.dark_roi,
        )
        mask = result["roi_anomaly_mask"]
        coverage = float(np.mean(mask > 0))
        self.assertGreater(coverage, 0.08)
        self.assertLess(coverage, 0.82)
        self.assertGreater(np.count_nonzero(mask == 0), 0)

    def test_views_maps_and_masks_preserve_exact_roi_dimensions(self):
        result = self.expert.analyze(
            self.pad_reference,
            red_pad_scene(True),
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=self.pad_roi,
        )
        expected_shape = (92, 92)
        self.assertEqual(result["roi_anomaly_mask"].shape, expected_shape)
        self.assertEqual(result["missing_residual_map"].shape, expected_shape)
        self.assertEqual(result["missing_reference_view"].shape[:2], expected_shape)
        self.assertEqual(result["missing_test_view"].shape[:2], expected_shape)
        self.assertEqual(result["missing_reconstruction_view"].shape[:2], expected_shape)


if __name__ == "__main__":
    unittest.main()
