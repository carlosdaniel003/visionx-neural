import unittest

import cv2
import numpy as np

from src.core.experts.missing_component_expert import MissingComponentExpert


def board_scene(component=True, shift_x=0, partial=False):
    image = np.full((140, 190, 3), (58, 102, 158), dtype=np.uint8)
    cv2.rectangle(image, (20, 50), (170, 92), (62, 142, 214), -1)
    cv2.rectangle(image, (24, 56), (48, 86), (205, 205, 205), -1)
    cv2.rectangle(image, (142, 56), (166, 86), (205, 205, 205), -1)
    if component:
        x1, x2 = 54 + shift_x, 140 + shift_x
        cv2.rectangle(image, (x1, 40), (x2, 102), (34, 38, 43), -1)
        cv2.rectangle(image, (x1 + 5, 45), (x2 - 5, 97), (58, 62, 67), 2)
        cv2.putText(
            image,
            "102",
            (x1 + 22, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (225, 225, 225),
            2,
            cv2.LINE_AA,
        )
        if partial:
            cv2.rectangle(image, (x1 + 43, 38), (x2 + 3, 105), (62, 142, 214), -1)
    return image


def empty_board_scene(with_intrusion=False):
    image = np.full((120, 170, 3), (52, 112, 164), dtype=np.uint8)
    cv2.line(image, (0, 25), (169, 25), (60, 124, 176), 2)
    cv2.line(image, (0, 95), (169, 95), (60, 124, 176), 2)
    if with_intrusion:
        cv2.rectangle(image, (62, 43), (108, 78), (28, 31, 36), -1)
        cv2.rectangle(image, (66, 47), (104, 74), (210, 210, 210), 2)
    return image


class MissingComponentExpertTests(unittest.TestCase):
    def setUp(self):
        self.expert = MissingComponentExpert()
        self.reference = board_scene(component=True)
        self.roi = [(46, 32, 104, 78)]

    def test_motor_is_inactive_outside_missing_category(self):
        result = self.expert.analyze(
            self.reference,
            board_scene(component=False),
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
        self.assertEqual(result["missing_expectation_mode"], "structure")
        self.assertFalse(result["missing_is_defect"])
        self.assertEqual(result["missing_classification"], "ROI CONFORME")
        self.assertLess(result["missing_score"], result["missing_tolerance"])
        self.assertGreater(result["missing_direct_similarity"], 0.80)

    def test_removed_component_is_detected_as_absent_or_divergent(self):
        result = self.expert.analyze(
            self.reference,
            board_scene(component=False),
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=self.roi,
        )
        self.assertTrue(result["missing_active"])
        self.assertTrue(result["missing_is_defect"])
        self.assertGreater(result["missing_score"], result["missing_tolerance"])
        self.assertIn(
            result["missing_classification"],
            {"COMPONENTE AUSENTE", "ESTRUTURA/ORIENTAÇÃO DIVERGENTE"},
        )
        self.assertGreater(result["missing_changed_coverage"], 0.20)

    def test_shifted_component_is_not_hidden_by_alignment(self):
        result = self.expert.analyze(
            self.reference,
            board_scene(component=True, shift_x=18),
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=self.roi,
        )
        self.assertTrue(result["missing_is_defect"])
        self.assertEqual(result["missing_classification"], "COMPONENTE DESLOCADO")
        self.assertGreater(result["missing_displacement_pixels"], 5.0)
        self.assertGreater(result["missing_best_similarity"], 0.40)

    def test_partial_presence_is_detected(self):
        result = self.expert.analyze(
            self.reference,
            board_scene(component=True, partial=True),
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=self.roi,
        )
        self.assertTrue(result["missing_is_defect"])
        self.assertIn(
            result["missing_classification"],
            {"PRESENÇA PARCIAL", "ESTRUTURA/ORIENTAÇÃO DIVERGENTE"},
        )
        self.assertGreater(result["missing_changed_coverage"], 0.15)

    def test_empty_reference_roi_accepts_clean_background(self):
        reference = empty_board_scene(False)
        roi = [(48, 34, 74, 54)]
        result = self.expert.analyze(
            reference,
            reference.copy(),
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=roi,
        )
        self.assertEqual(result["missing_expectation_mode"], "background")
        self.assertFalse(result["missing_is_defect"])
        self.assertEqual(result["missing_classification"], "ROI CONFORME")

    def test_empty_reference_roi_rejects_unexpected_occupation(self):
        reference = empty_board_scene(False)
        test = empty_board_scene(True)
        roi = [(48, 34, 74, 54)]
        result = self.expert.analyze(
            reference,
            test,
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=roi,
        )
        self.assertEqual(result["missing_expectation_mode"], "background")
        self.assertTrue(result["missing_is_defect"])
        self.assertEqual(result["missing_classification"], "OCUPAÇÃO INDEVIDA")
        self.assertGreater(result["missing_changed_coverage"], 0.12)

    def test_views_and_masks_preserve_roi_dimensions(self):
        result = self.expert.analyze(
            self.reference,
            board_scene(component=False),
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=self.roi,
        )
        expected_shape = (78, 104)
        self.assertEqual(result["roi_anomaly_mask"].shape, expected_shape)
        self.assertEqual(result["missing_reference_view"].shape[:2], expected_shape)
        self.assertEqual(result["missing_test_view"].shape[:2], expected_shape)
        self.assertEqual(result["missing_reconstruction_view"].shape[:2], expected_shape)


if __name__ == "__main__":
    unittest.main()
