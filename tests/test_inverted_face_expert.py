import unittest

import cv2
import numpy as np

from src.core.experts.inverted_face_expert import InvertedFaceExpert


def base_scene():
    return np.full((150, 200, 3), (45, 78, 112), dtype=np.uint8)


def vertical_mark_scene(diagonal=False, illumination_shift=0):
    image = base_scene().astype(np.int16)
    cv2.rectangle(image, (55, 25), (145, 125), (25, 28, 33), -1)
    if diagonal:
        cv2.line(image, (70, 108), (130, 43), (222, 222, 218), 7)
    else:
        cv2.line(image, (100, 42), (100, 108), (222, 222, 218), 7)
    return np.clip(image + illumination_shift, 0, 255).astype(np.uint8)


def triangle_scene(rotated=False, removed=False):
    image = base_scene()
    cv2.rectangle(image, (52, 23), (148, 127), (23, 27, 32), -1)
    if removed:
        return image
    points = np.asarray([[72, 102], [105, 45], [132, 104]], dtype=np.int32)
    if rotated:
        center = np.asarray([102, 76], dtype=np.float32)
        points = np.rint((points.astype(np.float32) - center) * -1.0 + center).astype(np.int32)
    cv2.polylines(image, [points], True, (220, 220, 212), 6, cv2.LINE_AA)
    cv2.line(image, tuple(points[0]), tuple(points[1]), (95, 110, 165), 2)
    return image


def circle_scene(moved=False):
    image = base_scene()
    cv2.rectangle(image, (48, 20), (152, 130), (22, 25, 30), -1)
    center = (100, 26) if moved else (100, 75)
    cv2.circle(image, center, 31, (128, 128, 126), 6, cv2.LINE_AA)
    return image


def double_mark_scene(removed=False):
    image = base_scene()
    cv2.rectangle(image, (45, 35), (155, 115), (21, 24, 29), -1)
    if not removed:
        cv2.circle(image, (82, 74), 18, (122, 122, 118), 5, cv2.LINE_AA)
        cv2.circle(image, (121, 74), 18, (122, 122, 118), 5, cv2.LINE_AA)
    return image


class InvertedFaceExpertTests(unittest.TestCase):
    def setUp(self):
        self.expert = InvertedFaceExpert()
        self.roi = [(50, 20, 100, 110)]

    def test_motor_is_inactive_outside_inverted_category(self):
        result = self.expert.analyze(
            vertical_mark_scene(False),
            vertical_mark_scene(True),
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=self.roi,
        )
        self.assertFalse(result["inverted_active"])
        self.assertFalse(result["inverted_is_defect"])

    def test_identical_witness_is_conforming(self):
        reference = vertical_mark_scene(False)
        result = self.expert.analyze(
            reference,
            reference.copy(),
            aoi_info={"category": "INVERTIDO"},
            aoi_epicenters=self.roi,
        )
        self.assertTrue(result["inverted_active"])
        self.assertFalse(result["inverted_is_defect"])
        self.assertEqual(result["inverted_classification"], "ROI CONFORME")
        self.assertLess(result["inverted_score"], result["inverted_tolerance"])
        self.assertGreater(result["inverted_witness_retention"], 0.85)
        self.assertGreater(result["inverted_direct_similarity"], 0.80)

    def test_vertical_witness_becoming_diagonal_is_detected(self):
        result = self.expert.analyze(
            vertical_mark_scene(False),
            vertical_mark_scene(True),
            aoi_info={"category": "INVERTIDO"},
            aoi_epicenters=self.roi,
        )
        self.assertTrue(result["inverted_is_defect"])
        self.assertGreater(result["inverted_score"], result["inverted_tolerance"])
        self.assertGreater(result["inverted_orientation_mismatch"], 0.20)
        self.assertGreater(result["inverted_topology_mismatch"], 0.20)
        self.assertIn(
            result["inverted_classification"],
            {
                "ORIENTAÇÃO DA MARCA DIVERGENTE",
                "ASSINATURA DA FACE DIVERGENTE",
                "FACE ALTERNATIVA PROVÁVEL",
                "MARCA TESTEMUNHA DESLOCADA",
            },
        )

    def test_expected_triangle_removed_is_detected_as_witness_loss(self):
        result = self.expert.analyze(
            triangle_scene(False),
            triangle_scene(removed=True),
            aoi_info={"category": "INVERTIDO"},
            aoi_epicenters=self.roi,
        )
        self.assertTrue(result["inverted_is_defect"])
        self.assertGreater(result["inverted_witness_loss"], 0.45)
        self.assertGreater(result["inverted_feature_loss"], 0.35)
        self.assertIn(
            result["inverted_classification"],
            {
                "MARCA TESTEMUNHA AUSENTE",
                "MARCA ESPERADA AUSENTE",
                "FACE ALTERNATIVA PROVÁVEL",
                "ASSINATURA DA FACE DIVERGENTE",
            },
        )

    def test_circle_leaving_expected_position_is_detected(self):
        result = self.expert.analyze(
            circle_scene(False),
            circle_scene(True),
            aoi_info={"category": "INVERTIDO"},
            aoi_epicenters=self.roi,
        )
        self.assertTrue(result["inverted_is_defect"])
        self.assertGreater(result["inverted_witness_loss"], 0.30)
        self.assertGreater(result["inverted_changed_coverage"], 0.05)

    def test_double_mark_disappearing_is_detected(self):
        result = self.expert.analyze(
            double_mark_scene(False),
            double_mark_scene(True),
            aoi_info={"category": "INVERTIDO"},
            aoi_epicenters=self.roi,
        )
        self.assertTrue(result["inverted_is_defect"])
        self.assertGreater(result["inverted_feature_loss"], 0.35)
        self.assertLess(result["inverted_witness_retention"], 0.65)

    def test_rotated_asymmetric_mark_keeps_transform_diagnostic(self):
        result = self.expert.analyze(
            triangle_scene(False),
            triangle_scene(rotated=True),
            aoi_info={"category": "INVERTIDO"},
            aoi_epicenters=self.roi,
        )
        self.assertTrue(result["inverted_is_defect"])
        self.assertGreater(result["inverted_best_transform_similarity"], 0.35)
        self.assertNotEqual(result["inverted_best_transform"], "none")
        self.assertGreaterEqual(result["inverted_transform_gain"], 0.0)

    def test_global_illumination_shift_is_normalized(self):
        result = self.expert.analyze(
            vertical_mark_scene(False),
            vertical_mark_scene(False, illumination_shift=12),
            aoi_info={"category": "INVERTIDO"},
            aoi_epicenters=self.roi,
        )
        self.assertFalse(result["inverted_is_defect"])
        self.assertLess(result["inverted_score"], result["inverted_tolerance"])
        self.assertGreater(result["inverted_witness_retention"], 0.70)

    def test_views_and_masks_preserve_exact_roi_dimensions(self):
        result = self.expert.analyze(
            vertical_mark_scene(False),
            vertical_mark_scene(True),
            aoi_info={"category": "INVERTIDO"},
            aoi_epicenters=self.roi,
        )
        expected_shape = (110, 100)
        self.assertEqual(result["inverted_witness_mask"].shape, expected_shape)
        self.assertEqual(result["inverted_anomaly_mask"].shape, expected_shape)
        self.assertEqual(result["inverted_residual_map"].shape, expected_shape)
        self.assertEqual(result["inverted_reference_view"].shape[:2], expected_shape)
        self.assertEqual(result["inverted_test_view"].shape[:2], expected_shape)
        self.assertEqual(result["inverted_reconstruction_view"].shape[:2], expected_shape)


if __name__ == "__main__":
    unittest.main()
