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


def missing_body_scene(component_present=True, shift_x=0):
    image = np.full((180, 240, 3), (42, 70, 176), dtype=np.uint8)
    if component_present:
        x1 = 82 + int(shift_x)
        x2 = 158 + int(shift_x)
        cv2.rectangle(image, (x1, 40), (x2, 140), (25, 27, 31), -1)
        cv2.rectangle(image, (x1 + 5, 45), (x2 - 5, 135), (37, 39, 44), 2)
        cv2.line(image, (x1 + 18, 55), (x2 - 18, 55), (230, 230, 220), 5)
        cv2.line(image, (x1 + 18, 85), (x2 - 18, 85), (230, 230, 220), 5)
        cv2.line(image, (x1 + 18, 115), (x2 - 18, 115), (230, 230, 220), 5)
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

    def test_complete_component_disappearance_sets_hard_absence(self):
        reference = missing_body_scene(component_present=True)
        test = missing_body_scene(component_present=False)
        roi = [(82, 40, 77, 101)]

        result = self.expert.analyze(
            reference,
            test,
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=roi,
        )

        self.assertTrue(result["missing_is_defect"])
        self.assertTrue(result["missing_hard_absence"])
        self.assertEqual(
            result["missing_classification"],
            "COMPONENTE FISICAMENTE AUSENTE",
        )
        self.assertGreaterEqual(result["missing_score"], 0.72)
        self.assertGreater(result["missing_changed_coverage"], 0.25)
        self.assertLess(
            result["missing_best_similarity"],
            self.expert.HARD_ABSENCE_MAX_NEARBY_SIMILARITY,
        )
        self.assertIn("AUSÊNCIA FÍSICA FORTE", result["missing_reason"])

    def test_displaced_component_does_not_become_hard_missing(self):
        reference = missing_body_scene(component_present=True)
        test = missing_body_scene(component_present=True, shift_x=18)
        roi = [(82, 40, 77, 101)]

        result = self.expert.analyze(
            reference,
            test,
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=roi,
        )

        self.assertFalse(result["missing_hard_absence"])
        self.assertNotEqual(
            result["missing_classification"],
            "COMPONENTE FISICAMENTE AUSENTE",
        )

    def test_dark_footprint_missing_case_from_real_aoi_becomes_hard_absence(self):
        # Regressão do evento bf6b6a2f1e844cc796ae355ae7ceb7e8:
        # componente removido deixa uma região escura, portanto o sinal de
        # background vermelho é zero e structure_loss fica abaixo de 30%.
        observed = {
            "missing_active": True,
            "missing_is_defect": True,
            "missing_classification": "DIVERGÊNCIA PARCIAL NA ROI",
            "missing_score": 0.964051919402973,
            "missing_changed_coverage": 0.5468713105076741,
            "missing_residual_mean": 0.6723987460136414,
            "missing_structure_loss": 0.23994452149791956,
            "missing_background_exposure": 0.0,
            "missing_best_similarity": 0.13180819153785706,
            "missing_appearance_loss": 0.61,
            "missing_direct_similarity": 0.39,
        }

        hard, reason = self.expert._hard_absence_evidence(observed)

        self.assertTrue(hard)
        self.assertIn("footprint/base", reason)

    def test_dark_footprint_rule_still_rejects_ambiguous_partial_difference(self):
        ambiguous = {
            "missing_active": True,
            "missing_is_defect": True,
            "missing_classification": "DIVERGÊNCIA PARCIAL NA ROI",
            "missing_score": 0.91,
            "missing_changed_coverage": 0.46,
            "missing_residual_mean": 0.61,
            "missing_structure_loss": 0.20,
            "missing_background_exposure": 0.0,
            "missing_best_similarity": 0.48,
            "missing_appearance_loss": 0.58,
            "missing_direct_similarity": 0.42,
        }

        hard, _reason = self.expert._hard_absence_evidence(ambiguous)

        self.assertFalse(hard)

    def test_dark_footprint_rule_never_overrides_probable_displacement(self):
        displaced = {
            "missing_active": True,
            "missing_is_defect": True,
            "missing_classification": "DESLOCAMENTO PROVÁVEL",
            "missing_score": 0.99,
            "missing_changed_coverage": 0.80,
            "missing_residual_mean": 0.80,
            "missing_structure_loss": 0.10,
            "missing_background_exposure": 0.0,
            "missing_best_similarity": 0.10,
            "missing_appearance_loss": 0.80,
            "missing_direct_similarity": 0.20,
        }

        hard, reason = self.expert._hard_absence_evidence(displaced)

        self.assertFalse(hard)
        self.assertIn("deslocado", reason)

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
