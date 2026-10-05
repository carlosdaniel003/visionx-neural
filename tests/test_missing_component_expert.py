import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.core.experts.dual_scale_presence import DualScalePresenceAnalyzer
from src.core.experts.missing_component_expert import MissingComponentExpert
from src.core.experts.roi_patch_expert import ROIPatchExpectationExpert


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


def component_body_variant_scene():
    """Mesmo corpo físico, mas brilho e marcação interna bem diferentes."""
    image = np.full((180, 240, 3), (42, 70, 176), dtype=np.uint8)
    x1, x2 = 82, 158
    cv2.rectangle(image, (x1, 40), (x2, 140), (83, 87, 92), -1)
    cv2.rectangle(image, (x1 + 5, 45), (x2 - 5, 135), (104, 108, 114), 2)
    cv2.line(image, (x1 + 14, 52), (x1 + 14, 128), (185, 188, 192), 4)
    cv2.line(image, (x1 + 26, 52), (x1 + 26, 128), (185, 188, 192), 4)
    cv2.line(image, (x1 + 39, 72), (x2 - 10, 72), (205, 205, 198), 4)
    cv2.line(image, (x1 + 39, 105), (x2 - 18, 105), (150, 135, 110), 4)
    return image


def real_like_tall_component_scene(variant=False, present=True):
    """Envelope alto inspirado no caso R9*5, com marcação interna variável."""
    image = np.full((540, 355, 3), (26, 28, 32), dtype=np.uint8)

    cv2.rectangle(image, (26, 26), (332, 539), (0, 255, 0), 2)
    cv2.rectangle(image, (128, 60), (228, 539), (0, 255, 0), 2)

    cv2.rectangle(image, (50, 28), (310, 180), (52, 58, 180), -1)
    cv2.rectangle(image, (50, 360), (310, 520), (52, 58, 180), -1)
    cv2.rectangle(image, (58, 155), (302, 205), (205, 205, 190), -1)
    cv2.rectangle(image, (58, 335), (302, 385), (205, 205, 190), -1)

    if present:
        body = (35, 37, 42) if not variant else (45, 47, 53)
        cv2.rectangle(image, (52, 190), (308, 350), body, -1)
        cv2.rectangle(image, (72, 205), (288, 335), (24, 25, 29), -1)

        if not variant:
            cv2.line(image, (145, 235), (215, 235), (222, 222, 214), 8)
            cv2.line(image, (145, 275), (205, 275), (222, 222, 214), 8)
            cv2.line(image, (145, 315), (215, 315), (222, 222, 214), 8)
        else:
            cv2.line(image, (150, 235), (215, 235), (225, 225, 218), 8)
            cv2.line(image, (150, 275), (215, 275), (225, 225, 218), 8)
            cv2.line(image, (160, 315), (220, 315), (225, 225, 218), 8)

    return image


def shifted_global_component_scene(shift_y=0, variant=False, present=True):
    """Corpo escuro preservado, mas deslocado dentro do envelope global."""
    image = np.full((540, 355, 3), (74, 88, 118), dtype=np.uint8)

    # Região fora da caixa global permanece estável para medir fundo real.
    cv2.rectangle(image, (26, 26), (333, 539), (0, 255, 0), 2)

    # Pads/placa permanecem como contexto.
    cv2.rectangle(image, (48, 28), (310, 150), (56, 70, 155), -1)
    cv2.rectangle(image, (48, 390), (310, 520), (56, 70, 155), -1)

    if present:
        top = 165 + int(shift_y)
        bottom = top + 220
        cv2.rectangle(image, (78, top), (290, bottom), (24, 27, 32), -1)
        cv2.rectangle(
            image,
            (88, top + 12),
            (280, bottom - 12),
            (34, 37, 42) if not variant else (43, 47, 52),
            2,
        )
        # Terminais metálicos permanecem, marcação interna muda.
        cv2.rectangle(
            image,
            (72, top - 18),
            (296, top + 18),
            (210, 210, 196),
            -1,
        )
        cv2.rectangle(
            image,
            (72, bottom - 18),
            (296, bottom + 18),
            (210, 210, 196),
            -1,
        )
        if variant:
            cv2.line(
                image,
                (145, top + 55),
                (225, top + 55),
                (235, 235, 225),
                8,
            )
            cv2.line(
                image,
                (155, top + 110),
                (235, top + 110),
                (235, 235, 225),
                8,
            )
        else:
            cv2.line(
                image,
                (135, top + 70),
                (220, top + 70),
                (230, 230, 220),
                8,
            )
            cv2.line(
                image,
                (145, top + 130),
                (215, top + 130),
                (230, 230, 220),
                8,
            )
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

    def test_component_envelope_can_confirm_presence_when_local_patch_is_misleading(self):
        reference = missing_body_scene(component_present=True)
        test = component_body_variant_scene()

        # ROI interna concentrada em marcação/serigrafia: pode divergir muito.
        local_box = (104, 54, 34, 72)
        # Epicentro final envolve o corpo físico completo.
        epicenter = [(82, 40, 77, 101)]

        local_presence = self.expert._component_body_presence(
            reference[54:126, 104:138],
            test[54:126, 104:138],
        )
        envelope_presence = self.expert._component_body_presence_witness(
            reference,
            test,
            local_box,
            epicenter,
        )

        self.assertTrue(envelope_presence["missing_component_body_present"])
        self.assertEqual(
            envelope_presence["missing_body_presence_source"],
            "aoi_epicenter",
        )
        self.assertEqual(
            envelope_presence["missing_body_presence_box"],
            [82, 40, 77, 101],
        )
        self.assertGreater(
            envelope_presence["missing_body_silhouette_dice"],
            local_presence.get("missing_body_silhouette_dice", 0.0),
        )

    def test_same_component_body_with_different_marking_blocks_false_missing(self):
        reference = missing_body_scene(component_present=True)
        test = component_body_variant_scene()
        roi = [(82, 40, 77, 101)]

        presence = self.expert._component_body_presence(
            reference[40:141, 82:159],
            test[40:141, 82:159],
        )

        self.assertTrue(presence["missing_component_body_present"])
        self.assertEqual(
            presence["missing_body_presence_policy"],
            "geometry_only",
        )
        self.assertLess(
            presence["missing_body_coarse_similarity"],
            self.expert.BODY_PRESENCE_MIN_COARSE_SIMILARITY,
        )
        self.assertGreaterEqual(
            presence["missing_body_silhouette_dice"],
            self.expert.BODY_GEOMETRY_MIN_SILHOUETTE_DICE,
        )
        self.assertGreaterEqual(
            presence["missing_body_area_ratio"],
            self.expert.BODY_GEOMETRY_MIN_AREA_RATIO,
        )
        self.assertLessEqual(
            presence["missing_body_area_ratio"],
            self.expert.BODY_GEOMETRY_MAX_AREA_RATIO,
        )

        result = self.expert.analyze(
            reference,
            test,
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=roi,
        )

        self.assertTrue(result["missing_component_body_present"])
        self.assertFalse(result["missing_hard_absence"])
        self.assertFalse(result["missing_is_defect"])
        self.assertTrue(result["missing_body_presence_veto"])
        self.assertEqual(
            result["missing_classification"],
            "COMPONENTE PRESENTE — APARÊNCIA DIVERGENTE",
        )
        self.assertIn("CORPO DO COMPONENTE PRESENTE", result["missing_reason"])

    def test_body_presence_does_not_hide_localized_non_missing_anomaly(self):
        result = self.expert.analyze(
            self.dark_reference,
            dark_body_scene(True),
            aoi_info={"category": "FALTANDO"},
            aoi_epicenters=self.dark_roi,
        )

        if result.get("missing_component_body_present", False):
            self.assertFalse(result["missing_hard_absence"])
            self.assertFalse(result["missing_body_presence_veto"])
            self.assertTrue(result["missing_is_defect"])

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
        self.assertFalse(result["missing_component_body_present"])
        self.assertNotEqual(
            result.get("missing_body_presence_policy"),
            "geometry_only",
        )
        self.assertIn(
            result.get("missing_body_presence_source"),
            {"missing_roi", "aoi_epicenter"},
        )
        self.assertFalse(result["missing_body_presence_veto"])
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

    def test_global_envelope_invariant_mass_survives_internal_shift_and_marking_change(self):
        reference = shifted_global_component_scene(
            shift_y=0,
            variant=False,
            present=True,
        )
        test = shifted_global_component_scene(
            shift_y=-58,
            variant=True,
            present=True,
        )

        evidence = self.expert._global_envelope_presence_support(
            reference,
            test,
            {
                "x": 26,
                "y": 26,
                "w": 308,
                "h": 514,
                "detected": True,
            },
        )

        self.assertTrue(evidence["missing_global_envelope_active"])
        self.assertTrue(evidence["missing_global_envelope_invariant_support"])
        # A rota invariável é auxiliar: sozinha não possui autoridade para
        # vetar hard missing, pois footprint escuro real pode preservar massa.
        self.assertFalse(evidence["missing_global_envelope_support"])
        self.assertLessEqual(
            evidence["missing_global_envelope_background_exposure"],
            self.expert.GLOBAL_ENVELOPE_INVARIANT_MAX_BACKGROUND_EXPOSURE,
        )
        self.assertGreaterEqual(
            evidence["missing_global_envelope_reference_dark_fraction"],
            self.expert.GLOBAL_ENVELOPE_MIN_DARK_REFERENCE_FRACTION,
        )
        self.assertGreaterEqual(
            evidence["missing_global_envelope_test_dark_fraction"],
            self.expert.GLOBAL_ENVELOPE_MIN_DARK_TEST_FRACTION,
        )
        self.assertGreaterEqual(
            evidence["missing_global_envelope_invariant_row_profile"],
            self.expert.GLOBAL_ENVELOPE_MIN_INVARIANT_ROW_PROFILE,
        )
        self.assertGreaterEqual(
            evidence["missing_global_envelope_invariant_col_profile"],
            self.expert.GLOBAL_ENVELOPE_MIN_INVARIANT_COL_PROFILE,
        )
        self.assertIn("massa física", evidence["missing_global_envelope_reason"])

    def test_global_envelope_invariant_mass_rejects_real_disappearance(self):
        reference = shifted_global_component_scene(
            shift_y=0,
            variant=False,
            present=True,
        )
        test = shifted_global_component_scene(
            shift_y=0,
            variant=False,
            present=False,
        )

        evidence = self.expert._global_envelope_presence_support(
            reference,
            test,
            {
                "x": 26,
                "y": 26,
                "w": 308,
                "h": 514,
                "detected": True,
            },
        )

        self.assertTrue(evidence["missing_global_envelope_active"])
        self.assertFalse(evidence["missing_global_envelope_invariant_support"])
        self.assertFalse(evidence["missing_global_envelope_support"])
        self.assertLess(
            evidence["missing_global_envelope_test_dark_fraction"],
            self.expert.GLOBAL_ENVELOPE_MIN_DARK_TEST_FRACTION,
        )

    def test_real_event_fb7de76_uses_global_invariant_presence_to_block_hard_missing(self):
        reference = shifted_global_component_scene(
            shift_y=0,
            variant=False,
            present=True,
        )
        test = shifted_global_component_scene(
            shift_y=-58,
            variant=True,
            present=True,
        )
        observed = {
            "missing_active": True,
            "missing_is_defect": True,
            "missing_classification": "COMPONENTE FISICAMENTE AUSENTE",
            "missing_score": 0.9961602686328229,
            "missing_changed_coverage": 0.6857581742374369,
            "missing_residual_mean": 0.7836465239524841,
            "missing_residual_p90": 1.0,
            "missing_structure_loss": 0.48948475289169296,
            "missing_background_exposure": 0.6269353181123734,
            "missing_best_similarity": 0.4097987413406372,
            "missing_appearance_loss": 0.6868006474077814,
            "missing_direct_similarity": 0.31319935259221865,
            "missing_roi_box": (39, 105, 279, 147),
            "missing_reason": "QUEBRA DA EXPECTATIVA VISUAL DA ROI",
        }

        with patch.object(
            ROIPatchExpectationExpert,
            "analyze",
            return_value=dict(observed),
        ), patch.object(
            self.expert,
            "_component_body_presence_witness",
            return_value={
                "missing_body_presence_active": True,
                "missing_component_body_present": False,
                "missing_body_presence_source": "missing_roi",
                "missing_body_presence_box": [39, 105, 279, 147],
                "missing_body_presence_policy": "none",
                "missing_body_presence_reason": (
                    "ROI local não confirmou o corpo completo"
                ),
            },
        ):
            result = self.expert.analyze(
                reference,
                test,
                global_box_info={
                    "x": 26,
                    "y": 26,
                    "w": 308,
                    "h": 514,
                    "detected": True,
                },
                aoi_info={"category": "FALTANDO"},
                aoi_epicenters=[(39, 105, 279, 147)],
            )

        self.assertTrue(result["missing_global_envelope_invariant_support"])
        self.assertFalse(result["missing_global_envelope_support"])
        self.assertFalse(result["missing_global_envelope_veto"])
        # A massa invariável é somente evidência auxiliar. Sem a memória KNN,
        # o especialista físico não pode transformar sozinho esse caso em OK.
        self.assertTrue(result["missing_hard_absence"])
        self.assertTrue(result["missing_is_defect"])
        self.assertFalse(result["missing_dual_scale_active"])

    def test_real_event_c5a70_small_roi_plus_invariant_occupancy_blocks_hard_missing(self):
        reference = np.zeros((331, 570, 3), dtype=np.uint8)
        test = reference.copy()
        observed = {
            "missing_active": True,
            "missing_is_defect": True,
            "missing_classification": "CONTEÚDO ESPERADO AUSENTE",
            "missing_score": 0.8937784610380831,
            "missing_changed_coverage": 0.4634194272880404,
            "missing_residual_mean": 0.6930411458015442,
            "missing_residual_p90": 0.8399999737739563,
            "missing_structure_loss": 0.46972477064220186,
            "missing_background_exposure": 0.47104059159755707,
            "missing_best_similarity": 0.41752538084983826,
            "missing_appearance_loss": 0.49676110615538727,
            "missing_direct_similarity": 0.5032388938446127,
            "missing_roi_box": (335, 36, 137, 260),
            "missing_reason": "QUEBRA DA EXPECTATIVA VISUAL DA ROI",
        }
        body = {
            "missing_body_presence_active": True,
            "missing_component_body_present": False,
            "missing_body_presence_source": "missing_roi",
            "missing_body_presence_box": [335, 36, 137, 260],
            "missing_body_coarse_similarity": 0.40989190340042114,
            "missing_body_silhouette_dice": 0.7397280304138032,
            "missing_body_area_ratio": 0.8858403419274784,
            "missing_body_centroid_shift": 0.08592147903990828,
            "missing_body_box_width_ratio": 1.0,
            "missing_body_box_height_ratio": 1.0,
            "missing_body_presence_policy": "none",
            "missing_body_presence_reason": (
                "sem testemunha geométrica suficiente de corpo preservado"
            ),
        }
        envelope = {
            "missing_global_envelope_active": True,
            "missing_global_envelope_support": False,
            "missing_global_envelope_veto": False,
            "missing_global_envelope_box": [25, 25, 525, 285],
            "missing_global_envelope_row_profile": 0.941638708114624,
            "missing_global_envelope_col_profile": 0.6714836359024048,
            "missing_global_envelope_coarse_similarity": 0.6775625944137573,
            "missing_global_envelope_background_exposure": 0.05692705512046814,
            "missing_global_envelope_dark_threshold": 58.0,
            "missing_global_envelope_reference_dark_fraction": 0.4337540566921234,
            "missing_global_envelope_test_dark_fraction": 0.5052304863929749,
            "missing_global_envelope_dark_retention": 1.1647856166370913,
            "missing_global_envelope_invariant_row_profile": 0.9056051969528198,
            "missing_global_envelope_invariant_col_profile": 0.9251869916915894,
            "missing_global_envelope_invariant_support": True,
            "missing_global_envelope_reason": (
                "massa física invariável preservada; requer testemunha OK forte "
                "antes de contrariar hard missing"
            ),
        }

        with patch.object(
            ROIPatchExpectationExpert,
            "analyze",
            return_value=dict(observed),
        ), patch.object(
            self.expert,
            "_component_body_presence_witness",
            return_value=dict(body),
        ), patch.object(
            self.expert,
            "_global_envelope_presence_support",
            return_value=dict(envelope),
        ):
            result = self.expert.analyze(
                reference,
                test,
                global_box_info={
                    "x": 25,
                    "y": 25,
                    "w": 525,
                    "h": 285,
                    "detected": True,
                },
                aoi_info={"category": "FALTANDO"},
                aoi_epicenters=[(335, 36, 137, 260)],
            )

        self.assertAlmostEqual(
            result["missing_local_global_area_ratio"],
            (137 * 260) / (525 * 285),
            places=3,
        )
        self.assertTrue(result["missing_invariant_occupancy_support"])
        self.assertTrue(result["missing_invariant_occupancy_veto"])
        self.assertFalse(result["missing_hard_absence"])
        self.assertFalse(result["missing_dual_scale_active"])
        self.assertTrue(result["missing_is_defect"])
        self.assertIn(
            "ocupação geométrica local preservadas",
            result["missing_hard_absence_reason"],
        )

    def test_invariant_mass_does_not_veto_real_missing_when_local_occupancy_collapses(self):
        result = {
            "missing_roi_box": (335, 36, 137, 260),
            "missing_global_envelope_invariant_support": True,
            "missing_body_silhouette_dice": 0.42,
            "missing_body_area_ratio": 0.41,
            "missing_body_centroid_shift": 0.22,
            "missing_body_box_width_ratio": 0.62,
            "missing_body_box_height_ratio": 0.58,
        }

        support = self.expert._invariant_occupancy_presence_support(
            result,
            (331, 570, 3),
            {
                "x": 25,
                "y": 25,
                "w": 525,
                "h": 285,
                "detected": True,
            },
        )

        self.assertLessEqual(
            support["missing_local_global_area_ratio"],
            DualScalePresenceAnalyzer.MAX_LOCAL_GLOBAL_AREA_RATIO,
        )
        self.assertFalse(support["missing_invariant_occupancy_support"])
        self.assertFalse(support["missing_invariant_occupancy_veto"])
        self.assertIn(
            "não confirmou presença",
            support["missing_invariant_occupancy_reason"],
        )

    def test_global_aoi_envelope_can_downgrade_false_hard_missing_to_normal_fusion(self):
        reference = real_like_tall_component_scene(
            variant=False,
            present=True,
        )
        test = real_like_tall_component_scene(
            variant=True,
            present=True,
        )
        observed = {
            "missing_active": True,
            "missing_is_defect": True,
            "missing_classification": "QUEBRA DA EXPECTATIVA VISUAL",
            "missing_score": 0.9650457569485484,
            "missing_changed_coverage": 0.5742780528052805,
            "missing_residual_mean": 0.592208743095398,
            "missing_structure_loss": 0.5261014439096631,
            "missing_background_exposure": 0.0,
            "missing_best_similarity": 0.26161158084869385,
            "missing_appearance_loss": 0.5514829146344776,
            "missing_direct_similarity": 0.44851708536552237,
            "missing_roi_box": (128, 60, 101, 480),
        }

        with patch.object(
            ROIPatchExpectationExpert,
            "analyze",
            return_value=dict(observed),
        ), patch.object(
            self.expert,
            "_component_body_presence_witness",
            return_value={
                "missing_body_presence_active": True,
                "missing_component_body_present": False,
                "missing_body_presence_source": "missing_roi",
                "missing_body_presence_box": [128, 60, 101, 480],
                "missing_body_presence_policy": "none",
                "missing_body_presence_reason": (
                    "ROI interna não prova presença"
                ),
            },
        ):
            result = self.expert.analyze(
                reference,
                test,
                global_box_info={
                    "x": 26,
                    "y": 26,
                    "w": 307,
                    "h": 514,
                    "detected": True,
                },
                aoi_info={"category": "FALTANDO"},
                aoi_epicenters=[(128, 60, 101, 480)],
            )

        self.assertTrue(result["missing_global_envelope_active"])
        self.assertTrue(result["missing_global_envelope_support"])
        self.assertTrue(result["missing_global_envelope_veto"])
        self.assertGreaterEqual(
            result["missing_global_envelope_row_profile"],
            self.expert.GLOBAL_ENVELOPE_MIN_ROW_PROFILE,
        )
        self.assertGreaterEqual(
            result["missing_global_envelope_col_profile"],
            self.expert.GLOBAL_ENVELOPE_MIN_COL_PROFILE,
        )
        self.assertGreaterEqual(
            result["missing_global_envelope_coarse_similarity"],
            self.expert.GLOBAL_ENVELOPE_MIN_COARSE_SIMILARITY,
        )
        self.assertTrue(result["missing_is_defect"])
        self.assertFalse(result["missing_hard_absence"])
        self.assertFalse(result["missing_dual_scale_active"])
        self.assertIn(
            "fusão normal",
            result["missing_hard_absence_reason"],
        )

    def test_global_envelope_does_not_hide_real_component_disappearance(self):
        reference = real_like_tall_component_scene(
            variant=False,
            present=True,
        )
        test = real_like_tall_component_scene(
            variant=False,
            present=False,
        )
        observed = {
            "missing_active": True,
            "missing_is_defect": True,
            "missing_classification": "QUEBRA DA EXPECTATIVA VISUAL",
            "missing_score": 0.97,
            "missing_changed_coverage": 0.60,
            "missing_residual_mean": 0.68,
            "missing_structure_loss": 0.62,
            "missing_background_exposure": 0.0,
            "missing_best_similarity": 0.18,
            "missing_appearance_loss": 0.68,
            "missing_direct_similarity": 0.32,
            "missing_roi_box": (128, 60, 101, 480),
        }

        with patch.object(
            ROIPatchExpectationExpert,
            "analyze",
            return_value=dict(observed),
        ), patch.object(
            self.expert,
            "_component_body_presence_witness",
            return_value={
                "missing_body_presence_active": True,
                "missing_component_body_present": False,
                "missing_body_presence_source": "missing_roi",
                "missing_body_presence_policy": "none",
            },
        ):
            result = self.expert.analyze(
                reference,
                test,
                global_box_info={
                    "x": 26,
                    "y": 26,
                    "w": 307,
                    "h": 514,
                    "detected": True,
                },
                aoi_info={"category": "FALTANDO"},
                aoi_epicenters=[(128, 60, 101, 480)],
            )

        self.assertTrue(result["missing_global_envelope_active"])
        self.assertFalse(result["missing_global_envelope_support"])
        self.assertFalse(result["missing_global_envelope_veto"])
        self.assertTrue(result["missing_hard_absence"])

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

    def test_dual_scale_context_can_promote_local_conforming_roi_to_hard_missing(self):
        reference = self.dark_reference.copy()
        test = self.dark_reference.copy()

        with patch(
            "src.core.experts.missing_component_expert."
            "DualScalePresenceAnalyzer.analyze",
            return_value={
                "missing_dual_scale_policy": "dual_scale_presence_v1",
                "missing_dual_scale_active": True,
                "missing_dual_scale_triggered": True,
                "missing_scale_disagreement": True,
                "missing_context_hard_absence": True,
                "missing_context_hard_reason": (
                    "contexto maior confirma desaparecimento físico"
                ),
            },
        ):
            result = self.expert.analyze(
                reference,
                test,
                global_box_info={"w": 180, "h": 120},
                aoi_info={"category": "FALTANDO"},
                aoi_epicenters=self.dark_roi,
                physical_detail={
                    "silk_error_pct": 0.56,
                    "semantic_loss": 0.56,
                },
            )

        self.assertTrue(result["missing_hard_absence"])
        self.assertTrue(result["missing_context_hard_absence"])
        self.assertEqual(
            result["missing_classification"],
            "COMPONENTE FISICAMENTE AUSENTE — DUAL-SCALE",
        )
        self.assertIn("contexto maior", result["missing_hard_absence_reason"])

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
