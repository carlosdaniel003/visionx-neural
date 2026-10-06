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

    @staticmethod
    def _make_mid_scene() -> np.ndarray:
        """Cena clara que reproduz o aspecto fotométrico da iluminação MID."""
        image = np.full((123, 68, 3), 245, dtype=np.uint8)

        # Corpo e lateral do componente dão estrutura suficiente ao alinhamento,
        # enquanto a região à esquerda permanece quase branca no gabarito.
        image[18:104, 34:64] = (205, 205, 210)
        image[28:94, 40:60] = (70, 70, 75)

        # Região estável levemente quente: não pode virar adesivo por si só.
        image[105:120, 12:60] = (205, 220, 240)
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

    def test_mid_profile_detects_bright_cream_adhesive_witness(self):
        reference = self._make_mid_scene()
        test = reference.copy()

        # Filme claro/amarelado semelhante ao observado na iluminação MID real:
        # não é escuro o suficiente para o perfil legado, mas diverge do branco.
        test[38:88, 24:30] = (210, 225, 245)

        result = ShiftExpert().analyze(
            reference,
            test,
            aoi_info={
                "category": "Much Adhesive",
                "lighting_mode": "MID",
            },
            aoi_epicenters=[(0, 0, 68, 123)],
        )

        self.assertEqual(
            result["adhesive_detector_profile"],
            "mid_bright_resin_v1",
        )
        self.assertEqual(result["adhesive_lighting_mode"], "MID")
        self.assertTrue(result["adhesive_is_defect"])
        self.assertGreater(
            result["adhesive_score"],
            result["adhesive_tolerance"],
        )
        self.assertGreater(result["mid_bright_witness_coverage"], 0.01)
        self.assertGreater(result["mid_bright_witness_peak"], 0.20)
        self.assertGreater(result["mid_bright_witness_score"], 0.20)
        self.assertGreater(
            cv2.countNonZero(result["mid_bright_witness_mask"]),
            0,
        )

    def test_same_bright_cream_region_is_not_enabled_for_top_profile(self):
        reference = self._make_mid_scene()
        test = reference.copy()
        test[38:88, 24:30] = (210, 225, 245)

        result = ShiftExpert().analyze(
            reference,
            test,
            aoi_info={
                "category": "Much Adhesive",
                "lighting_mode": "TOP",
            },
            aoi_epicenters=[(0, 0, 68, 123)],
        )

        self.assertEqual(
            result["adhesive_detector_profile"],
            "dark_warm_v1",
        )
        self.assertEqual(result["mid_bright_witness_coverage"], 0.0)
        self.assertEqual(
            cv2.countNonZero(result["mid_bright_witness_mask"]),
            0,
        )

    def test_mid_identical_bright_scene_remains_stable(self):
        reference = self._make_mid_scene()

        result = ShiftExpert().analyze(
            reference,
            reference.copy(),
            aoi_info={
                "category": "Much Adhesive",
                "lighting_mode": "MID",
            },
            aoi_epicenters=[(0, 0, 68, 123)],
        )

        self.assertFalse(result["adhesive_is_defect"])
        self.assertEqual(result["mid_bright_witness_coverage"], 0.0)
        self.assertEqual(result["excess_coverage"], 0.0)

    def test_mid_neutral_brightness_change_is_not_called_resin(self):
        reference = self._make_mid_scene()
        test = reference.copy()

        # Diferença luminosa neutra sem ganho amarelo/vermelho.
        test[38:88, 24:30] = (205, 205, 205)

        result = ShiftExpert().analyze(
            reference,
            test,
            aoi_info={
                "category": "Much Adhesive",
                "lighting_mode": "MID",
            },
            aoi_epicenters=[(0, 0, 68, 123)],
        )

        self.assertEqual(result["mid_bright_witness_coverage"], 0.0)
        self.assertFalse(result["adhesive_is_defect"])

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
