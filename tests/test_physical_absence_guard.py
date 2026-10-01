import unittest
from unittest.mock import patch

import numpy as np

from src.core.experts.physical_absence_guard import PhysicalAbsenceGuard


class PhysicalAbsenceGuardEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.guard = PhysicalAbsenceGuard()

    @staticmethod
    def real_emborcado_vector():
        # Métricas reproduzidas a partir do evento
        # 1ef8605d905e4fac998cee708c89cbf6 usando o mesmo ROI do VisionX.
        return {
            "missing_active": True,
            "missing_is_defect": True,
            "missing_classification": "CONTEÚDO INESPERADO NA ROI",
            "missing_score": 0.8487753587129022,
            "missing_changed_coverage": 0.5243551587301587,
            "missing_residual_mean": 0.40792080760002136,
            "missing_appearance_loss": 0.46180088711708467,
            "missing_direct_similarity": 0.5381991128829153,
            "missing_edge_mismatch": 0.5096905814348861,
            "missing_best_similarity": 0.09315446019172668,
        }

    @staticmethod
    def physical_detail():
        return {
            "silk_error_pct": 0.54,
            "semantic_loss": 0.71,
            "physical_score": 1.0,
        }

    def test_real_emborcado_missing_vector_is_confirmed(self):
        hard, reason, support = self.guard.hard_absence_evidence(
            self.real_emborcado_vector(),
            self.physical_detail(),
        )

        self.assertTrue(hard)
        self.assertTrue(support["supported"])
        self.assertIn("desapareceu", reason)

    def test_real_invertido_extreme_collapse_is_confirmed(self):
        result = {
            "missing_active": True,
            "missing_is_defect": True,
            "missing_classification": "QUEBRA DA EXPECTATIVA VISUAL",
            "missing_score": 0.9447108065489517,
            "missing_changed_coverage": 0.7555259146341463,
            "missing_residual_mean": 0.5955082774162292,
            "missing_appearance_loss": 0.6620358168598258,
            "missing_direct_similarity": 0.3379641831401742,
            "missing_edge_mismatch": 0.33433916716958356,
            "missing_best_similarity": 0.14604395627975464,
        }
        detail = {
            "silk_error_pct": 0.4574468085106383,
            "semantic_loss": 0.5713105201721191,
            "physical_score": 0.8569657802581787,
        }

        hard, reason, support = self.guard.hard_absence_evidence(
            result,
            detail,
        )

        self.assertTrue(hard)
        self.assertFalse(support["primary_supported"])
        self.assertTrue(support["extreme_supported"])
        self.assertIn("colapso visual extremo", reason)

    def test_cross_category_guard_requests_strict_dual_scale_support(self):
        reference = np.full((160, 240, 3), 40, dtype=np.uint8)
        test = reference.copy()

        with patch(
            "src.core.experts.physical_absence_guard."
            "DualScalePresenceAnalyzer.analyze",
            return_value={
                "missing_dual_scale_policy": "dual_scale_presence_v1",
                "missing_dual_scale_active": True,
                "missing_dual_scale_triggered": True,
                "missing_context_hard_absence": False,
                "missing_context_hard_reason": (
                    "colapso visual extremo sem confirmação física independente"
                ),
            },
        ) as mocked:
            result = self.guard.analyze(
                reference,
                test,
                global_box_info={"w": 220, "h": 140},
                aoi_info={"category": "DESLOCADO"},
                aoi_epicenters=[(90, 55, 40, 35)],
                physical_detail={
                    "silk_error_pct": 0.4867,
                    "semantic_loss": 0.3014,
                    "physical_score": 0.8780,
                },
            )

        self.assertFalse(result["missing_hard_absence"])
        self.assertTrue(
            mocked.call_args.kwargs["require_physical_support_for_extreme"]
        )

    def test_dual_scale_context_can_promote_cross_category_guard(self):
        reference = np.full((160, 240, 3), 40, dtype=np.uint8)
        test = reference.copy()

        with patch(
            "src.core.experts.physical_absence_guard."
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
            result = self.guard.analyze(
                reference,
                test,
                global_box_info={"w": 220, "h": 140},
                aoi_info={"category": "EMBORCADO"},
                aoi_epicenters=[(90, 55, 40, 35)],
                physical_detail={
                    "silk_error_pct": 0.55,
                    "semantic_loss": 0.61,
                    "physical_score": 0.90,
                },
            )

        self.assertTrue(result["missing_hard_absence"])
        self.assertTrue(result["missing_context_hard_absence"])
        self.assertEqual(
            result["missing_classification"],
            "COMPONENTE FISICAMENTE AUSENTE — DUAL-SCALE",
        )

    def test_nearby_component_match_blocks_cross_category_override(self):
        result = self.real_emborcado_vector()
        result["missing_best_similarity"] = 0.52

        hard, _reason, _support = self.guard.hard_absence_evidence(
            result,
            self.physical_detail(),
        )

        self.assertFalse(hard)

    def test_weak_independent_physical_support_blocks_override(self):
        detail = self.physical_detail()
        detail["semantic_loss"] = 0.41

        hard, reason, support = self.guard.hard_absence_evidence(
            self.real_emborcado_vector(),
            detail,
        )

        self.assertFalse(hard)
        self.assertFalse(support["supported"])
        self.assertIn("motores físicos independentes", reason)

    def test_probable_displacement_is_never_reclassified_as_missing(self):
        result = self.real_emborcado_vector()
        result["missing_classification"] = "DESLOCAMENTO PROVÁVEL"

        hard, reason, _support = self.guard.hard_absence_evidence(
            result,
            self.physical_detail(),
        )

        self.assertFalse(hard)
        self.assertIn("deslocado", reason)

    def test_guard_categories_do_not_include_adhesive_or_missing(self):
        self.assertIn("EMBORCADO", self.guard.CATEGORIES)
        self.assertIn("DESLOCADO", self.guard.CATEGORIES)
        self.assertIn("INVERTIDO", self.guard.CATEGORIES)
        self.assertNotIn("MUITO ADESIVO", self.guard.CATEGORIES)
        self.assertNotIn("FALTANDO", self.guard.CATEGORIES)


if __name__ == "__main__":
    unittest.main()
