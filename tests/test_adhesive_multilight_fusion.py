import unittest

from src.core.adhesive_multilight_fusion import (
    fuse_adhesive_multilight,
)


def _analysis(
    mode,
    *,
    adhesive_score,
    physical_score,
    adhesive_is_defect,
    local_verdict="FALHA FALSA",
    memory_label="OK",
    memory_similarity=0.9,
):
    return {
        "lighting_mode": mode,
        "verdict": local_verdict,
        "is_defect": local_verdict == "DEFEITO REAL",
        "confidence": 0.99,
        "active_engines": [
            "shift_expert.py",
            "silk_expert.py",
            "semantic_expert.py",
            "ssim_expert.py",
            "knn_expert.py",
        ],
        "detail": {
            "adhesive_score": adhesive_score,
            "adhesive_is_defect": adhesive_is_defect,
            "adhesive_tolerance": 0.32,
            "physical_score": physical_score,
            "best_match_label": memory_label,
            "best_similarity": memory_similarity,
            "adhesive_reason": f"adesivo {mode}",
            "decision_trace": {
                "operator_review_required": False,
            },
        },
    }


class AdhesiveMultiLightFusionTests(unittest.TestCase):
    def test_real_case_side_ok_top_ng_mid_ok_becomes_single_defect(self):
        analyses = {
            "SIDE": _analysis(
                "SIDE",
                adhesive_score=0.7921925340496478,
                physical_score=0.85,
                adhesive_is_defect=True,
                local_verdict="FALHA FALSA",
                memory_label="OK",
                memory_similarity=0.963599935989827,
            ),
            "TOP": _analysis(
                "TOP",
                adhesive_score=0.9722685571309425,
                physical_score=0.9722685571309425,
                adhesive_is_defect=True,
                local_verdict="DEFEITO REAL",
                memory_label="NG",
                memory_similarity=1.0,
            ),
            "MID": _analysis(
                "MID",
                adhesive_score=0.0,
                physical_score=0.19002199972850198,
                adhesive_is_defect=False,
                local_verdict="FALHA FALSA",
                memory_label="OK",
                memory_similarity=0.855,
            ),
        }

        result = fuse_adhesive_multilight(analyses)

        self.assertIsInstance(result, dict)
        self.assertTrue(result["is_defect"])
        self.assertEqual(result["verdict"], "DEFEITO REAL")
        self.assertTrue(result["multilight_final"])
        self.assertTrue(result["eligible_for_final_decision"])
        self.assertEqual(result["lighting_mode"], "MULTILIGHT")

        detail = result["detail"]
        self.assertEqual(
            detail["fusion_rule"],
            "adhesive_multilight_strong_auxiliary",
        )
        self.assertEqual(
            detail["adhesive_multilight_dominant_mode"],
            "TOP",
        )
        self.assertIn(
            "TOP",
            detail["adhesive_multilight_strong_auxiliary_modes"],
        )
        self.assertIn(
            "SIDE",
            detail["adhesive_multilight_positive_modes"],
        )
        self.assertEqual(
            detail["adhesive_multilight_memory_role"],
            "audit_only",
        )
        self.assertAlmostEqual(detail["final_score"], 0.9722685571309425)

    def test_two_moderate_positive_lights_are_corroborated_defect(self):
        analyses = {
            "SIDE": _analysis(
                "SIDE",
                adhesive_score=0.55,
                physical_score=0.60,
                adhesive_is_defect=True,
            ),
            "TOP": _analysis(
                "TOP",
                adhesive_score=0.60,
                physical_score=0.62,
                adhesive_is_defect=True,
            ),
            "MID": _analysis(
                "MID",
                adhesive_score=0.05,
                physical_score=0.20,
                adhesive_is_defect=False,
            ),
        }

        result = fuse_adhesive_multilight(analyses)

        self.assertEqual(result["verdict"], "DEFEITO REAL")
        self.assertEqual(
            result["detail"]["fusion_rule"],
            "adhesive_multilight_corroborated",
        )

    def test_single_non_strong_positive_requires_review(self):
        analyses = {
            "SIDE": _analysis(
                "SIDE",
                adhesive_score=0.10,
                physical_score=0.20,
                adhesive_is_defect=False,
            ),
            "TOP": _analysis(
                "TOP",
                adhesive_score=0.60,
                physical_score=0.70,
                adhesive_is_defect=True,
            ),
            "MID": _analysis(
                "MID",
                adhesive_score=0.0,
                physical_score=0.15,
                adhesive_is_defect=False,
            ),
        }

        result = fuse_adhesive_multilight(analyses)

        self.assertFalse(result["is_defect"])
        self.assertEqual(result["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertTrue(result["production_review_required"])
        self.assertTrue(
            result["detail"]["decision_trace"]["operator_review_required"]
        )

    def test_no_positive_physical_evidence_is_false_failure(self):
        analyses = {
            mode: _analysis(
                mode,
                adhesive_score=0.0,
                physical_score=0.15,
                adhesive_is_defect=False,
            )
            for mode in ("SIDE", "TOP", "MID")
        }

        result = fuse_adhesive_multilight(analyses)

        self.assertFalse(result["is_defect"])
        self.assertEqual(result["verdict"], "FALHA FALSA")
        self.assertEqual(
            result["detail"]["fusion_rule"],
            "adhesive_multilight_no_physical_evidence",
        )

    def test_missing_lighting_never_produces_final_decision(self):
        analyses = {
            "SIDE": _analysis(
                "SIDE",
                adhesive_score=0.9,
                physical_score=0.9,
                adhesive_is_defect=True,
            ),
            "TOP": _analysis(
                "TOP",
                adhesive_score=0.9,
                physical_score=0.9,
                adhesive_is_defect=True,
            ),
        }

        self.assertIsNone(fuse_adhesive_multilight(analyses))


if __name__ == "__main__":
    unittest.main()
