"""Trava da KNN C precisa sobreviver à fusão de SIDE/TOP/MID."""
from __future__ import annotations

import unittest

from src.core.knn_c_multilight_gate import gate_knn_c_multilight
from src.core.multilight_fusion import fuse_multilight
from src.ui.production_confidence_gate import production_decision_policy


def analysis(mode, *, c=None, status="SUGERE_NG"):
    detail = {
        "final_score": 0.15,
        "physical_score": 0.15,
        "dominant_engine": "ssim",
        "fusion_rule": "local",
        "decision_trace": {},
    }
    if c:
        detail["knn_c"] = {
            "status": status, "source": "A_E_B",
            "requires_human_confirmation": True,
            "conflicts_with_existing_verdict": status == "SUGERE_NG",
        }
    return {
        "lighting_mode": mode,
        "verdict": "FALHA FALSA",
        "is_defect": False,
        "confidence": .98,
        "detail": detail,
    }


class KNNCMultilightGateTests(unittest.TestCase):
    def test_three_light_ok_with_single_c_ng_never_auto_ok(self):
        values = {x: analysis(x, c=x == "TOP") for x in ("SIDE", "TOP", "MID")}
        final = fuse_multilight(values, "FALTANDO")
        self.assertEqual(final["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertTrue(final["production_review_required"])
        self.assertTrue(final["detail"]["operator_review_required"])
        self.assertTrue(
            final["detail"]["decision_trace"]["knn_c_multilight_review_gate"]
        )
        self.assertEqual(
            final["detail"]["knn_c_light_diagnostics"]["TOP"]["status"],
            "SUGERE_NG",
        )
        policy = production_decision_policy(final)
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])

    def test_all_clear_without_c_is_unchanged(self):
        values = {x: analysis(x) for x in ("SIDE", "TOP", "MID")}
        final = fuse_multilight(values, "FALTANDO")
        self.assertEqual(final["verdict"], "FALHA FALSA")
        self.assertFalse(final["production_review_required"])
        self.assertNotIn("knn_c_multilight_review_gate", final["detail"])

    def test_c_ok_never_turns_an_existing_ng_into_ok(self):
        fused = {"verdict": "DEFEITO REAL", "is_defect": True,
                 "confidence": .98, "detail": {"decision_trace": {}}}
        values = {x: analysis(x, c=x == "SIDE", status="SUGERE_OK")
                  for x in ("SIDE", "TOP", "MID")}
        result = gate_knn_c_multilight(fused, values)
        self.assertEqual(result["verdict"], "DEFEITO REAL")
        self.assertTrue(result["is_defect"])
        self.assertTrue(result["production_review_required"])

    def test_unselected_or_unconfirmed_candidate_never_forces_review(self):
        values = {x: analysis(x) for x in ("SIDE", "TOP", "MID")}
        values["MID"]["detail"]["knn_c"] = {
            "status": "SUGERE_NG", "requires_human_confirmation": False,
        }
        final = fuse_multilight(values, "FALTANDO")
        self.assertEqual(final["verdict"], "FALHA FALSA")
        self.assertNotIn("knn_c_multilight_review_gate", final["detail"])


if __name__ == "__main__":
    unittest.main()
