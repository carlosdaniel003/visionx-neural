"""Regressão do NG corroborado: classificação binária e revisão são independentes."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from src.core.corroborated_ng_verdict import corroborated_ng_verdict
from src.core.neural.faltando_checkpoint_contract import TRIAL_97P70_SHA256
from src.services.capture_debug_payload import decision_record
from src.ui.decision_verdict_feedback import verdict_feedback_state
from src.ui.production_confidence_gate import (
    _record_policy, production_decision_policy,
)
from src.ui.network_xp_debug import format_network_debug_report


LIGHTS = ("SIDE", "TOP", "MID")


def ng_corroborated():
    return {
        "multilight_final": True,
        "is_defect": True,
        "verdict": "DEFEITO REAL",
        "production_review_required": True,
        "confidence": 0.5,
        "reason": "Vizinhos NG e ausência física corroborados",
        "detail": {
            "multilight_category": "FALTANDO",
            "recognition_route": "NEW_CNN",
            "cnn_v2_consensus": "OK",
            "fusion_rule": "knn_visual_cnn_physical_corroboration_v1",
            "operator_review_required": True,
            "cnn_v2_experimental": True,
            "cnn_v2_active": True,
            "cnn_v2_supervised_auto_eligible": False,
            "knn_visual_corroboration": {
                "status": "NG_CORROBORATED",
                "strong_ng_lights": list(LIGHTS),
                "auto_decision_authorized": False,
                "lights": {
                    light: {
                        "status": "NG_SIMILAR", "ng_support": 2,
                        "ok": {"count": 125, "best_similarity": 0.81},
                        "ng": {"count": 4, "best_similarity": 0.96},
                        "cross_category_ng": False,
                    } for light in LIGHTS
                },
            },
            "physical_missing_veto": {
                "status": "SUSPECT_ABSENCE",
                "suspect_lights": ["TOP"],
                "unavailable_lights": [],
                "geometry_review_lights": [],
            },
            "decision_trace": {
                "operator_review_required": True,
                "fusion_rule": "knn_visual_cnn_physical_corroboration_v1",
                "memory": {},
            },
        },
    }


class BinaryNGWithReviewTests(unittest.TestCase):
    def test_real_event_binary_ng_and_operator_requirement_are_both_preserved(self):
        analysis = ng_corroborated()
        self.assertTrue(corroborated_ng_verdict(analysis))
        self.assertEqual(verdict_feedback_state(analysis), ("DEFEITO REAL", "ng"))
        policy = production_decision_policy(analysis)
        self.assertEqual(policy["verdict"], "DEFEITO REAL")
        self.assertEqual(policy["proposed_decision"], "NG")
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(policy["reason"], "corroborated_ng_requires_human_confirmation")
        self.assertTrue(analysis["production_review_required"])

    def test_production_record_preserves_pending_even_with_ng_verdict(self):
        analysis = ng_corroborated()
        panel = SimpleNamespace(current_analysis=analysis, production_review_policy=None)
        policy = production_decision_policy(analysis)
        _record_policy(panel, policy, resolution="pending")
        self.assertTrue(analysis["production_review_required"])
        self.assertTrue(analysis["detail"]["production_decision_policy"]["operator_review_required"])
        self.assertEqual(panel.production_review_policy["verdict"], "DEFEITO REAL")
        _record_policy(panel, policy, resolution="confirmed")
        self.assertFalse(analysis["production_review_required"])

    def test_raw_ng_keeps_visual_verdict_without_bypassing_operational_review(self):
        for field, value in (
            ("physical_missing_veto", {"status": "NO_STRONG_EVIDENCE"}),
            ("knn_visual_corroboration", {"status": "CONFLICT"}),
        ):
            case = ng_corroborated()
            case["detail"][field] = value
            with self.subTest(field=field):
                self.assertFalse(corroborated_ng_verdict(case))
                self.assertEqual(verdict_feedback_state(case),
                                 ("DEFEITO REAL", "ng"))
                self.assertFalse(production_decision_policy(case)["auto_allowed"])
        older = {
            "verdict": "DEFEITO REAL", "is_defect": True,
            "production_review_required": True,
        }
        self.assertFalse(corroborated_ng_verdict(older))
        self.assertEqual(verdict_feedback_state(older),
                         ("DEFEITO REAL", "ng"))
        self.assertFalse(production_decision_policy(older)["auto_allowed"])

    def test_trial_cnn_three_ng_lights_display_ng_but_do_not_auto_send_xp(self):
        # Reprodução dos estados relevantes do evento XP de 21/10.
        analysis = {
            "multilight_final": True,
            "is_defect": True,
            "verdict": "DEFEITO REAL",
            "production_review_required": True,
            "detail": {
                "multilight_category": "FALTANDO",
                "recognition_route": "NEW_CNN",
                "fusion_rule": "multilight_strong_single",
                "cnn_v2_checkpoint_sha256": TRIAL_97P70_SHA256,
                "cnn_v2_trial_manual_only": True,
                "cnn_v2_consensus": "REVIEW",
                "cnn_v2_consensus_reason": "INVALID_LIGHT_EVIDENCE",
                "operator_review_required": True,
                "decision_trace": {"operator_review_required": True},
            },
        }
        self.assertEqual(verdict_feedback_state(analysis),
                         ("DEFEITO REAL", "ng"))
        policy = production_decision_policy(analysis)
        self.assertEqual(policy["verdict"], "DEFEITO REAL")
        self.assertEqual(policy["proposed_decision"], "NG")
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(policy["reason"], "cnn_97p70_unqualified_operator_review_only")

    def test_uncertain_physical_only_is_still_review_not_ng(self):
        candidate = ng_corroborated()
        candidate["verdict"] = "REVISÃO OBRIGATÓRIA"
        candidate["is_defect"] = False
        self.assertFalse(corroborated_ng_verdict(candidate))
        self.assertFalse(production_decision_policy(candidate)["auto_allowed"])
        self.assertEqual(verdict_feedback_state(candidate),
                         ("REVISÃO OBRIGATÓRIA", "ng"))

    def test_debug_shows_actual_ok_and_ng_neighbor_evidence_per_light(self):
        analysis = ng_corroborated()
        record = decision_record(analysis, {"category": "FALTANDO"})
        self.assertEqual(record["verdict"], "DEFEITO REAL")
        self.assertTrue(record["operator_review_required"])
        self.assertTrue(record["corroborated_ng_classification"])
        self.assertEqual(record["knn_visual"]["status"], "NG_CORROBORATED")
        self.assertEqual(record["knn_visual"]["lights"]["SIDE"]["ok_count"], 125)
        self.assertEqual(record["knn_visual"]["lights"]["SIDE"]["ng_count"], 4)
        self.assertEqual(record["physical_missing_veto"]["suspect_lights"], ["TOP"])
        lines = format_network_debug_report({
            "schema": "visionx.network_xp_debug.v1",
            "event_id": "0e87f2ab7d0947248c2fa674df723fc5",
            "source": "windows_xp",
            "validation": {"valid": True},
            "decision": record,
        })
        self.assertIn("KNN VISUAL (VIZINHOS SIMILARES)", lines)
        self.assertIn("vizinhos OK=125", lines)
        self.assertIn("NG=4", lines)
        self.assertIn("Veredito: DEFEITO REAL", lines)


if __name__ == "__main__":
    unittest.main()
