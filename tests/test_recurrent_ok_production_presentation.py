"""Regression: operator-confirmed recurring OK must not be painted as review.

The expert verdict, production authorization, visible verdict card and manual
intervention card are distinct. No automatic XP 0/1 is authorized.
"""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication, QWidget

from src.core.knn_visual_corroboration import apply_knn_visual_corroboration
from src.ui.decision_verdict_feedback import verdict_feedback_state
from src.ui.production_autonomy_controller import ProductionAutonomyController
from src.ui.production_confidence_gate import production_decision_policy
from src.ui.production_session_feedback import ProductionInterventionFeedbackOverlay
from tests.test_knn_three_light_ok_counterevidence import fused, report
from tests.test_production_autonomy_controller import _Panel


def _confirmed_recurrent_ok():
    f = fused()
    f["detail"]["knn_c_light_neighbor_evidence"] = {
        "SIDE": {
            "neighbors_A": [
                {"path": "public/ok_archive/2026-10-21_1412_FALTANDO.png",
                 "source_pixel_sha256": "c9dec1d85617aa074e71b8b3640cace88f3a23c123185f7be84d6c06e2946bd9",
                 "label": "OK", "similarity": 0.99325113},
                {"path": "public/ng_archive/2026-10-21_1432_FALTANDO_SIDE.png",
                 "label": "NG", "similarity": 0.85977105},
            ],
            "neighbors_B": [
                {"path": "public/ok_archive/2026-10-21_1412_FALTANDO.png",
                 "source_pixel_sha256": "c9dec1d85617aa074e71b8b3640cace88f3a23c123185f7be84d6c06e2946bd9",
                 "label": "OK", "similarity": 0.97756581},
                {"path": "public/ng_archive/2026-10-21_1432_FALTANDO_SIDE.png",
                 "label": "NG", "similarity": 0.82187452},
            ],
        },
        "TOP": {"neighbors_A": [], "neighbors_B": []},
        "MID": {"neighbors_A": [], "neighbors_B": []},
    }
    r = report()
    r["lights"]["SIDE"] = {
        "status": "CONFLICT", "cross_category_ng": False,
        "ng_margin_over_ok": -0.045818,
        "ok": {"best_similarity": .995819},
        "ng": {"best_similarity": .950001},
    }
    with patch(
        "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
        return_value=r,
    ):
        result = apply_knn_visual_corroboration(f, {}, "FALTANDO", object())
    assert result["verdict"] == "FALHA FALSA"
    return result


class RecurrentOKPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_final_label_is_not_overwritten_by_required_manual_confirmation(self):
        result = _confirmed_recurrent_ok()
        self.assertTrue(result["production_review_required"])
        policy = production_decision_policy(result)
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertEqual(
            policy["reason"], "operator_confirmed_recurring_ok_requires_operator",
        )
        self.assertFalse(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "")
        self.assertEqual(
            verdict_feedback_state(result), ("FALHA FALSA", "ok"),
        )

    def test_unverified_false_failure_still_displays_mandatory_review(self):
        result = _confirmed_recurrent_ok()
        result["detail"]["recognized_recurrent_ok"]["source_pixel_sha256"] = "0" * 64
        self.assertEqual(
            verdict_feedback_state(result), ("REVISÃO OBRIGATÓRIA", "ng"),
        )
        self.assertEqual(
            production_decision_policy(result)["verdict"], "REVISÃO OBRIGATÓRIA",
        )

    def test_intervention_overlay_keeps_ok_label_and_manual_keys(self):
        panel = QWidget()
        panel.resize(900, 700)
        panel.show()
        overlay = ProductionInterventionFeedbackOverlay(panel)
        overlay.show_reason("FALHA FALSA")
        self.assertEqual(overlay.reason_label.text(), "FALHA FALSA")
        self.assertEqual(overlay.reason_label.property("tone"), "ok")
        self.assertEqual(
            overlay.header_label.text(), "CONFIRMAÇÃO HUMANA NECESSÁRIA",
        )
        self.assertIn("0 = OK", overlay.hint_label.text())
        self.assertIn("1 = NG", overlay.hint_label.text())
        overlay.show_reason("REVISÃO OBRIGATÓRIA")
        self.assertEqual(overlay.reason_label.text(), "REVISÃO OBRIGATÓRIA")
        self.assertEqual(overlay.reason_label.property("tone"), "ng")
        overlay.show_reason("DEFEITO REAL")
        self.assertEqual(overlay.reason_label.text(), "DEFEITO REAL")
        overlay.clear()
        panel.close()

    def test_controller_keeps_recurring_ok_class_without_emitting_xp(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        analysis = _confirmed_recurrent_ok()
        panel.current_analysis = analysis
        controller.pending_analysis = analysis
        controller.generation += 1
        controller._finish_presentation(controller.generation)
        self.assertEqual(controller.state, "operator_review")
        self.assertEqual(panel.interventions[-1], "FALHA FALSA")
        self.assertTrue(panel.production_review_pending)
        self.assertEqual(panel.production_review_policy["verdict"], "FALHA FALSA")
        self.assertEqual(panel.saved, [])
        self.assertEqual(panel.auto_records, [])


if __name__ == "__main__":
    unittest.main()
