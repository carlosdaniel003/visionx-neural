"""Teste de integração dos três modos e consenso CNN FALTANDO v2.

Demonstra somente invariantes de software, não segurança de defeitos inéditos.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.core.multilight_fusion import fuse_multilight
from src.core.neural.faltando_category_scope import uses_faltando_v2
from src.core.neural.faltando_multilight_consensus import summarize_cnn_multilight
from src.ui.production_confidence_gate import production_decision_policy
from src.ui.production_autonomy_controller import ProductionAutonomyController, AUTO_DECISION_DELAY_MS
from tests.test_production_autonomy_controller import _Panel

LIGHTS = ("SIDE", "TOP", "MID")
CHECKPOINT = "a" * 64
CATEGORIES = ("FALTANDO", "EMBORCADO", "INVERTIDO", "DESLOCADO")


def frame(category, light, vote="OK", *, model_hash=CHECKPOINT):
    score = .01 if vote == "OK" else .99 if vote == "NG" else .48
    return {
        "is_defect": vote == "NG",
        "verdict": (
            "FALHA FALSA" if vote == "OK" else
            "DEFEITO REAL" if vote == "NG" else "REVISÃO OBRIGATÓRIA"
        ),
        "confidence": .99 if vote != "REVIEW" else .52,
        "lighting_mode": light,
        "production_review_required": vote == "REVIEW",
        "active_engines": ["faltando_cnn_v2.py"],
        "detail": {
            "recognition_route": "NEW_CNN",
            "cnn_v2_aoi_category": category,
            "cnn_v2_active": True,
            "cnn_v2_experimental": True,
            "cnn_v2_checkpoint_verified": True,
            "cnn_v2_checkpoint_sha256": model_hash,
            "cnn_v2_status": "INFERENCE_OK",
            "cnn_v2_lighting_mode": light,
            "cnn_v2_ng_score_uncalibrated": score,
            "final_score": score,
            "fusion_rule": "cnn_v2_dualscale_direct",
            "operator_review_required": vote == "REVIEW",
            "decision_trace": {
                "fusion_rule": "cnn_v2_dualscale_direct",
                "operator_review_required": vote == "REVIEW",
                "final_score": score,
                "physical_score": 0.0,
            },
        }
    }


def final(category="FALTANDO", *, votes=None):
    votes = votes or ("OK", "OK", "OK")
    return fuse_multilight(
        {light: frame(category, light, v) for light, v in zip(LIGHTS, votes)},
        category,
    )


class SharedCategoryTests(unittest.TestCase):
    def test_explicit_scope_excludes_all_adhesives_and_unknown(self):
        for category in CATEGORIES:
            self.assertTrue(uses_faltando_v2(category))
        for category in ("MUITO ADESIVO", "Much Adhesive", "EXCESS ADHESIVE", "OTHER", ""):
            self.assertFalse(uses_faltando_v2(category))

    def test_all_four_categories_produce_three_light_ok(self):
        for category in CATEGORIES:
            with self.subTest(category=category):
                fused = final(category)
                policy = production_decision_policy(fused)
                self.assertEqual(fused["detail"]["cnn_v2_consensus"], "OK")
                self.assertTrue(fused["detail"]["cnn_v2_supervised_auto_eligible"])
                self.assertEqual(policy["proposed_decision"], "OK")
                self.assertTrue(policy["auto_allowed"])

    def test_three_ng_votes_permit_supervised_ng(self):
        for category in CATEGORIES:
            with self.subTest(category=category):
                fused = final(category, votes=("NG", "NG", "NG"))
                self.assertEqual(fused["detail"]["cnn_v2_consensus"], "NG")
                policy = production_decision_policy(fused)
                self.assertEqual(policy["proposed_decision"], "NG")
                self.assertTrue(policy["auto_allowed"])
                self.assertFalse(policy["operator_review_required"])

    def test_single_positive_cannot_send_zero_or_one(self):
        fused = final("INVERTIDO", votes=("NG", "OK", "OK"))
        self.assertFalse(fused["detail"]["cnn_v2_supervised_auto_eligible"])
        self.assertFalse(production_decision_policy(fused)["auto_allowed"])
        self.assertEqual(fused["detail"]["cnn_v2_consensus_reason"],
                         "DISAGREEMENT_BETWEEN_LIGHTS")

    def test_low_confidence_or_unavailable_light_review(self):
        for votes in (("OK", "REVIEW", "OK"), ("NG", "NG", "REVIEW")):
            with self.subTest(votes=votes):
                fused = final("EMBORCADO", votes=votes)
                self.assertFalse(production_decision_policy(fused)["auto_allowed"])

    def test_model_changed_during_cycle_blocks_both_votes(self):
        sources = {light: frame("FALTANDO", light) for light in LIGHTS}
        sources["MID"]["detail"]["cnn_v2_checkpoint_sha256"] = "b"*64
        fused = fuse_multilight(sources, "FALTANDO")
        self.assertFalse(production_decision_policy(fused)["auto_allowed"])
        self.assertEqual(fused["detail"]["cnn_v2_consensus_reason"],
                         "CHECKPOINT_CHANGED_DURING_CYCLE")

    def test_missing_checkpoint_confirmation_never_auto_ok(self):
        sources = {light: frame("DESLOCADO", light) for light in LIGHTS}
        sources["MID"]["detail"]["cnn_v2_checkpoint_verified"] = False
        fused = fuse_multilight(sources, "DESLOCADO")
        self.assertFalse(production_decision_policy(fused)["auto_allowed"])

    def test_unknown_event_and_single_light_never_auto(self):
        report = frame("FALTANDO", "SIDE")
        self.assertFalse(production_decision_policy(report)["auto_allowed"])
        report["detail"]["cnn_v2_consensus"] = "OK"
        report["detail"]["cnn_v2_supervised_auto_eligible"] = True
        self.assertFalse(production_decision_policy(report)["auto_allowed"])

    def test_memory_mixed_with_cnn_forces_review(self):
        sources = {light: frame("FALTANDO", light) for light in LIGHTS}
        sources["SIDE"]["detail"]["recognition_route"] = "KNOWN_KNN"
        fused = fuse_multilight(sources, "FALTANDO")
        self.assertFalse(production_decision_policy(fused)["auto_allowed"])

    def test_malformed_one_light_cannot_make_fake_consensus(self):
        sources = {light: frame("FALTANDO", light) for light in LIGHTS}
        sources["TOP"]["detail"]["cnn_v2_ng_score_uncalibrated"] = float("nan")
        summary = summarize_cnn_multilight(sources, "FALTANDO")
        self.assertFalse(summary["cnn_v2_supervised_auto_eligible"])
        self.assertEqual(summary["cnn_v2_consensus_reason"], "INVALID_SCORE")

    def test_physical_adhesive_cannot_borrow_cnn_authority(self):
        sources = {light: frame("MUITO ADESIVO", light) for light in LIGHTS}
        summary = summarize_cnn_multilight(sources, "MUITO ADESIVO")
        self.assertFalse(summary["cnn_v2_supervised_auto_eligible"])

    def test_training_replay_flag_does_not_become_auto_ok(self):
        # Real orchestration bypasses the CNN wrapper during physical replay;
        # even a synthetic CNN-shaped single-frame result cannot auto-release.
        result = frame("FALTANDO", "SIDE")
        result["detail"]["recognition_route"] = "NEW_EXPERTS"
        self.assertFalse(production_decision_policy(result)["auto_allowed"])


class SupervisedAutonomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setup_cycle(self, decision="NG"):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        analysis = final("INVERTIDO", votes=(decision, decision, decision))
        panel.current_analysis = analysis
        controller.pending_analysis = analysis
        controller.generation += 1
        return panel, controller

    def test_auto_ng_transmits_one_and_updates_counter(self):
        panel, controller = self.setup_cycle("NG")
        controller._emit_auto_ng(controller.generation)
        self.assertEqual(panel.saved, [("NG", "production_auto")])
        self.assertEqual(panel.auto_records, ["NG"])
        self.assertEqual(controller.state, "idle")

    def test_auto_ok_transmits_zero(self):
        panel, controller = self.setup_cycle("OK")
        controller._emit_auto_ok(controller.generation)
        self.assertEqual(panel.saved, [("OK", "production_auto")])
        self.assertEqual(panel.auto_records, ["OK"])

    def test_pause_forbids_send_until_resume(self):
        panel, controller = self.setup_cycle("NG")
        self.assertTrue(controller.toggle_pause())
        controller._emit_auto_ng(controller.generation)
        self.assertEqual(panel.saved, [])
        self.assertFalse(controller.toggle_pause())
        controller._emit_auto_ng(controller.generation)
        self.assertEqual(panel.saved, [("NG", "production_auto")])

    def test_switch_to_test_or_shadow_cancels_auto_action(self):
        for mode in ("Modo Teste", "Modo Sombra"):
            with self.subTest(mode=mode):
                panel, controller = self.setup_cycle("NG")
                generation = controller.generation
                panel.combo_mode.setCurrentText(mode)
                controller._emit_auto_ng(generation)
                self.assertEqual(panel.saved, [])

    def test_current_frame_replaced_before_dispatch_is_cancelled(self):
        panel, controller = self.setup_cycle("NG")
        panel.current_analysis = dict(panel.current_analysis)
        controller._emit_auto_ng(controller.generation)
        self.assertEqual(panel.saved, [])

    def test_review_never_auto_transmits(self):
        panel, controller = self.setup_cycle("NG")
        panel.current_analysis = final("INVERTIDO", votes=("NG", "OK", "OK"))
        controller.pending_analysis = panel.current_analysis
        controller._emit_auto_ng(controller.generation)
        self.assertEqual(panel.saved, [])
        self.assertTrue(panel.production_review_pending)

    def test_failed_ng_command_not_counted_as_success(self):
        panel, controller = self.setup_cycle("NG")

        def fail(decision, source="button"):
            panel.saved.append((decision, source))
            panel.last_decision_command_success = False
            return False
        panel.save_label = fail
        controller._emit_auto_ng(controller.generation)
        self.assertEqual(panel.auto_records, [])
        self.assertTrue(panel.production_review_pending)

    def test_operator_grace_period_is_two_seconds(self):
        self.assertGreaterEqual(AUTO_DECISION_DELAY_MS, 2000)


if __name__ == "__main__":
    unittest.main()
