"""Regressão: inspeção FALTANDO 3/3 OK sem confirmação humana XP."""
from __future__ import annotations

import os
import unittest
from copy import deepcopy

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.core.multilight_fusion import fuse_multilight
from src.core.neural.aoi_v3_epoch5_contract import (
    EPOCH5_VOTE_SCHEMA, SNAPSHOT_SHA256,
)
from src.core.production_multilight_all_clear import (
    RULE, verified_supervised_multilight_all_clear,
)
from src.ui.production_confidence_gate import production_decision_policy
from src.ui.production_autonomy_controller import (
    ProductionAutonomyController, VERDICT_HOLD_MS,
)
from tests.test_production_autonomy_controller import _Panel


SCORES = {"SIDE": 3.726585617869205e-8,
          "TOP": 6.200603364092672e-10,
          "MID": 0.0034973658621311188}


def real_all_clear_shape(*, category="FALTANDO"):
    """Os scores são do evento XP; dados KNN são estrutura mínima."""
    analyses = {}
    for index, (light, score) in enumerate(SCORES.items(), 1):
        same_source = format(index, "064x")
        neighbor = {
            "label": "OK", "path": "public/ok_archive/example.png",
            "source_pixel_sha256": same_source, "similarity": .79,
        }
        analyses[light] = {
            "lighting_mode": light,
            "verdict": "FALHA FALSA",
            "is_defect": False,
            "confidence": 1.0 - score,
            "production_review_required": False,
            "detail": {
                "cnn_v3_primary": True,
                "cnn_v3_status": "INFERENCE_OK",
                "cnn_v3_checkpoint_verified": True,
                "cnn_v3_checkpoint_sha256": SNAPSHOT_SHA256,
                "cnn_v3_ng_score_uncalibrated": score,
                "cnn_v2_aoi_category": category,
                "recognition_route": "NEW_CNN",
                "fusion_rule": "cnn_v3_primary_direct",
                "dominant_engine": "aoi_quality_cnn_v3",
                "final_score": score, "physical_score": 0.0,
                "operator_review_required": False,
                "decision_trace": {"operator_review_required": False},
                "aoi_v3_trial": {
                    "schema": EPOCH5_VOTE_SCHEMA, "status": "READY",
                    "epoch": 5, "light": light,
                    "checkpoint_sha256": SNAPSHOT_SHA256,
                    "ng_score_uncalibrated": score, "predicted": "OK",
                    "ng_veto": False,
                    "automatic_ok_from_v3_authorized": False,
                },
                "knn_c": {
                    "schema": "visionx.knn_c_live_review_gated.v1",
                    "status": "REVISAO", "bank_status": "READY",
                    "diagnostic_only": True,
                    "auto_decision_authorized": False,
                    "vote_A": "OK", "vote_B": "OK",
                    "neighbors_A": [dict(neighbor)],
                    "neighbors_B": [dict(neighbor)],
                },
            },
        }
    result = fuse_multilight(analyses, category)
    assert result is not None
    result["detail"]["production_continuous_auto"] = True
    result["detail"]["production_binary_auto"] = True
    return result


class SupervisedAllClearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_complete_three_light_ok_is_auto_zero_not_operator_review(self):
        for category in ("FALTANDO", "INVERTIDO", "EMBORCADO", "DESLOCADO"):
            with self.subTest(category=category):
                analysis = real_all_clear_shape(category=category)
                self.assertEqual(
                    analysis["detail"]["fusion_rule"], "multilight_all_clear",
                )
                self.assertTrue(verified_supervised_multilight_all_clear(analysis))
                decision = production_decision_policy(analysis)
                self.assertTrue(decision["auto_allowed"])
                self.assertFalse(decision["operator_review_required"])
                self.assertEqual(decision["reason"], RULE)
                self.assertEqual(decision["verdict"], "FALHA FALSA")
                self.assertEqual(decision["proposed_decision"], "OK")
                self.assertTrue(decision["unverified_physical_label"])

    def test_incomplete_corrupted_or_contradictory_proof_cannot_release_zero(self):
        base = real_all_clear_shape()
        mutations = {
            "incomplete": lambda a: a.update(multilight_final=False),
            "capture": lambda a: a["detail"].update(capture_error=True),
            "veto_ng": lambda a: a["detail"]["aoi_v3_multilight"]["lights"][
                "TOP"].update(predicted="NG"),
            "epoch": lambda a: a["detail"]["aoi_v3_multilight"].update(epoch=3),
            "sha": lambda a: a["detail"]["cnn_v3_light_diagnostics"][
                "TOP"].update(checkpoint_sha256="0"*64),
            "physical": lambda a: a["detail"]["multilight_evidence"][
                "MID"].update(physical_score=.5),
            "score_mismatch": lambda a: a["detail"]["multilight_evidence"][
                "SIDE"].update(final_score=.5),
            "missing": lambda a: a["detail"].update(raw_hard_missing_evidence=True),
            "memory_conflict": lambda a: a["detail"].update(memory_conflict=True),
            "weak_knn": lambda a: a["detail"]["knn_c_light_neighbor_evidence"][
                "MID"]["neighbors_B"][0].update(similarity=.60),
            "knn_ng": lambda a: a["detail"]["knn_c_light_observer_votes"][
                "SIDE"].update(vote_A="NG"),
            "different_sha": lambda a: a["detail"][
                "knn_c_light_neighbor_evidence"]["TOP"]["neighbors_B"][0].update(
                    source_pixel_sha256="f"*64),
            "fake_early": lambda a: a["detail"].update(
                fusion_rule="cnn_v3_primary_direct"),
            "pending_review": lambda a: a.update(production_review_required=True),
            "raw_trace": lambda a: a["detail"]["decision_trace"].update(
                raw_hard_missing_evidence=True),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                analysis = deepcopy(base)
                mutation(analysis)
                self.assertFalse(
                    verified_supervised_multilight_all_clear(analysis),
                )
                policy = production_decision_policy(analysis)
                self.assertFalse(
                    policy["proposed_decision"] == "OK"
                    and policy["auto_allowed"],
                )

    def test_controller_waits_one_second_and_sends_only_one_zero(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        analysis = real_all_clear_shape()
        panel.current_analysis = analysis
        self.assertTrue(controller.analysis_ready(analysis))
        self.assertEqual(controller._stage_remaining_ms, VERDICT_HOLD_MS)
        self.assertEqual(panel.saved, [])
        self.assertEqual(panel.visible_verdicts[-1][0], "FALHA FALSA")
        generation = controller.generation
        controller._stage_timer.stop()
        controller._finish_presentation(generation)
        self.assertEqual(controller.state, "auto_decision_pending")
        self.assertFalse(panel.production_review_pending)
        self.assertEqual(panel.saved, [])
        controller._stage_timer.stop()
        controller._emit_auto_decision(generation, "OK")
        self.assertEqual(panel.saved, [("OK", "production_auto")])
        self.assertEqual(panel.auto_records, ["OK"])
        controller._emit_auto_decision(generation, "OK")
        self.assertEqual(len(panel.saved), 1)
        panel.close()

    def test_pause_does_not_send_xp(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        analysis = real_all_clear_shape()
        panel.current_analysis = analysis
        self.assertTrue(controller.analysis_ready(analysis))
        gen = controller.generation
        controller.toggle_pause()
        controller._finish_presentation(gen)
        controller._emit_auto_decision(gen, "OK")
        self.assertEqual(panel.saved, [])
        self.assertTrue(controller.paused)
        panel.close()


if __name__ == "__main__":
    unittest.main()
