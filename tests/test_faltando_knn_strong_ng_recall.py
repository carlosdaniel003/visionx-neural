"""Replay do evento 0652ff8f: CNN 3 OK, KNN NG extremo em 3 luzes.

A prova física inconclusiva por geometria não vira voto NG; a classificação
por MEMÓRIA requer confirmação humana e nunca autoriza comando XP.
"""
from __future__ import annotations

from copy import deepcopy
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from src.core.knn_visual_corroboration import (
    apply_knn_visual_corroboration, evaluate_knn_visual_corroboration,
    trusted_three_light_ng_recall,
)
from src.core.corroborated_ng_verdict import corroborated_ng_verdict
from src.ui.production_confidence_gate import production_decision_policy
from src.ui.decision_verdict_feedback import verdict_feedback_state
from src.ui.neural_telemetry_model import memory_seen_state

LIGHTS = ("SIDE", "TOP", "MID")
SHA = "6e4a31e8826d7b2afa18fbecb579a4d8979067713faa032d329f37f54729b599"
DATA = {
    "SIDE": (0.998912, 0.957091, 8, 129, 350, "NG_LEANING"),
    "TOP": (0.998856, 0.902414, 8, 95, 56, "NG_SIMILAR"),
    "MID": (0.999230, 0.897332, 16, 95, 56, "NG_SIMILAR"),
}


def knn_report():
    return {
        "schema": "visionx.knn_visual_corroboration.v1",
        "category": "FALTANDO",
        "status": "NG_CORROBORATED",
        "auto_decision_authorized": False,
        "trend_ng_lights": list(LIGHTS),
        "strong_ng_lights": ["TOP", "MID"],
        "ng_margin_mean": 0.080053,
        "lights": {
            light: {
                "status": status, "cross_category_ng": False,
                "ng_support": support,
                "ng_margin_over_ok": round(ng - ok, 6),
                "ok": {"count": ok_count, "best_similarity": ok},
                "ng": {"count": ng_count, "best_similarity": ng},
            }
            for light, (ng, ok, support, ng_count, ok_count, status) in DATA.items()
        },
    }


def event():
    return {
        "multilight_final": True,
        "verdict": "REVISÃO OBRIGATÓRIA", "is_defect": False,
        "production_review_required": True,
        "eligible_for_final_decision": False, "confidence": 0.5,
        "detail": {
            "multilight_category": "FALTANDO",
            "recognition_route": "NEW_CNN", "recognition_match": "NOT_FOUND",
            "cnn_v2_consensus": "OK", "cnn_v2_active": True,
            "cnn_v2_experimental": True,
            "cnn_v2_supervised_auto_eligible": False,
            "cnn_v2_light_diagnostics": {
                light: {
                    "category": "FALTANDO", "route": "NEW_CNN",
                    "cnn_active": True, "checkpoint_verified": True,
                    "cnn_status": "INFERENCE_OK", "checkpoint_sha256": SHA,
                    "verdict": "FALHA FALSA", "ng_score_uncalibrated": score,
                }
                for light, score in {
                    "SIDE": 0.0020141236018389463,
                    "TOP": 0.0011869892477989197,
                    "MID": 0.000015825486116227694,
                }.items()
            },
            "physical_missing_veto": {
                "status": "GEOMETRY_REVIEW",
                "suspect_lights": [], "unavailable_lights": [],
                "geometry_review_lights": list(LIGHTS),
                "lights": {
                    light: {"status": "GEOMETRY_REVIEW"} for light in LIGHTS
                },
            },
            "knn_visual_corroboration": knn_report(),
            "decision_trace": {"operator_review_required": True},
        },
    }


class StrongThreeLightNGRecallTests(unittest.TestCase):
    def test_real_numbers_produce_manual_ng_despite_three_cnn_ok(self):
        fused = event()
        self.assertTrue(trusted_three_light_ng_recall(fused["detail"]))
        with patch(
            "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
            return_value=knn_report(),
        ):
            result = apply_knn_visual_corroboration(fused, {}, "FALTANDO", None)
        self.assertEqual(result["verdict"], "DEFEITO REAL")
        self.assertTrue(result["is_defect"])
        self.assertEqual(
            result["detail"]["fusion_rule"],
            "knn_three_light_strong_ng_recall_vs_cnn_ok_v1",
        )
        self.assertFalse(result["eligible_for_final_decision"])
        self.assertTrue(result["production_review_required"])
        self.assertTrue(corroborated_ng_verdict(result))
        self.assertEqual(verdict_feedback_state(result), ("DEFEITO REAL", "ng"))
        policy = production_decision_policy(result)
        self.assertEqual(policy["verdict"], "DEFEITO REAL")
        self.assertEqual(policy["proposed_decision"], "NG")
        self.assertTrue(policy["operator_review_required"])
        self.assertFalse(policy["auto_allowed"])
        self.assertEqual(memory_seen_state(result)["status"], "VI_SEMELHANTE")

    def test_full_knn_aggregation_keeps_trend_and_strong_lights(self):
        memory = SimpleNamespace(
            _memory_lock=RLock(), signatures_ok=[], signatures_ng=[],
        )
        dataset = knn_report()["lights"]
        with patch(
            "src.core.knn_visual_corroboration._one_light",
            side_effect=lambda sample, light, *args: dataset[light],
        ):
            report = evaluate_knn_visual_corroboration({}, "FALTANDO", memory)
        self.assertEqual(report["status"], "NG_CORROBORATED")
        self.assertEqual(report["strong_ng_lights"], ["TOP", "MID"])
        self.assertEqual(report["trend_ng_lights"], list(LIGHTS))
        self.assertAlmostEqual(report["ng_margin_mean"], 0.080054, places=5)

    def test_close_ok_prevents_ng(self):
        item = event()
        item["detail"]["knn_visual_corroboration"]["lights"]["SIDE"]["ok"]["best_similarity"] = 0.995
        self.assertFalse(trusted_three_light_ng_recall(item["detail"]))

    def test_no_physical_measurement_or_unavailable_stays_review(self):
        for status in ("UNAVAILABLE", "NO_STRONG_EVIDENCE", "SUSPECT_ABSENCE", ""):
            item = event()
            item["detail"]["physical_missing_veto"]["status"] = status
            with self.subTest(status=status):
                self.assertFalse(trusted_three_light_ng_recall(item["detail"]))
        item = event()
        del item["detail"]["physical_missing_veto"]["lights"]["MID"]
        self.assertFalse(trusted_three_light_ng_recall(item["detail"]))

    def test_does_not_promote_two_lights_or_weak_ng_neighbors(self):
        for light in LIGHTS:
            item = event()
            item["detail"]["knn_visual_corroboration"]["lights"][light]["ng_support"] = 2
            with self.subTest(light=light):
                self.assertFalse(trusted_three_light_ng_recall(item["detail"]))
        item = event()
        item["detail"]["knn_visual_corroboration"]["lights"]["SIDE"]["ng"]["best_similarity"] = 0.97
        self.assertFalse(trusted_three_light_ng_recall(item["detail"]))

    def test_cnn_invalid_checkpoint_or_non_ok_votes_do_not_trigger(self):
        changes = (
            ("SIDE", "checkpoint_verified", False),
            ("TOP", "checkpoint_sha256", "b" * 64),
            ("MID", "cnn_status", "MODEL_UNAVAILABLE"),
            ("SIDE", "verdict", "REVISÃO OBRIGATÓRIA"),
            ("TOP", "ng_score_uncalibrated", 0.75),
        )
        for light, key, value in changes:
            item = event()
            item["detail"]["cnn_v2_light_diagnostics"][light][key] = value
            with self.subTest(light=light, key=key):
                self.assertFalse(trusted_three_light_ng_recall(item["detail"]))

    def test_cross_category_ng_not_promoted(self):
        item = event()
        item["detail"]["knn_visual_corroboration"]["lights"]["MID"]["cross_category_ng"] = True
        self.assertFalse(trusted_three_light_ng_recall(item["detail"]))

    def test_synthetic_ng_label_without_reliable_evidence_is_not_accepted(self):
        item = event()
        item["verdict"] = "DEFEITO REAL"
        item["is_defect"] = True
        item["detail"]["fusion_rule"] = "knn_three_light_strong_ng_recall_vs_cnn_ok_v1"
        item["detail"]["knn_visual_corroboration"]["lights"]["SIDE"]["ok"]["best_similarity"] = 0.999
        self.assertFalse(corroborated_ng_verdict(item))
        policy = production_decision_policy(item)
        self.assertFalse(policy["auto_allowed"])
        self.assertNotEqual(policy["proposed_decision"], "NG")

    def test_production_mode_never_sends_automatic_ng_for_recall(self):
        item = event()
        item["detail"]["fusion_rule"] = "knn_three_light_strong_ng_recall_vs_cnn_ok_v1"
        item["verdict"] = "DEFEITO REAL"
        item["is_defect"] = True
        item["production_review_required"] = False
        item["eligible_for_final_decision"] = True
        item["detail"]["cnn_v2_supervised_auto_eligible"] = True
        self.assertTrue(corroborated_ng_verdict(item))
        policy = production_decision_policy(item)
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])


if __name__ == "__main__":
    unittest.main()
