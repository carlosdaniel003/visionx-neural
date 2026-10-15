"""Replay dos scores reais: uma luz CNN REVIEW + 3 direções KNN NG.

O teste não simula que são novas imagens de holdout. Reproduz exatamente os
números de similaridade do debug AOI 63c48e2b81c14106841e4b1220da0afc
e verifica contrato final e trava de produção.
"""
from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch
import unittest

from src.core.knn_visual_corroboration import (
    apply_knn_visual_corroboration,
    trusted_three_light_ng_trend, verified_partial_cnn_review,
)
from src.core.corroborated_ng_verdict import corroborated_ng_verdict
from src.core.physical_missing_veto import apply_physical_missing_veto
from src.ui.decision_verdict_feedback import verdict_feedback_state
from src.ui.production_confidence_gate import production_decision_policy

SHA = "6e4a31e8826d7b2afa18fbecb579a4d8979067713faa032d329f37f54729b599"
LIGHTS = ("SIDE", "TOP", "MID")
DATA = {
    "SIDE": (.951926, .908387, 29, 128, 350),
    "TOP": (.947828, .903074, 17, 94, 56),
    "MID": (.915403, .850404, 7, 94, 56),
}


def case():
    return {
        "verdict": "REVISÃO OBRIGATÓRIA", "is_defect": False,
        "multilight_final": True, "production_review_required": True,
        "eligible_for_final_decision": True, "confidence": .5,
        "detail": {
            "multilight_category": "FALTANDO",
            "multilight_final": True, "eligible_for_final_decision": True,
            "recognition_route": "NEW_CNN",
            "cnn_v2_consensus": "REVIEW",
            "cnn_v2_active": True, "cnn_v2_experimental": True,
            "cnn_v2_supervised_auto_eligible": False,
            "cnn_v2_light_diagnostics": {
                light: {
                    "route": "NEW_CNN", "cnn_active": True,
                    "cnn_status": "INFERENCE_OK", "checkpoint_verified": True,
                    "checkpoint_sha256": SHA, "category": "FALTANDO",
                    "ng_score_uncalibrated": score,
                    "verdict": "REVISÃO OBRIGATÓRIA" if light == "TOP"
                               else "FALHA FALSA",
                }
                for light, score in {
                    "SIDE": .0000436467053,
                    "TOP": .6390351653,
                    "MID": .0000331601696,
                }.items()
            },
            "decision_trace": {"operator_review_required": True},
            "fusion_rule": "multilight_local_review",
        },
    }


def report():
    return {
        "status": "NG_TREND_THREE_LIGHTS",
        "trend_ng_lights": list(LIGHTS),
        "strong_ng_lights": [],
        "ng_margin_mean": .0511,
        "auto_decision_authorized": False,
        "lights": {
            light: {
                "status": "NG_LEANING",
                "cross_category_ng": False,
                "ng_support": n,
                "ng_margin_over_ok": round(ng-ok, 6),
                "ok": {"count": oks, "best_similarity": ok},
                "ng": {"count": ngs, "best_similarity": ng},
            }
            for light, (ng, ok, n, ngs, oks) in DATA.items()
        },
    }


class ThreeLightNGTrendTests(unittest.TestCase):
    def test_real_log_numbers_support_three_light_trend(self):
        entry = case()
        entry["detail"]["knn_visual_corroboration"] = report()
        self.assertEqual(verified_partial_cnn_review(entry["detail"])["review_light"], "TOP")
        self.assertTrue(trusted_three_light_ng_trend(entry["detail"]))
        with patch(
            "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
            return_value=report()
        ):
            result = apply_knn_visual_corroboration(entry, {}, "FALTANDO", None)
        self.assertEqual(result["verdict"], "DEFEITO REAL")
        self.assertTrue(result["is_defect"])
        self.assertEqual(result["detail"]["fusion_rule"],
                         "knn_visual_three_light_ng_with_cnn_review_v1")
        self.assertTrue(result["production_review_required"])
        self.assertFalse(result["eligible_for_final_decision"])
        self.assertTrue(corroborated_ng_verdict(result))
        self.assertEqual(verdict_feedback_state(result), ("DEFEITO REAL", "ng"))
        policy = production_decision_policy(result)
        self.assertEqual(policy["verdict"], "DEFEITO REAL")
        self.assertEqual(policy["proposed_decision"], "NG")
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])

    def test_one_ok_neighbor_too_close_blocks_ng(self):
        data = report()
        data["lights"]["SIDE"]["ok"]["best_similarity"] = .940
        entry = case()
        entry["detail"]["knn_visual_corroboration"] = data
        self.assertFalse(trusted_three_light_ng_trend(entry["detail"]))

    def test_two_lights_only_cannot_label_ng(self):
        data = report()
        data["lights"]["MID"]["status"] = "CONFLICT"
        data["lights"]["MID"]["ng_support"] = 1
        entry = case()
        entry["detail"]["knn_visual_corroboration"] = data
        self.assertFalse(trusted_three_light_ng_trend(entry["detail"]))

    def test_unverified_cnn_or_mismatched_checkpoint_fails_closed(self):
        base = case()
        for light, key, value in (
            ("SIDE", "checkpoint_verified", False),
            ("TOP", "checkpoint_sha256", "a"*64),
            ("MID", "cnn_status", "FAILED"),
            ("TOP", "ng_score_uncalibrated", .51),
            ("SIDE", "category", "DESLOCADO"),
        ):
            entry = deepcopy(base)
            entry["detail"]["cnn_v2_light_diagnostics"][light][key] = value
            entry["detail"]["knn_visual_corroboration"] = report()
            with self.subTest(light=light, key=key):
                self.assertIsNone(verified_partial_cnn_review(entry["detail"]))
                self.assertFalse(trusted_three_light_ng_trend(entry["detail"]))

    def test_cnn_three_ok_is_not_promoted_by_relaxed_trend(self):
        entry = case()
        entry["detail"]["cnn_v2_consensus"] = "OK"
        entry["detail"]["knn_visual_corroboration"] = report()
        self.assertFalse(trusted_three_light_ng_trend(entry["detail"]))

    def test_invalid_shape_and_no_evidence_stay_review(self):
        entry = case()
        bad = report()
        bad["lights"]["MID"]["ng"]["best_similarity"] = None
        entry["detail"]["knn_visual_corroboration"] = bad
        self.assertFalse(trusted_three_light_ng_trend(entry["detail"]))
        self.assertEqual(entry["verdict"], "REVISÃO OBRIGATÓRIA")

    def test_physical_probe_runs_as_diagnostic_on_mixed_cnn_review(self):
        entry = case()
        image = __import__("numpy").zeros((80, 96, 3), dtype="uint8")
        samples = {light: {
            "lighting_mode": light, "sample_image": image.copy(),
            "test_image": image.copy()
        } for light in LIGHTS}
        observed = []
        def probe(ref, test):
            observed.append((ref.shape, test.shape))
            return {
                "status": "SUSPECT_ABSENCE" if len(observed) == 2
                          else "NO_STRONG_EVIDENCE",
                "scales": [],
            }
        result = apply_physical_missing_veto(
            entry, samples, "FALTANDO", probe=probe
        )
        self.assertEqual(len(observed), 3)
        self.assertEqual(result["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertEqual(result["detail"]["physical_missing_veto"]["status"],
                         "SUSPECT_ABSENCE")
        self.assertEqual(result["detail"]["physical_missing_veto"]["suspect_lights"],
                         ["TOP"])

    def test_physical_probe_not_started_on_invalid_mixed_model(self):
        entry = case()
        entry["detail"]["cnn_v2_light_diagnostics"]["TOP"]["checkpoint_verified"] = False
        with patch(
            "src.core.physical_missing_veto.physical_absence_probe",
            side_effect=AssertionError("Unexpected physical probe")
        ):
            result = apply_physical_missing_veto(entry, {}, "FALTANDO")
        self.assertNotIn("physical_missing_veto", result["detail"])


if __name__ == "__main__":
    unittest.main()
