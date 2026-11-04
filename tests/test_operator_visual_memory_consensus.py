"""Regressões da prioridade humana visual sem confundir similaridade com par exato."""
from __future__ import annotations

from copy import deepcopy
import unittest

from src.core.operator_visual_memory_consensus import (
    RULE, apply_operator_visual_memory_consensus, memory_ok_counterevidence,
)
from src.ui.production_confidence_gate import production_decision_policy


LIGHTS = ("SIDE", "TOP", "MID")
CAT = "EMBORCADO"
SHA = "c13ca3296eee5e7741c58f29a6ef035296c0fe84a5651a9481462d72042b120c"


def _event():
    """Scores do evento AOI 2c2f436f... de 04/11/2026 (sem PNG industrial)."""
    human_ok = {"SIDE": .98464556, "TOP": .98073817, "MID": .96719095}
    human_ng = {"SIDE": .90181659, "TOP": .89982695, "MID": .88312874}
    archive_ok = {"SIDE": .984646, "TOP": .980738, "MID": .967191}
    archive_ng = {"SIDE": .901817, "TOP": .876938, "MID": .856974}
    cnn = {"SIDE": .999968886, "TOP": .949882507, "MID": .999832034}
    analyses = {}
    visual_lights = {}
    for light in LIGHTS:
        human = {
            "schema": "visionx.human_visual_recall.v1",
            "source": "PERSISTENT_HUMAN_LABELS",
            "diagnostic_only": True,
            "auto_ok_authorized": False,
            "category": CAT,
            "status": (
                "NO_MATCH" if light == "MID"
                else "HUMAN_OK_VISUAL_CANDIDATE"
            ),
            "reason": "VISUAL_SIMILARITY_NOT_EXACT_IDENTITY",
            "best_ok_similarity": human_ok[light],
            "best_ng_similarity": human_ng[light],
            "best_ok_source": {
                "source": "OPERATOR_CONFIRMED", "label": "OK",
                "lighting_mode": light, "record_id": light[0].lower() * 64,
            },
        }
        knn = {}
        if light == "SIDE":
            knn = {
                "schema": "visionx.knn_c_live_review_gated.v1",
                "bank_status": "READY", "status": "SUGERE_OK",
                "reason": "A_E_B_CONCORDAM", "diagnostic_only": True,
                "auto_decision_authorized": False,
                "vote_A": "OK", "vote_B": "OK",
                "suggestion_A": "SUGERE_OK", "suggestion_B": "SUGERE_OK",
                "neighbors_A": [
                    {"label": "OK", "similarity": x}
                    for x in (.96532639, .91104899, .89971381, .86382785, .85307751)
                ],
                "neighbors_B": [
                    {"label": "OK", "similarity": x}
                    for x in (.92206981, .87193952, .84984882, .82359927, .80697191)
                ],
            }
        analyses[light] = {
            "lighting_mode": light,
            "verdict": "DEFEITO REAL",
            "is_defect": True,
            "production_review_required": True,
            "detail": {
                "cnn_v2_lighting_mode": light,
                "cnn_v2_checkpoint_verified": True,
                "cnn_v2_checkpoint_sha256": SHA,
                "cnn_v2_status": "INFERENCE_OK",
                "cnn_v2_aoi_category": CAT,
                "cnn_v2_ng_score_uncalibrated": cnn[light],
                "recognition_route": "NEW_CNN",
                "confirmed_visual_recall": human,
                "knn_c": knn,
            },
        }
        visual_lights[light] = {
            "status": "NO_CONCLUSION", "cross_category_ng": False,
            "ng_support": 1 if light == "SIDE" else 0,
            "ok": {"best_similarity": archive_ok[light]},
            "ng": {"best_similarity": archive_ng[light]},
            "ng_margin_over_ok": round(archive_ng[light] - archive_ok[light], 6),
        }

    fused = {
        "verdict": "DEFEITO REAL",
        "is_defect": True,
        "lighting_mode": "MULTILIGHT",
        "multilight_final": True,
        "multilight_category": CAT,
        "production_review_required": False,
        "eligible_for_final_decision": True,
        "detail": {
            "multilight_final": True,
            "multilight_category": CAT,
            "eligible_for_final_decision": True,
            "production_continuous_auto": True,
            "production_binary_auto": True,
            "recognition_route": "NEW_CNN",
            "recognition_light_routes": {light: "NEW_CNN" for light in LIGHTS},
            "cnn_v2_consensus": "REVIEW",
            "cnn_v2_consensus_reason": "INVALID_LIGHT_EVIDENCE",
            "multilight_strong_positive_modes": list(LIGHTS),
            "hard_missing_evidence": False,
            "physical_score": 0.0,
            "decision_trace": {"hard_missing_evidence": False},
            "knn_visual_corroboration": {
                "schema": "visionx.knn_visual_corroboration.v1",
                "category": CAT, "status": "NO_CONCLUSION",
                "strong_ng_lights": [], "trend_ng_lights": [],
                "lights": visual_lights,
            },
        },
    }
    return analyses, fused


class OperatorVisualMemoryConsensusTests(unittest.TestCase):
    def test_real_emborcado_false_ng_recovers_multilight_ok_visual(self):
        analyses, fused = _event()
        proof = memory_ok_counterevidence(analyses, fused)
        self.assertEqual(proof["label"], "OK")
        self.assertFalse(proof["rgb_exact_pair"])
        self.assertFalse(proof["automatic_xp_authorized"])
        final = apply_operator_visual_memory_consensus(fused, analyses)
        self.assertEqual(final["verdict"], "FALHA FALSA")
        self.assertFalse(final["is_defect"])
        self.assertEqual(final["detail"]["fusion_rule"], RULE)
        self.assertEqual(final["detail"]["cnn_raw_verdict_before_memory"], "DEFEITO REAL")
        self.assertTrue(final["production_review_required"])
        self.assertEqual(fused["verdict"], "DEFEITO REAL")

    def test_same_visual_arbitration_in_all_three_modes(self):
        analyses, fused = _event()
        for mode in ("Modo Teste", "Modo Produção", "Modo Sombra"):
            with self.subTest(mode=mode):
                out = apply_operator_visual_memory_consensus(fused, analyses)
                self.assertEqual(out["verdict"], "FALHA FALSA")
                self.assertEqual(out["detail"]["fusion_rule"], RULE)

    def test_conflicting_or_incomplete_memory_must_not_override_ng(self):
        cases = [
            lambda a, f: a["TOP"]["detail"]["confirmed_visual_recall"].update(
                best_ok_similarity=.95),
            lambda a, f: a["MID"]["detail"]["confirmed_visual_recall"].update(
                best_ng_similarity=.945),
            lambda a, f: a["SIDE"]["detail"]["confirmed_visual_recall"].update(
                best_ok_source={"source": "ARCHIVE_PNG", "label": "OK",
                                "lighting_mode": "SIDE", "record_id": "s"*64}),
            lambda a, f: a["TOP"]["detail"]["confirmed_visual_recall"].update(
                status="HUMAN_NG_VISUAL_CANDIDATE"),
            lambda a, f: a["MID"]["detail"].update(cnn_v2_checkpoint_sha256="b"*64),
            lambda a, f: a["MID"]["detail"]["confirmed_visual_recall"]["best_ok_source"].update(
                lighting_mode="SIDE"),
            lambda a, f: f["detail"].update(hard_missing_evidence=True),
            lambda a, f: f["detail"]["decision_trace"].update(
                raw_hard_missing_evidence=True),
            lambda a, f: f["detail"]["knn_visual_corroboration"]["lights"]["TOP"].update(
                ng_margin_over_ok=.03),
            lambda a, f: f["detail"]["knn_visual_corroboration"].update(
                strong_ng_lights=["MID"]),
            lambda a, f: a["SIDE"]["detail"]["knn_c"].update(vote_A="NG"),
            lambda a, f: f.update(multilight_category="UNKNOWN"),
        ]
        for mutation in cases:
            with self.subTest(mutation=str(mutation)):
                analyses, fused = _event()
                mutation(analyses, fused)
                self.assertIsNone(memory_ok_counterevidence(analyses, fused))
                self.assertEqual(
                    apply_operator_visual_memory_consensus(fused, analyses)["verdict"],
                    "DEFEITO REAL",
                )

    def test_production_binary_fallback_cannot_turn_human_ok_into_ng(self):
        analyses, fused = _event()
        recovered = apply_operator_visual_memory_consensus(fused, analyses)
        policy = production_decision_policy(recovered)
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertEqual(policy["proposed_decision"], "")
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(
            policy["reason"], "operator_visual_memory_requires_confirmation_v1",
        )

    def test_invalid_counterevidence_marker_stops_without_sending_xp(self):
        analyses, fused = _event()
        recovered = apply_operator_visual_memory_consensus(fused, analyses)
        recovered["detail"]["operator_visual_memory_consensus"]["lights"].pop("MID")
        policy = production_decision_policy(recovered)
        self.assertFalse(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "")
        self.assertTrue(policy["operational_error"])

    def test_unsupervised_reject_remains_binary_ng(self):
        _, fused = _event()
        policy = production_decision_policy(fused)
        self.assertTrue(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "NG")


if __name__ == "__main__":
    unittest.main()
