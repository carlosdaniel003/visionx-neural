"""Regressão do evento INVERTIDO 5ba7734c810e44f8a3c2fa6b3f575293.

Os valores são os do log XP de 04/11/2026 às 10:30:21.
Não se presume equivalência física de captura nem acerto industrial.
"""
from __future__ import annotations

from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

from src.core.knn_visual_corroboration import apply_knn_visual_corroboration
from src.core.multilight_fusion import fuse_multilight
from src.core.operator_visual_memory_consensus import apply_operator_visual_memory_consensus
from src.core.physical_missing_veto import apply_physical_missing_veto
from src.core.review_multilight_ok_consensus import (
    RULE, apply_review_multilight_ok_consensus,
    review_multilight_ok_evidence, verified_review_multilight_ok,
)
from src.ui.production_confidence_gate import production_decision_policy

LIGHTS = ("SIDE", "TOP", "MID")
CATEGORY = "INVERTIDO"
MODEL = "c13ca3296eee5e7741c58f29a6ef035296c0fe84a5651a9481462d72042b120c"
EVENT = "5ba7734c810e44f8a3c2fa6b3f575293"
SCORES = {"SIDE": .13915790617465973, "TOP": .2759160101413727,
          "MID": .004452975001186132}
HUMAN_OK = {"SIDE": .95869114, "TOP": .97217664, "MID": .95243550}
HUMAN_NG = {"SIDE": .93314572, "TOP": .94045643, "MID": .92008216}
VISUAL_OK = {"SIDE": .980273, "TOP": .972177, "MID": .952435}
VISUAL_NG = {"SIDE": .933146, "TOP": .940456, "MID": .920082}
OK_RECORDS = {
    "SIDE": "dffce43f5a55daf39d54184aa7707ccff9fdb60f014c903e773efe37891cde03",
    "TOP": "4c56256eedb4a70996479ce106fe30b7b55673f6d9690113f24c9fd7e476e531",
    "MID": "aa391dfae969f75347a1e071cf2ef9b07ec247f13fc90bc5d23a91d707f70933",
}
NG_RECORDS = {
    "SIDE": "fb458ecad2a76e0c0bf3de0483da1c768c996f3ae3a503f133347b9dc4ea066e",
    "TOP": "4236f3604d08de08d2e57bc169695ab7b59e76954268ea5d2cdd43706f8d0d05",
    "MID": "b4afaf17820a8418718b10cd9ed6a722f73487249dde519b5fa915ac8849ab66",
}
SIDE_OK_PATH = "public/ok_archive/2026-10-05_1549_INVERTIDO.png"
SIDE_OK_SHA = "f2b44e858996debe2a0c22a673ab60c7eb7607b1143059c7f6eec03043f39665"


def _neighbor(label, similarity, path=SIDE_OK_PATH, sha=SIDE_OK_SHA):
    return {
        "label": label, "similarity": similarity,
        "path": path, "source_pixel_sha256": sha,
    }


def real_event_inputs():
    """Reproduz votos e margens do evento com os registros OK/NG exibidos."""
    analyses = {}
    for light in LIGHTS:
        score = SCORES[light]
        original_verdict = (
            "FALHA FALSA" if score <= .10 else "REVISÃO OBRIGATÓRIA"
        )
        knn = {
            "schema": "visionx.knn_c_live_review_gated.v1",
            "bank_status": "READY", "diagnostic_only": True,
            "auto_decision_authorized": False,
            "status": "SUGERE_OK" if light == "SIDE" else "REVISAO",
            "reason": (
                "B_SELETIVA_E_VOTOS_A_B_CONCORDAM" if light == "SIDE"
                else "SEM_SUGESTAO_CONFIAVEL"
            ),
            "vote_A": "OK" if light == "SIDE" else "NG",
            "vote_B": "OK" if light == "SIDE" else "NG",
            "suggestion_A": "REVISAO",
            "suggestion_B": "SUGERE_OK" if light == "SIDE" else "REVISAO",
            "similarity_A": {
                "SIDE": .87879183, "TOP": .65710305, "MID": .80031445
            }[light],
            "similarity_B": {
                "SIDE": .91059492, "TOP": .74667473, "MID": .77383393
            }[light],
            "neighbors_A": [],
            "neighbors_B": [],
        }
        if light == "SIDE":
            knn["source"] = "B_COM_VOTO_A"
            knn["requires_human_confirmation"] = True
            knn["neighbors_A"] = [
                _neighbor("OK", .87879183),
                _neighbor("NG", .81949380,
                          "public/ng_archive/2026-10-21_1416_FALTANDO_4.png",
                          "af8f20a270fdd6b71084619a77f0f89249f85202a8234e3cccfe0dbdf2c8611a"),
                _neighbor("OK", .81853099),
                _neighbor("OK", .81152618),
                _neighbor("OK", .80935113),
            ]
            knn["neighbors_B"] = [
                _neighbor("OK", .91059492),
                _neighbor("OK", .85155211),
                _neighbor("OK", .85155211),
                _neighbor("OK", .85039611),
                _neighbor("OK", .83808151),
            ]
        human_status = (
            "AMBIGUOUS" if light == "TOP" else "NO_MATCH"
        )
        analyses[light] = {
            "lighting_mode": light,
            "verdict": original_verdict,
            "is_defect": False,
            "confidence": 1-score,
            "production_review_required": True,
            "detail": {
                "recognition_route": "NEW_CNN",
                "cnn_v2_aoi_category": CATEGORY,
                "cnn_v2_active": True,
                "cnn_v2_experimental": True,
                "cnn_v2_trial_manual_only": True,
                "cnn_v2_checkpoint_verified": True,
                "cnn_v2_checkpoint_sha256": MODEL,
                "cnn_v2_checkpoint_best_epoch": 18,
                "cnn_v2_status": "INFERENCE_OK",
                "cnn_v2_lighting_mode": light,
                "cnn_v2_ng_score_uncalibrated": score,
                "final_score": score,
                "physical_score": 0.0,
                "fusion_rule": "cnn_v2_dualscale_direct",
                "operator_review_required": True,
                "decision_trace": {
                    "fusion_rule": "cnn_v2_dualscale_direct",
                    "operator_review_required": True,
                    "physical_score": 0.0,
                    "final_score": score,
                    "hard_missing_evidence": False,
                },
                "knn_c": knn,
                "confirmed_visual_recall": {
                    "schema": "visionx.human_visual_recall.v1",
                    "source": "PERSISTENT_HUMAN_LABELS",
                    "category": CATEGORY,
                    "status": human_status,
                    "reason": "VISUAL_SIMILARITY_NOT_EXACT_IDENTITY",
                    "diagnostic_only": True,
                    "auto_ok_authorized": False,
                    "best_ok_similarity": HUMAN_OK[light],
                    "best_ng_similarity": HUMAN_NG[light],
                    "best_ok_source": {
                        "label": "OK",
                        "source": "OPERATOR_CONFIRMED",
                        "record_id": OK_RECORDS[light],
                        "lighting_mode": light,
                    },
                    "best_ng_source": {
                        "label": "NG",
                        "source": "OPERATOR_CONFIRMED",
                        "record_id": NG_RECORDS[light],
                        "lighting_mode": light,
                    },
                },
            },
        }
    report = {
        "schema": "visionx.knn_visual_corroboration.v1",
        "status": "CONFLICT", "category": CATEGORY,
        "strong_ng_lights": [], "trend_ng_lights": [],
        "diagnostic_only": False,
        "auto_decision_authorized": False,
        "lights": {
            light: {
                "status": "CONFLICT", "cross_category_ng": False,
                "ng_support": {"SIDE": 8, "TOP": 7, "MID": 3}[light],
                "ok": {"count": {"SIDE": 155, "TOP": 4, "MID": 4}[light],
                       "best_similarity": VISUAL_OK[light]},
                "ng": {"count": {"SIDE": 39, "TOP": 17, "MID": 17}[light],
                       "best_similarity": VISUAL_NG[light]},
                "ng_margin_over_ok": round(
                    VISUAL_NG[light] - VISUAL_OK[light], 6
                ),
            } for light in LIGHTS
        },
    }
    return analyses, report


def fuse_real_event(analyses=None, report=None):
    if analyses is None or report is None:
        analyses, report = real_event_inputs()
    fused = fuse_multilight(analyses, CATEGORY)
    if not isinstance(fused, dict):
        raise AssertionError("Multilight fusion failed")
    fused = apply_physical_missing_veto(fused, {}, CATEGORY)
    with patch(
        "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
        return_value=deepcopy(report),
    ):
        fused = apply_knn_visual_corroboration(fused, {}, CATEGORY, None)
    fused = apply_operator_visual_memory_consensus(fused, analyses)
    return fused


class ReviewMultilightOKTests(TestCase):
    def test_real_1030_raw_fusion_is_review_and_cnn_disagrees_with_itself(self):
        analyses, report = real_event_inputs()
        fused = fuse_real_event(analyses, report)
        self.assertEqual(fused["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertEqual(fused["detail"]["fusion_rule"], "multilight_local_review")
        self.assertEqual(fused["detail"]["multilight_strong_positive_modes"], [])
        self.assertEqual(fused["detail"]["multilight_review_modes"], list(LIGHTS))
        self.assertEqual(fused["detail"]["cnn_v2_consensus"], "REVIEW")
        self.assertEqual(
            fused["detail"]["cnn_v2_consensus_reason"], "INVALID_LIGHT_EVIDENCE",
        )
        self.assertEqual(fused["detail"]["physical_score"], 0.0)
        self.assertEqual(
            fused["detail"]["knn_visual_corroboration"]["status"], "CONFLICT",
        )

    def test_real_1030_now_classifies_visual_false_alarm(self):
        analyses, report = real_event_inputs()
        fused = fuse_real_event(analyses, report)
        proof = review_multilight_ok_evidence(
            fused, {
                light: analyses[light]["detail"]["confirmed_visual_recall"]
                for light in LIGHTS
            },
        )
        self.assertIsNotNone(proof)
        self.assertEqual(proof["label"], "OK")
        self.assertFalse(proof["physical_orientation_verified"])
        self.assertFalse(proof["automatic_xp_authorized"])
        self.assertEqual(proof["side_ok_archive_path"], SIDE_OK_PATH)
        final = apply_review_multilight_ok_consensus(fused, analyses)
        self.assertEqual(final["verdict"], "FALHA FALSA")
        self.assertFalse(final["is_defect"])
        self.assertEqual(final["detail"]["fusion_rule"], RULE)
        self.assertTrue(final["production_review_required"])
        self.assertTrue(verified_review_multilight_ok(final))
        self.assertEqual(fused["verdict"], "REVISÃO OBRIGATÓRIA")

    def test_visual_false_alarm_does_not_send_any_automatic_xp_key(self):
        analyses, report = real_event_inputs()
        final = apply_review_multilight_ok_consensus(
            fuse_real_event(analyses, report), analyses,
        )
        final["detail"].update(
            production_binary_auto=True, production_continuous_auto=True,
        )
        policy = production_decision_policy(final)
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "")
        self.assertEqual(
            policy["reason"], "cnn_review_multilight_ok_requires_confirmation_v1",
        )

    def test_same_visual_classification_in_all_modes(self):
        analyses, report = real_event_inputs()
        fused = fuse_real_event(analyses, report)
        for mode in ("Modo Teste", "Modo Produção", "Modo Sombra"):
            with self.subTest(mode=mode):
                final = apply_review_multilight_ok_consensus(fused, analyses)
                self.assertEqual(final["verdict"], "FALHA FALSA")
                self.assertTrue(verified_review_multilight_ok(final))

    def test_physical_ng_and_close_ng_references_block_visual_ok(self):
        mutations = [
            lambda f, a: f["detail"].update(physical_score=.2),
            lambda f, a: f["detail"].update(hard_missing_evidence=True),
            lambda f, a: f["detail"]["decision_trace"].update(
                raw_hard_missing_evidence=True),
            lambda f, a: f["detail"].update(
                physical_missing_veto={
                    "status": "SUSPECT_ABSENCE", "suspect_lights": ["TOP"],
                }),
            lambda f, a: f["detail"]["knn_visual_corroboration"].update(
                strong_ng_lights=["TOP", "MID"]),
            lambda f, a: f["detail"]["knn_visual_corroboration"].update(
                trend_ng_lights=["SIDE", "TOP", "MID"]),
            lambda f, a: f["detail"]["knn_visual_corroboration"][
                "lights"]["TOP"]["ng"].update(best_similarity=.98),
            lambda f, a: a["MID"]["detail"]["confirmed_visual_recall"].update(
                best_ng_similarity=.945),
            lambda f, a: a["TOP"]["detail"]["confirmed_visual_recall"].update(
                best_ok_similarity=.94),
            lambda f, a: a["SIDE"]["detail"]["confirmed_visual_recall"][
                "best_ok_source"].update(label="NG"),
            lambda f, a: a["SIDE"]["detail"]["confirmed_visual_recall"][
                "best_ok_source"].update(record_id="not-a-hash"),
            lambda f, a: f["detail"]["cnn_v2_light_diagnostics"]["TOP"].update(
                ng_score_uncalibrated=.91),
            lambda f, a: f["detail"]["multilight_evidence"]["MID"].update(
                local_is_defect=True),
            lambda f, a: f["detail"]["knn_c_light_observer_votes"]["SIDE"].update(
                vote_A="NG"),
            lambda f, a: f["detail"]["knn_c_light_observer_votes"]["TOP"].update(
                similarity_B=.94),
            lambda f, a: f["detail"]["knn_c_light_neighbor_evidence"]["SIDE"][
                "neighbors_B"][0].update(source_pixel_sha256="a"*64),
            lambda f, a: f["detail"]["knn_c_light_neighbor_evidence"]["SIDE"][
                "neighbors_A"][1].update(similarity=.875),
        ]
        for n, mutate in enumerate(mutations):
            with self.subTest(case=n):
                analyses, report = real_event_inputs()
                fused = fuse_real_event(analyses, report)
                mutate(fused, analyses)
                final = apply_review_multilight_ok_consensus(fused, analyses)
                self.assertEqual(
                    final["verdict"], "REVISÃO OBRIGATÓRIA",
                )
                self.assertFalse(verified_review_multilight_ok(final))

    def test_corrupted_proof_blocks_visual_ok_and_xp_fallback(self):
        analyses, report = real_event_inputs()
        for mutation in (
            lambda f: f["detail"]["review_multilight_ok_proof"].update(
                automatic_xp_authorized=True),
            lambda f: f["detail"]["review_multilight_ok_proof"][
                "human_memory"]["TOP"].update(ok=.91),
            lambda f: f["detail"]["knn_visual_corroboration"][
                "lights"]["SIDE"]["ok"].update(best_similarity=.80),
            lambda f: f["detail"]["review_multilight_ok_human_snapshot"][
                "MID"].update(best_ok_similarity=.50),
        ):
            with self.subTest(mutation=str(mutation)):
                final = apply_review_multilight_ok_consensus(
                    fuse_real_event(analyses, report), analyses,
                )
                mutation(final)
                self.assertFalse(verified_review_multilight_ok(final))
                final["detail"].update(
                    production_binary_auto=True,
                    production_continuous_auto=True,
                )
                policy = production_decision_policy(final)
                self.assertTrue(policy["operational_error"])
                self.assertFalse(policy["auto_allowed"])
                self.assertEqual(policy["proposed_decision"], "")

    def test_cannot_claim_false_failure_from_only_cnn_low_score(self):
        analyses, report = real_event_inputs()
        fused = fuse_real_event(analyses, report)
        for light in LIGHTS:
            analyses[light]["detail"]["confirmed_visual_recall"]["status"] = "NO_MATCH"
            analyses[light]["detail"]["confirmed_visual_recall"][
                "best_ok_similarity"
            ] = .50
        final = apply_review_multilight_ok_consensus(fused, analyses)
        self.assertEqual(final["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertFalse(verified_review_multilight_ok(final))


if __name__ == "__main__":
    import unittest
    unittest.main()
