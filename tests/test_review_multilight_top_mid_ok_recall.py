"""Regressão do log XP INVERTIDO 11:43, evento a1e18960ebea42e4a180592137fea410.

Reconstrói os scores, fontes humanas, vizinhos KNN C A/B e índice visual
registrados pelo operador. Não substitui avaliação com as imagens/peça reais.
"""
from __future__ import annotations

from copy import deepcopy
from unittest import TestCase

from src.core.review_multilight_ok_consensus import (
    TOP_MID_MODE, apply_review_multilight_ok_consensus,
    verified_review_multilight_ok,
)
from src.ui.production_confidence_gate import production_decision_policy
from src.ui.decision_verdict_feedback import verdict_feedback_state
from tests.test_review_multilight_ok_consensus import (
    LIGHTS, CATEGORY, MODEL, real_event_inputs, fuse_real_event,
)

EVENT = "a1e18960ebea42e4a180592137fea410"
SCORES = {
    "SIDE": .14601385593414307,
    "TOP": .8789026141166687,
    "MID": .10622026026248932,
}
HUMAN_OK = {"SIDE": .96360328, "TOP": .98118429, "MID": .96861766}
HUMAN_NG = {"SIDE": .95218287, "TOP": .94545744, "MID": .93912127}
VISUAL_OK = {"SIDE": .971562, "TOP": .981184, "MID": .968618}
VISUAL_NG = {"SIDE": .952183, "TOP": .945457, "MID": .939121}
OK_IDS = {
    "SIDE": "a1a29e62dba0547c5a1f9932da6b740b881afd3d970ae13dee3af820fb58a50f",
    "TOP": "183d383391aa3279b60446efa246fff5c189b80ece4a3f25b193b1c255818b21",
    "MID": "decfa9784df0d53db3e64cd752109990e04864b516d1d38120a3a6e50856ffff",
}
NG_IDS = {
    "SIDE": "fb458ecad2a76e0c0bf3de0483da1c768c996f3ae3a503f133347b9dc4ea066e",
    "TOP": "4caf1af7506d9b8f2cee906e83d9e7bd49877bb6d81bed05ab5b5d0612295e38",
    "MID": "b4afaf17820a8418718b10cd9ed6a722f73487249dde519b5fa915ac8849ab66",
}
TOP_OK = (
    "public/ok_archive/2026-11-04_0904_INVERTIDO_TOP.png",
    "cba456f5a490b82a19d8a94e0ed4397462a935e06ed0afff7bd905468a1cd005",
)
MID_OK = (
    "public/ok_archive/2026-11-04_0904_INVERTIDO_MID.png",
    "b868e0ab8fd949d61d90ec5c5a57f5336e08ccfb6e3c9379b134ce662b7a46d6",
)
SIDE_OK = (
    "public/ok_archive/2026-10-05_1549_INVERTIDO.png",
    "f2b44e858996debe2a0c22a673ab60c7eb7607b1143059c7f6eec03043f39665",
)
SIDE_NG = (
    "public/ng_archive/2026-10-21_1413_FALTANDO.png",
    "3317191211b28d99f1948f1009659ad6ff4332ab2ea6a6e1855b05ef3e7d24f6",
)
TOP_NG = (
    "public/ng_archive/2026-10-21_1414_INVERTIDO_TOP.png",
    "d2e6b9f28f726a8fa423491ea035f19bd3e4aec50ec738c023bf4238b0c05b10",
)
MID_NG = (
    "public/ng_archive/2026-10-21_1414_INVERTIDO_MID.png",
    "845537c1d517e3d55620901416b707816789143ae4a85bceecaa4080659fabac",
)


def _n(label, similarity, source):
    return {
        "label": label,
        "similarity": similarity,
        "path": source[0],
        "source_pixel_sha256": source[1],
    }


def event_1143():
    analyses, report = real_event_inputs()
    for light in LIGHTS:
        a = analyses[light]
        score = SCORES[light]
        a["verdict"] = "REVISÃO OBRIGATÓRIA"
        a["confidence"] = 1 - score if light != "TOP" else score
        d = a["detail"]
        d["cnn_v2_ng_score_uncalibrated"] = score
        d["final_score"] = score
        d["decision_trace"]["final_score"] = score
        d["decision_trace"]["cnn_ng_score_uncalibrated"] = score
        h = d["confirmed_visual_recall"]
        h["status"] = "AMBIGUOUS" if light == "TOP" else "NO_MATCH"
        h["best_ok_similarity"] = HUMAN_OK[light]
        h["best_ng_similarity"] = HUMAN_NG[light]
        h["best_ok_source"]["record_id"] = OK_IDS[light]
        h["best_ng_source"]["record_id"] = NG_IDS[light]
        knn = d["knn_c"]
        knn["status"] = "REVISAO"
        knn["reason"] = (
            "DISCORDANCIA_KNN_A_B" if light == "SIDE"
            else "SEM_SUGESTAO_CONFIAVEL"
        )
        knn["vote_A"] = "NG" if light != "MID" else "OK"
        knn["vote_B"] = "NG" if light == "TOP" else "OK"
        knn["suggestion_A"] = knn["suggestion_B"] = "REVISAO"
        knn["similarity_A"] = {
            "SIDE": .80083917, "TOP": .84033818, "MID": .91249781,
        }[light]
        knn["similarity_B"] = {
            "SIDE": .84548128, "TOP": .88436082, "MID": .89390919,
        }[light]
        if light == "SIDE":
            knn["neighbors_A"] = [
                _n("NG", .80083917, SIDE_NG),
                _n("NG", .79829052, SIDE_NG),
                _n("OK", .79592849, SIDE_OK),
                _n("NG", .79547121, SIDE_NG),
                _n("OK", .78535632, SIDE_OK),
            ]
            knn["neighbors_B"] = [
                _n("OK", .84548128, SIDE_OK),
                _n("OK", .84531991, SIDE_OK),
                _n("OK", .84531991, SIDE_OK),
                _n("OK", .83358445, SIDE_OK),
                _n("OK", .82350120, SIDE_OK),
            ]
        elif light == "TOP":
            knn["neighbors_A"] = [
                _n("OK", .84033818, TOP_OK),
                _n("NG", .70691185, TOP_NG),
                _n("NG", .70433411, TOP_NG),
                _n("NG", .68900623, TOP_NG),
                _n("NG", .68781912, TOP_NG),
            ]
            knn["neighbors_B"] = [
                _n("OK", .88436082, TOP_OK),
                _n("NG", .77986343, TOP_NG),
                _n("NG", .77979103, TOP_NG),
                _n("NG", .77815013, TOP_NG),
                _n("OK", .77170999, TOP_OK),
            ]
        else:
            knn["neighbors_A"] = [
                _n("OK", .91249781, MID_OK),
                _n("NG", .81333494, MID_NG),
                _n("OK", .80428984, MID_OK),
                _n("OK", .79534075, MID_OK),
                _n("NG", .79176786, MID_NG),
            ]
            knn["neighbors_B"] = [
                _n("OK", .89390919, MID_OK),
                _n("OK", .79225873, MID_OK),
                _n("OK", .77725941, MID_OK),
                _n("NG", .74192769, MID_NG),
                _n("NG", .74161380, MID_NG),
            ]
        visual = report["lights"][light]
        visual["ok"]["best_similarity"] = VISUAL_OK[light]
        visual["ng"]["best_similarity"] = VISUAL_NG[light]
        visual["ng_margin_over_ok"] = round(
            VISUAL_NG[light] - VISUAL_OK[light], 6
        )
    return analyses, report


class ReviewTopMidArchiveOKTests(TestCase):
    def test_1143_is_reproduced_and_two_archive_anchors_classify_visual_ok(self):
        analyses, report = event_1143()
        fused = fuse_real_event(analyses, report)
        self.assertEqual(fused["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertEqual(fused["detail"]["fusion_rule"], "multilight_local_review")
        self.assertEqual(fused["detail"]["cnn_v2_consensus"], "REVIEW")
        self.assertEqual(fused["detail"]["cnn_v2_consensus_reason"], "INVALID_LIGHT_EVIDENCE")
        self.assertEqual(fused["detail"]["multilight_strong_positive_modes"], [])
        self.assertEqual(fused["detail"]["knn_visual_corroboration"]["status"], "CONFLICT")
        corrected = apply_review_multilight_ok_consensus(fused, analyses)
        self.assertEqual(corrected["verdict"], "FALHA FALSA")
        self.assertFalse(corrected["is_defect"])
        self.assertTrue(corrected["production_review_required"])
        self.assertTrue(verified_review_multilight_ok(corrected))
        proof = corrected["detail"]["review_multilight_ok_proof"]
        self.assertEqual(proof["evidence_mode"], TOP_MID_MODE)
        self.assertEqual(proof["knn_c_archive"]["archive_anchors"]["TOP"]["path"], TOP_OK[0])
        self.assertEqual(proof["knn_c_archive"]["archive_anchors"]["MID"]["path"], MID_OK[0])
        self.assertEqual(proof["cnn_ng_scores_uncalibrated"]["TOP"], SCORES["TOP"])
        self.assertFalse(proof["automatic_xp_authorized"])
        self.assertFalse(proof["physical_orientation_verified"])
        self.assertEqual(verdict_feedback_state(corrected), ("FALHA FALSA", "ok"))

    def test_1143_production_does_not_automatically_press_zero_or_one(self):
        analyses, report = event_1143()
        out = apply_review_multilight_ok_consensus(
            fuse_real_event(analyses, report), analyses,
        )
        out["detail"].update(
            production_continuous_auto=True, production_binary_auto=True,
        )
        policy = production_decision_policy(out)
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "")
        self.assertEqual(
            policy["reason"], "cnn_review_multilight_ok_requires_confirmation_v1",
        )

    def test_no_ok_if_physical_ng_or_conflicting_human_or_archive(self):
        mutations = [
            lambda f, a: f["detail"].update(physical_score=.1),
            lambda f, a: f["detail"]["decision_trace"].update(
                raw_hard_missing_evidence=True),
            lambda f, a: f["detail"].update(
                physical_missing_veto={
                    "status": "SUSPECT_ABSENCE", "suspect_lights": ["TOP"],
                }),
            lambda f, a: f["detail"]["knn_visual_corroboration"].update(
                strong_ng_lights=["TOP"]),
            lambda f, a: f["detail"]["knn_visual_corroboration"].update(
                trend_ng_lights=["SIDE", "TOP", "MID"]),
            lambda f, a: f["detail"]["knn_visual_corroboration"][
                "lights"]["SIDE"]["ng"].update(best_similarity=.97),
            lambda f, a: a["SIDE"]["detail"]["confirmed_visual_recall"].update(
                best_ok_similarity=.954),
            lambda f, a: a["TOP"]["detail"]["confirmed_visual_recall"].update(
                best_ng_similarity=.97),
            lambda f, a: a["MID"]["detail"]["confirmed_visual_recall"][
                "best_ok_source"].update(label="NG"),
            lambda f, a: f["detail"]["cnn_v2_light_diagnostics"]["TOP"].update(
                ng_score_uncalibrated=.92),
            lambda f, a: f["detail"]["cnn_v2_light_diagnostics"]["SIDE"].update(
                ng_score_uncalibrated=.80),
            lambda f, a: f["detail"]["knn_c_light_observer_votes"]["SIDE"].update(
                similarity_A=.93),
            lambda f, a: f["detail"]["knn_c_light_observer_votes"]["TOP"].update(
                status="SUGERE_NG"),
            lambda f, a: f["detail"]["knn_c_light_neighbor_evidence"]["TOP"][
                "neighbors_B"][0].update(source_pixel_sha256="a" * 64),
            lambda f, a: f["detail"]["knn_c_light_neighbor_evidence"]["MID"][
                "neighbors_B"][0].update(path="public/ok_archive/other_INVERTIDO_MID.png"),
            lambda f, a: f["detail"]["knn_c_light_neighbor_evidence"]["TOP"][
                "neighbors_A"][1].update(similarity=.81),
            lambda f, a: f["detail"]["knn_c_light_neighbor_evidence"]["SIDE"][
                "neighbors_B"][0].update(label="NG"),
            lambda f, a: f["detail"]["knn_c_light_neighbor_evidence"].pop("TOP"),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                analyses, report = event_1143()
                fused = fuse_real_event(analyses, report)
                mutation(fused, analyses)
                final = apply_review_multilight_ok_consensus(fused, analyses)
                self.assertEqual(final["verdict"], "REVISÃO OBRIGATÓRIA")
                self.assertFalse(verified_review_multilight_ok(final))

    def test_forged_two_light_proof_fails_safe_for_production(self):
        analyses, report = event_1143()
        mutations = [
            lambda f: f["detail"]["review_multilight_ok_proof"].update(
                automatic_xp_authorized=True),
            lambda f: f["detail"]["review_multilight_ok_proof"][
                "knn_c_archive"]["archive_anchors"]["TOP"].update(
                pixel_sha256="f" * 64),
            lambda f: f["detail"]["review_multilight_ok_human_snapshot"][
                "MID"].update(best_ok_similarity=.50),
            lambda f: f["detail"]["knn_c_light_neighbor_evidence"]["MID"][
                "neighbors_A"][0].update(similarity=.70),
        ]
        for mutate in mutations:
            with self.subTest(mutate=str(mutate)):
                corrected = apply_review_multilight_ok_consensus(
                    fuse_real_event(analyses, report), analyses,
                )
                mutate(corrected)
                self.assertFalse(verified_review_multilight_ok(corrected))
                self.assertEqual(verdict_feedback_state(corrected),
                                 ("REVISÃO OBRIGATÓRIA", "ng"))
                corrected["detail"].update(
                    production_continuous_auto=True,
                    production_binary_auto=True,
                )
                policy = production_decision_policy(corrected)
                self.assertTrue(policy["operational_error"])
                self.assertFalse(policy["auto_allowed"])
                self.assertEqual(policy["proposed_decision"], "")


if __name__ == "__main__":
    import unittest
    unittest.main()
