"""Regression for event 8525be296415464a83fcd0764ce7e9d7.

Replay of the logged evidence, NOT a pixel-level inspection of the local
AOI station. Historical OK identity is supported by the recorded KNN
hashes. The current physical board is still NOT individually certified.
"""
from copy import deepcopy
import unittest
from unittest.mock import patch

from src.core.knn_visual_corroboration import (
    _recognized_recurrent_ok_20261029, apply_knn_visual_corroboration,
)
from src.ui.production_confidence_gate import production_decision_policy
from src.ui.decision_verdict_feedback import verdict_feedback_state
from tests.test_knn_three_light_ok_counterevidence import fused, report


SHA = {
    "TOP": "046effa074997ca1211b45e905745f2f6ec68263a0613854e4130eb7168e7d5d",
    "MID": "338f1cc612656297c21db284ea097489b90eb6e0537c3231dde45740f6a16e29",
}
SOURCES = {
    light: f"public/ok_archive/2026-10-27_1318_FALTANDO_{light}.png"
    for light in ("TOP", "MID")
}


def neighbor(label, similarity, light="SIDE", anchor=False):
    return {
        "label": label, "similarity": similarity,
        "path": SOURCES[light] if anchor else
                f"public/{label.lower()}_archive/{light}_example.png",
        "source_pixel_sha256": SHA[light] if anchor else "f" * 64,
    }


def real_event():
    a = fused()
    d = a["detail"]
    d["multilight_final"] = True
    d["eligible_for_final_decision"] = True
    d["recognition_light_routes"] = {x: "NEW_CNN" for x in ("SIDE","TOP","MID")}
    d["cnn_v2_light_diagnostics"] = {
        light: {
            "cnn_active": True, "checkpoint_verified": True,
            "cnn_status": "INFERENCE_OK", "route": "NEW_CNN",
            "verdict": "DEFEITO REAL", "category": "FALTANDO",
            "checkpoint_sha256": "c13ca3296eee5e7741c58f29a6ef035296c0fe84a5651a9481462d72042b120c",
            "ng_score_uncalibrated": score,
        }
        for light, score in (
            ("SIDE", .9987717270851135),
            ("TOP", .9987717270851135),
            ("MID", .999889612197876),
        )
    }
    d["knn_c_light_observer_votes"]["MID"].update({
        "status": "SUGERE_OK",
        "reason": "B_SELETIVA_E_VOTOS_A_B_CONCORDAM",
        "suggestion_A": "REVISAO", "suggestion_B": "SUGERE_OK",
    })
    d["knn_c_light_neighbor_evidence"] = {
        "SIDE": {
            "neighbors_A": [
                neighbor("NG", .89339485), neighbor("NG", .88418264),
                neighbor("OK", .88014235), neighbor("OK", .86061695),
                neighbor("OK", .85896562),
            ],
            "neighbors_B": [
                neighbor("NG", .83807385), neighbor("NG", .82729085),
                neighbor("OK", .82618206), neighbor("OK", .8120654),
                neighbor("OK", .80632833),
            ],
        },
        "TOP": {
            "neighbors_A": [
                neighbor("OK", .98621705, "TOP", True),
                neighbor("NG", .9127867, "TOP"),
                neighbor("OK", .88240339, "TOP"),
                neighbor("OK", .87639197, "TOP"),
                neighbor("OK", .87000136, "TOP"),
            ],
            "neighbors_B": [
                neighbor("OK", .97805809, "TOP", True),
                neighbor("NG", .88268065, "TOP"),
                neighbor("OK", .85127438, "TOP"),
                neighbor("OK", .84643866, "TOP"),
                neighbor("OK", .84335523, "TOP"),
            ],
        },
        "MID": {
            "neighbors_A": [
                neighbor("OK", .9825117, "MID", True),
                neighbor("OK", .86939965, "MID"),
                neighbor("OK", .86184946, "MID"),
                neighbor("OK", .8583928, "MID"),
                neighbor("NG", .84971955, "MID"),
            ],
            "neighbors_B": [
                neighbor("OK", .9710344, "MID", True),
                neighbor("OK", .81653948, "MID"),
                neighbor("OK", .81091247, "MID"),
                neighbor("OK", .81078255, "MID"),
                neighbor("OK", .80364104, "MID"),
            ],
        },
    }
    visual = report()
    visual["trend_ng_lights"] = []
    visual["strong_ng_lights"] = []
    for light, ok, ng, status in (
        ("SIDE", .991922, .947968, "CONFLICT"),
        ("TOP", .991922, .942076, "CONFLICT"),
        ("MID", .988892, .924092, "NO_CONCLUSION"),
    ):
        visual["lights"][light].update({
            "ok": {"best_similarity": ok},
            "ng": {"best_similarity": ng},
            "ng_margin_over_ok": round(ng - ok, 6),
            "status": status,
        })
    return a, visual


def apply(fused_analysis, visual):
    with patch(
        "src.core.knn_visual_corroboration.evaluate_knn_visual_corroboration",
        return_value=deepcopy(visual),
    ):
        return apply_knn_visual_corroboration(
            deepcopy(fused_analysis), {}, "FALTANDO", object(),
        )


class Oct29RecurringOkTests(unittest.TestCase):
    def test_human_known_top_mid_auto_zero_only_in_continuous_production(self):
        raw, visual = real_event()
        result = apply(raw, visual)
        self.assertEqual(result["verdict"], "FALHA FALSA")
        self.assertIs(result["is_defect"], False)
        self.assertEqual(result["detail"]["fusion_rule"],
                         "operator_known_top_mid_ok_review_20261029_v1")
        self.assertEqual(
            result["detail"]["recognized_top_mid_ok_20261029"][
                "per_light_reference"]["MID"]["path"], SOURCES["MID"])
        self.assertFalse(result["detail"]["recognized_top_mid_ok_20261029"][
            "current_pair_exact_match"])
        self.assertFalse(result["eligible_for_final_decision"])
        self.assertTrue(result["production_review_required"])
        self.assertEqual(verdict_feedback_state(result),
                         ("FALHA FALSA", "ok"))
        result["detail"]["production_continuous_auto"] = True
        policy = production_decision_policy(result)
        self.assertEqual(policy["verdict"], "FALHA FALSA")
        self.assertTrue(policy["auto_allowed"])
        self.assertFalse(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "OK")
        self.assertEqual(policy["reason"],
                         "supervised_operator_known_top_mid_ok_20261029_auto_v1")
        self.assertEqual(verdict_feedback_state(result),
                         ("FALHA FALSA", "ok"))
        # Teste/Sombra não recebem o armamento do controlador Produção.
        result["detail"].pop("production_continuous_auto")
        manual = production_decision_policy(result)
        self.assertFalse(manual["auto_allowed"])
        self.assertTrue(manual["operator_review_required"])
        self.assertEqual(manual["proposed_decision"], "")
        self.assertEqual(manual["reason"],
                         "operator_known_top_mid_ok_20261029_requires_operator")

    def test_auto_ok_requires_complete_verified_capture_and_clean_physical_evidence(self):
        raw, visual = real_event()
        baseline = apply(raw, visual)
        baseline["detail"]["production_continuous_auto"] = True
        for name, change in (
            ("missing_top_route", lambda a: a["detail"]["recognition_light_routes"].pop("TOP")),
            ("mixed_route", lambda a: a["detail"]["recognition_light_routes"].update(SIDE="MEMORY_EXACT")),
            ("physical_review", lambda a: a["detail"]["physical_missing_veto"].update(status="GEOMETRY_REVIEW")),
            ("physical_unavailable", lambda a: a["detail"]["physical_missing_veto"].update(status="UNAVAILABLE")),
            ("physical_suspect", lambda a: a["detail"]["physical_missing_veto"].update(status="SUSPECT_ABSENCE")),
            ("trial_not_verified", lambda a: a["detail"].update(cnn_v2_trial_manual_only=False)),
            ("hard_ng", lambda a: a["detail"].update(hard_missing_evidence=True)),
            ("other_category", lambda a: a["detail"].update(multilight_category="DESLOCADO")),
            ("missing_light", lambda a: a["detail"].update(multilight_final=False)),
            ("wrong_anchor", lambda a: a["detail"]["knn_c_light_neighbor_evidence"]["TOP"]["neighbors_A"][0].update(source_pixel_sha256="0"*64)),
        ):
            with self.subTest(name=name):
                invalid = deepcopy(baseline)
                change(invalid)
                policy = production_decision_policy(invalid)
                self.assertNotEqual(policy.get("proposed_decision"), "OK")
                self.assertFalse(policy.get("auto_allowed"))

    def test_failure_of_any_independent_proof_blocks_recall(self):
        cases = (
            ("wrong_sha", lambda a,r:
                a["detail"]["knn_c_light_neighbor_evidence"]["MID"][
                    "neighbors_A"][0].update(source_pixel_sha256="0"*64)),
            ("wrong_source", lambda a,r:
                a["detail"]["knn_c_light_neighbor_evidence"]["TOP"][
                    "neighbors_B"][0].update(path="public/ok_archive/other.png")),
            ("top_ng_competing", lambda a,r:
                a["detail"]["knn_c_light_neighbor_evidence"]["TOP"][
                    "neighbors_A"][1].update(similarity=.94)),
            ("side_ng_competing", lambda a,r:
                a["detail"]["knn_c_light_neighbor_evidence"]["SIDE"][
                    "neighbors_A"][0].update(similarity=.95)),
            ("side_ng_majority", lambda a,r:
                a["detail"]["knn_c_light_neighbor_evidence"]["SIDE"][
                    "neighbors_A"][2].update(label="NG")),
            ("low_top_signature", lambda a,r:
                a["detail"]["knn_c_light_neighbor_evidence"]["TOP"][
                    "neighbors_A"][0].update(similarity=.984)),
            ("low_mid_signature", lambda a,r:
                a["detail"]["knn_c_light_neighbor_evidence"]["MID"][
                    "neighbors_B"][0].update(similarity=.965)),
            ("wrong_mid_vote", lambda a,r:
                a["detail"]["knn_c_light_observer_votes"]["MID"].update(
                    vote_B="NG")),
            ("wrong_mid_reason", lambda a,r:
                a["detail"]["knn_c_light_observer_votes"]["MID"].update(
                    reason="A_E_B_CONCORDAM")),
            ("bad_checkpoint", lambda a,r:
                a["detail"]["cnn_v2_light_diagnostics"]["MID"].update(
                    checkpoint_verified=False)),
            ("cnn_valid_ng_consensus", lambda a,r:
                a["detail"].update(cnn_v2_consensus="NG")),
            ("hard_missing", lambda a,r:
                a["detail"]["decision_trace"].update(
                    raw_hard_missing_evidence=True)),
            ("physical_suspect", lambda a,r:
                a["detail"]["physical_missing_veto"].update(
                    status="SUSPECT_ABSENCE")),
            ("visual_side_ng_wins", lambda a,r:
                r["lights"]["SIDE"].update(
                    ng={"best_similarity": .998},
                    ng_margin_over_ok=round(.998-.991922,6))),
            ("visual_mid_weak", lambda a,r:
                r["lights"]["MID"].update(
                    ok={"best_similarity": .96},
                    ng_margin_over_ok=round(.924092-.96,6))),
            ("no_mid", lambda a,r:
                a["detail"]["knn_c_light_observer_votes"].pop("MID")),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                a, r = real_event()
                mutate(a, r)
                self.assertIsNone(
                    _recognized_recurrent_ok_20261029(a,r,"FALTANDO"))
                out = apply(a, r)
                self.assertNotEqual(
                    out["detail"].get("fusion_rule"),
                    "operator_known_top_mid_ok_review_20261029_v1")
                self.assertFalse(
                    production_decision_policy(out).get("proposed_decision")
                    == "OK")

    def test_final_marker_revalidated_on_policy_and_feedback(self):
        a, r = real_event()
        final = apply(a,r)
        final["detail"]["production_continuous_auto"] = True
        forged = deepcopy(final)
        forged["detail"]["recognized_top_mid_ok_20261029"][
            "per_light_reference"]["TOP"]["source_pixel_sha256"] = "0"*64
        p = production_decision_policy(forged)
        self.assertNotEqual(p["verdict"], "FALHA FALSA")
        self.assertNotEqual(verdict_feedback_state(forged), ("FALHA FALSA","ok"))
        self.assertNotEqual(p["proposed_decision"], "OK")


if __name__ == "__main__":
    unittest.main()
