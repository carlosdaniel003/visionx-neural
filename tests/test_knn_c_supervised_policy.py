"""KNN C 0/1 supervisionada: somente consenso real de 3 luzes com armamento."""
from __future__ import annotations

import unittest

from src.core.multilight_fusion import fuse_multilight
from src.core.knn_c_supervised_policy import supervised_knn_c_decision
from src.ui.production_confidence_gate import production_decision_policy

LIGHTS = ("SIDE", "TOP", "MID")


def light_analysis(light, label, *, status=None, conflict=False,
                   independent_review=False, route="NEW_EXPERTS", bank="READY"):
    selected = status or "SUGERE_" + label
    ng = label == "NG"
    raw = {
        "is_defect": ng,
        "verdict": "DEFEITO REAL" if ng else "FALHA FALSA",
        "confidence": .97,
        "production_review_required": True,
        "detail": {
            "recognition_route": route,
            "final_score": .93 if ng else .08,
            "physical_score": .90 if ng else .07,
            "fusion_rule": "local",
            "dominant_engine": "ssim",
            "operator_review_required": True,
            "decision_trace": {"operator_review_required": True},
        },
    }
    raw["detail"]["knn_c"] = {
        "schema": "visionx.knn_c_live_review_gated.v1",
        "bank_status": bank,
        "status": selected, "source": "A_E_B",
        "vote_A": label, "vote_B": label,
        "suggestion_A": selected, "suggestion_B": selected,
        "pre_c_review_required": independent_review,
        "pre_c_verdict": "DEFEITO REAL" if ng else "FALHA FALSA",
        "pre_c_is_defect": ng,
        "requires_human_confirmation": True,
        "conflicts_with_existing_verdict": conflict,
    }
    return raw


def aggregate(label, *, arm=False, edit=None):
    cases = {light: light_analysis(light, label) for light in LIGHTS}
    if edit:
        edit(cases)
    result = fuse_multilight(cases, "INVERTIDO")
    result["detail"]["knn_c_supervised_armed"] = bool(arm)
    return result


class KNNCSupervisedXPTests(unittest.TestCase):
    def test_both_ok_and_ng_reach_existing_xp_policy_only_when_armed(self):
        for label in ("OK", "NG"):
            with self.subTest(label=label):
                result = aggregate(label, arm=False)
                self.assertEqual(
                    result["detail"]["knn_c_supervised_candidate"]["label"], label
                )
                self.assertIsNone(supervised_knn_c_decision(result))
                self.assertFalse(production_decision_policy(result)["auto_allowed"])
                result["detail"]["knn_c_supervised_armed"] = True
                self.assertEqual(supervised_knn_c_decision(result), label)
                policy = production_decision_policy(result)
                self.assertTrue(policy["auto_allowed"])
                self.assertFalse(policy["operator_review_required"])
                self.assertEqual(policy["proposed_decision"], label)
                self.assertEqual(
                    policy["reason"],
                    "knn_c_supervised_three_light_" + label.lower(),
                )

    def test_one_light_in_review_never_arms_auto_xp(self):
        def mutate(values):
            values["MID"]["detail"]["knn_c"]["status"] = "REVISAO"
        result = aggregate("OK", arm=True, edit=mutate)
        self.assertNotIn("knn_c_supervised_candidate", result["detail"])
        self.assertFalse(production_decision_policy(result)["auto_allowed"])

    def test_wrong_or_conflicting_proposal_never_auto(self):
        for alteration in ("mismatch", "review", "bank", "exact"):
            with self.subTest(alteration=alteration):
                def mutate(values):
                    cand = values["SIDE"]["detail"]["knn_c"]
                    if alteration == "mismatch":
                        cand["vote_A"] = "NG"
                    elif alteration == "review":
                        cand["pre_c_review_required"] = True
                    elif alteration == "bank":
                        cand["bank_status"] = "UNAVAILABLE"
                    else:
                        values["SIDE"]["detail"]["recognition_route"] = "KNOWN_KNN"
                result = aggregate("OK", arm=True, edit=mutate)
                self.assertNotIn("knn_c_supervised_candidate", result["detail"])
                self.assertFalse(production_decision_policy(result)["auto_allowed"])

    def test_ng_precedence_prevents_auto_ok(self):
        def mutate(values):
            values["TOP"] = light_analysis("TOP", "NG")
        result = aggregate("OK", arm=True, edit=mutate)
        self.assertNotIn("knn_c_supervised_candidate", result["detail"])
        self.assertFalse(production_decision_policy(result)["auto_allowed"])

    def test_arm_marker_removed_before_dispatch_denies(self):
        result = aggregate("OK", arm=True)
        self.assertTrue(production_decision_policy(result)["auto_allowed"])
        result["detail"]["knn_c_supervised_armed"] = False
        self.assertFalse(production_decision_policy(result)["auto_allowed"])

    def test_unfinished_and_forged_diagnostics_fail_closed(self):
        result = aggregate("OK", arm=True)
        for key, value in (
            ("multilight_final", False),
            ("production_review_required", False),
        ):
            modified = dict(result)
            modified[key] = value
            self.assertIsNone(supervised_knn_c_decision(modified))
        result["detail"]["knn_c_light_diagnostics"].pop("SIDE")
        self.assertIsNone(supervised_knn_c_decision(result))


if __name__ == "__main__":
    unittest.main()
