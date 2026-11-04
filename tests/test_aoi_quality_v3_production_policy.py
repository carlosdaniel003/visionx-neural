"""V3 época 3 na política XP: veto NG verificável, nunca auto-OK isolado."""
from __future__ import annotations

from copy import deepcopy
import unittest

from src.core.neural.aoi_v3_trial_contract import POINTER_SCHEMA, SNAPSHOT_SHA256
from src.core.neural.aoi_quality_cnn_v3_trial_consensus import (
    attach_v3_multilight_evidence,
)
from src.ui.production_confidence_gate import production_decision_policy


def complete():
    return {
        "verdict": "FALHA FALSA",
        "is_defect": False,
        "confidence": .96,
        "multilight_final": True,
        "detail": {
            "multilight_final": True,
            "production_continuous_auto": True,
            "production_binary_auto": True,
            "recognition_light_routes": {
                "SIDE": "NEW_CNN", "TOP": "NEW_EXPERTS", "MID": "NEW_CNN",
            },
            "decision_trace": {"operator_review_required": False},
        },
    }


def vote(light, score):
    return {
        "schema": POINTER_SCHEMA, "status": "READY", "epoch": 3,
        "light": light, "checkpoint_sha256": SNAPSHOT_SHA256,
        "ng_score_uncalibrated": score,
        "predicted": "NG" if score >= .5 else "OK",
        "ng_veto": score >= .5,
        "automatic_ok_from_v3_authorized": False,
    }


def with_lights(analysis, scores):
    sources = {
        light: {"detail": {"aoi_v3_trial": vote(light, score)}}
        for light, score in scores.items()
    }
    return attach_v3_multilight_evidence(analysis, sources)


class V3ProductionPolicyTests(unittest.TestCase):
    def test_ng_veto_changes_previous_auto_ok_to_auto_ng(self):
        analysis = with_lights(complete(), {
            "SIDE": .01, "TOP": .96, "MID": .03,
        })
        result = production_decision_policy(analysis)
        self.assertTrue(result["auto_allowed"])
        self.assertEqual(result["proposed_decision"], "NG")
        self.assertEqual(result["verdict"], "DEFEITO REAL")
        self.assertEqual(result["reason"], "aoi_quality_v3_epoch3_ng_veto_trial_v1")
        self.assertTrue(result["unverified_physical_label"])
        self.assertEqual(result["v3_checkpoint_sha256"], SNAPSHOT_SHA256)

    def test_v3_ok_never_grants_auto_ok(self):
        a = complete()
        # Não inventar autorização onde o veredito antigo era incerto.
        a["verdict"] = "REVISÃO OBRIGATÓRIA"
        a["production_review_required"] = True
        baseline = production_decision_policy(deepcopy(a))
        with_lights(a, {"SIDE": .02, "TOP": .03, "MID": .01})
        after = production_decision_policy(a)
        self.assertEqual(after["proposed_decision"], baseline["proposed_decision"])
        self.assertEqual(after["reason"], baseline["reason"])
        self.assertNotEqual(after["reason"], "aoi_quality_v3_epoch3_ng_veto_trial_v1")

    def test_missing_light_or_wrong_provenance_stops_xp(self):
        a = with_lights(complete(), {"SIDE": .97, "TOP": .02})
        p = production_decision_policy(a)
        self.assertTrue(p["operational_error"])
        self.assertFalse(p["auto_allowed"])
        self.assertEqual(p["proposed_decision"], "")
        a = with_lights(complete(), {
            "SIDE": .98, "TOP": .02, "MID": .03,
        })
        a["detail"]["aoi_v3_multilight"]["checkpoint_sha256"] = "0"*64
        p = production_decision_policy(a)
        self.assertTrue(p["operational_error"])
        self.assertFalse(p["auto_allowed"])

    def test_disabled_v3_does_not_change_existing_policy(self):
        original = complete()
        before = production_decision_policy(original)
        after = production_decision_policy(deepcopy(original))
        self.assertEqual(before, after)

    def test_partial_single_view_cannot_trigger_xp_ng_without_side_contract(self):
        a = complete()
        a["multilight_final"] = False
        a["detail"]["multilight_final"] = False
        a["lighting_mode"] = "SIDE"
        a["detail"]["aoi_v3_trial"] = vote("SIDE", .99)
        p = production_decision_policy(a)
        self.assertTrue(p["operational_error"])
        self.assertFalse(p["auto_allowed"])


if __name__ == "__main__":
    unittest.main()
