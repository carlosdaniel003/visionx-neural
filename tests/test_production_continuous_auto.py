"""Production continuous automation: qualified OK, conservative NG, pause, quarantine.

Physical evidence unresolved must never promote an uncertain PCB to OK.
An incomplete capture or failed XP command still requires a controlled stop.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtWidgets import QApplication

from src.services.decision_persistence import (
    _is_unverified_automatic, _quarantine_unverified_automatic,
)
from src.ui.decision_verdict_feedback import verdict_feedback_state
from src.ui.production_confidence_gate import production_decision_policy
from src.ui.production_autonomy_controller import (
    KNN_C_SUPERVISED_DELAY_MS,
    ProductionAutonomyController,
)
from tests.test_production_autonomy_controller import _Panel


def uncertain_three_light():
    return {
        "verdict": "REVISÃO OBRIGATÓRIA", "is_defect": False,
        "multilight_final": True, "production_review_required": True,
        "confidence": .5,
        "detail": {
            "multilight_final": True,
            "production_continuous_auto": True,
            "recognition_light_routes": {
                "SIDE": "NEW_EXPERTS", "TOP": "NEW_EXPERTS",
                "MID": "NEW_EXPERTS",
            },
            "decision_trace": {"operator_review_required": True},
        },
    }


class ContinuousAutoPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_uncertain_full_capture_automatically_rejects_not_approves(self):
        analysis = uncertain_three_light()
        policy = production_decision_policy(analysis)
        self.assertEqual(policy["proposed_decision"], "NG")
        self.assertEqual(policy["verdict"], "NG PREVENTIVO")
        self.assertTrue(policy["auto_allowed"])
        self.assertFalse(policy["operator_review_required"])
        self.assertTrue(policy["unverified_physical_label"])
        self.assertEqual(
            policy["reason"], "production_uncertain_auto_ng_quarantine_v1",
        )
        self.assertEqual(verdict_feedback_state(analysis), ("NG PREVENTIVO", "ng"))

    def test_not_engaged_outside_production_controller(self):
        a = uncertain_three_light()
        a["detail"].pop("production_continuous_auto")
        policy = production_decision_policy(a)
        self.assertFalse(policy["auto_allowed"])
        self.assertTrue(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "")

    def test_incomplete_or_invalid_multilight_never_autosends(self):
        variants = (
            lambda a: a.update(multilight_final=False),
            lambda a: a["detail"].update(multilight_final=False),
            lambda a: a["detail"].pop("recognition_light_routes"),
            lambda a: a["detail"]["recognition_light_routes"].pop("TOP"),
            lambda a: a["detail"]["recognition_light_routes"].update(MID="UNKNOWN"),
            lambda a: a["detail"].update(capture_error=True),
            lambda a: a["detail"].update(multilight_physical_disagreement=True),
        )
        for mutation in variants:
            with self.subTest(mutation=str(mutation)):
                a = uncertain_three_light()
                mutation(a)
                policy = production_decision_policy(a)
                self.assertFalse(policy["auto_allowed"])

    def test_controller_send_ng_once_after_five_second_pause_window(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        a = uncertain_three_light()
        a["detail"].pop("production_continuous_auto")
        panel.current_analysis = a
        self.assertTrue(controller.analysis_ready(a))
        self.assertTrue(a["detail"]["production_continuous_auto"])
        controller._stage_timer.stop()
        controller.pending_analysis = a
        controller.generation += 1
        controller._finish_presentation(controller.generation)
        self.assertEqual(controller.state, "auto_decision_pending")
        self.assertEqual(controller._stage_remaining_ms, KNN_C_SUPERVISED_DELAY_MS)
        self.assertEqual(panel.saved, [])
        controller._stage_timer.stop()
        controller._emit_auto_decision(controller.generation, "NG")
        self.assertEqual(panel.saved, [("NG", "production_auto")])
        self.assertEqual(panel.auto_records, ["NG"])
        controller._emit_auto_decision(controller.generation, "NG")
        self.assertEqual(panel.saved, [("NG", "production_auto")])

    def test_pause_or_changed_capture_cancels_auto_ng(self):
        for scenario in ("pause", "change", "mode"):
            with self.subTest(scenario=scenario):
                panel = _Panel()
                controller = ProductionAutonomyController(panel)
                panel.combo_mode.setCurrentText("Modo Produção")
                a = uncertain_three_light()
                panel.current_analysis = a
                controller.pending_analysis = a
                controller.generation += 1
                controller._finish_presentation(controller.generation)
                controller._stage_timer.stop()
                if scenario == "pause":
                    controller.toggle_pause()
                elif scenario == "change":
                    panel.current_analysis = deepcopy(a)
                else:
                    panel.combo_mode.setCurrentText("Modo Teste")
                controller._emit_auto_decision(controller.generation, "NG")
                self.assertEqual(panel.saved, [])


class QuarantineUnverifiedLabelsTests(unittest.TestCase):
    def test_unverified_auto_is_not_used_as_training_truth(self):
        a = uncertain_three_light()
        a["detail"]["production_decision_policy"] = {
            "reason": "production_uncertain_auto_ng_quarantine_v1",
            "resolution": "automatic",
        }
        samples = [
            {
                "lighting_mode": light,
                "sample_image": np.full((16, 20, 3), 30, dtype=np.uint8),
                "test_image": np.full((16, 20, 3), 40, dtype=np.uint8),
            }
            for light in ("SIDE", "TOP", "MID")
        ]
        task = {
            "source": "production_auto", "label": "NG",
            "event_id": "test-unverified", "analysis": a,
            "multilight_samples": samples,
        }
        self.assertTrue(_is_unverified_automatic(task))
        self.assertFalse(_is_unverified_automatic({**task, "source": "button"}))
        with tempfile.TemporaryDirectory() as temporary:
            with patch("src.services.decision_persistence.Path.cwd",
                       return_value=Path(temporary)):
                result = _quarantine_unverified_automatic(task)
            info = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(info["decision_sent_for_xp"], "NG")
            self.assertFalse(info["used_for_knn_training"])
            self.assertFalse(info["used_for_cnn_training"])
            self.assertEqual(len(info["files"]), 6)
            self.assertTrue(all(
                (result.parent / item["file"]).is_file()
                for item in info["files"]
            ))

    def test_two_approved_auto_ok_routes_also_quarantined(self):
        for reason in (
            "supervised_recurrent_ok_top_mid_auto_v1",
            "supervised_mixed_light_ok_auto_v1",
        ):
            with self.subTest(reason=reason):
                self.assertTrue(_is_unverified_automatic({
                    "source": "production_auto",
                    "analysis": {"detail": {"production_decision_policy": {
                        "reason": reason, "resolution": "automatic",
                    }}},
                }))


if __name__ == "__main__":
    unittest.main()
