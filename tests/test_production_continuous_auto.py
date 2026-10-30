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
from tests.test_oct29_operator_recurring_ok import real_event, apply as apply_oct29


def uncertain_three_light():
    return {
        "verdict": "REVISÃO OBRIGATÓRIA", "is_defect": False,
        "multilight_final": True, "production_review_required": True,
        "confidence": .5,
        "detail": {
            "multilight_final": True,
            "production_continuous_auto": True,
            "production_binary_auto": True,
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
        self.assertEqual(policy["verdict"], "DEFEITO REAL")
        self.assertTrue(policy["auto_allowed"])
        self.assertFalse(policy["operator_review_required"])
        self.assertTrue(policy["unverified_physical_label"])
        self.assertEqual(
            policy["reason"], "production_binary_unverified_ng_v1",
        )
        self.assertEqual(verdict_feedback_state(analysis), ("DEFEITO REAL", "ng"))
        self.assertEqual(policy["source_verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertTrue(policy["unverified_physical_label"])

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
            lambda a: a["detail"].update(category_unknown=True),
        )
        for mutation in variants:
            with self.subTest(mutation=str(mutation)):
                a = uncertain_three_light()
                mutation(a)
                policy = production_decision_policy(a)
                self.assertFalse(policy["auto_allowed"])
                self.assertFalse(policy["operator_review_required"])
                self.assertEqual(policy["proposed_decision"], "")
                self.assertTrue(policy["operational_error"])

    def test_complete_physical_disagreement_is_binary_ng_not_manual(self):
        analysis = uncertain_three_light()
        analysis["detail"]["multilight_physical_disagreement"] = True
        policy = production_decision_policy(analysis)
        self.assertTrue(policy["auto_allowed"])
        self.assertFalse(policy["operator_review_required"])
        self.assertEqual(policy["proposed_decision"], "NG")
        self.assertEqual(policy["verdict"], "DEFEITO REAL")

    def test_verdict_binary_for_complete_ng_or_ok_only(self):
        analysis = uncertain_three_light()
        for raw_verdict in ("REVISÃO OBRIGATÓRIA", "NG PREVENTIVO", "DEFEITO REAL"):
            with self.subTest(raw_verdict=raw_verdict):
                analysis["verdict"] = raw_verdict
                policy = production_decision_policy(analysis)
                self.assertEqual(
                    (policy["proposed_decision"], policy["verdict"]),
                    ("NG", "DEFEITO REAL"),
                )
        self.assertEqual(verdict_feedback_state(analysis), ("DEFEITO REAL", "ng"))

    def test_operational_fault_stops_without_human_quality_review(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        analysis = uncertain_three_light()
        analysis["detail"]["capture_error"] = True
        panel.current_analysis = analysis
        controller.pending_analysis = analysis
        controller.generation += 1
        controller._finish_presentation(controller.generation)
        self.assertEqual(controller.state, "operational_fault")
        self.assertFalse(panel.production_review_pending)
        self.assertEqual(panel.saved, [])
        self.assertEqual(verdict_feedback_state(analysis), ("", ""))

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

    def test_verified_oct29_recurring_ok_sends_zero_once_after_pause_window(self):
        panel = _Panel()
        controller = ProductionAutonomyController(panel)
        panel.combo_mode.setCurrentText("Modo Produção")
        raw, visual = real_event()
        analysis = apply_oct29(raw, visual)
        panel.current_analysis = analysis
        self.assertTrue(controller.analysis_ready(analysis))
        self.assertTrue(analysis["detail"]["production_continuous_auto"])
        controller._stage_timer.stop()
        controller.pending_analysis = analysis
        controller._finish_presentation(controller.generation)
        self.assertEqual(controller.state, "auto_decision_pending")
        self.assertEqual(controller._stage_remaining_ms, KNN_C_SUPERVISED_DELAY_MS)
        self.assertEqual(panel.saved, [])
        controller._stage_timer.stop()
        controller._emit_auto_decision(controller.generation, "OK")
        self.assertEqual(panel.saved, [("OK", "production_auto")])
        self.assertEqual(panel.auto_records, ["OK"])
        controller._emit_auto_decision(controller.generation, "OK")
        self.assertEqual(panel.saved, [("OK", "production_auto")])

    def test_verified_oct29_auto_zero_cannot_bypass_pause_or_event_change(self):
        for state in ("pause", "new_event", "exit_production"):
            with self.subTest(state=state):
                panel = _Panel()
                controller = ProductionAutonomyController(panel)
                panel.combo_mode.setCurrentText("Modo Produção")
                raw, visual = real_event()
                analysis = apply_oct29(raw, visual)
                analysis["detail"]["production_continuous_auto"] = True
                panel.current_analysis = analysis
                controller.pending_analysis = analysis
                controller.generation += 1
                controller._finish_presentation(controller.generation)
                controller._stage_timer.stop()
                if state == "pause":
                    controller.toggle_pause()
                elif state == "new_event":
                    panel.current_analysis = deepcopy(analysis)
                else:
                    panel.combo_mode.setCurrentText("Modo Teste")
                controller._emit_auto_decision(controller.generation, "OK")
                self.assertEqual(panel.saved, [])

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
            "supervised_operator_known_top_mid_ok_20261029_auto_v1",
        ):
            with self.subTest(reason=reason):
                self.assertTrue(_is_unverified_automatic({
                    "source": "production_auto",
                    "analysis": {"detail": {"production_decision_policy": {
                        "reason": reason, "resolution": "automatic",
                    }}},
                }))


class ProductionAutoArchiveIsolationTests(unittest.TestCase):
    def test_control_panel_auto_ng_does_not_write_training_dataset(self):
        from src.ui.control_panel import ControlPanel
        fake = _Panel()
        fake.current_ng = np.zeros((32, 32, 3), dtype=np.uint8)
        fake.current_analysis = uncertain_three_light()
        fake.current_analysis["detail"]["production_decision_policy"] = {
            "reason": "production_uncertain_auto_ng_quarantine_v1",
            "resolution": "automatic",
        }
        sent = []
        fake.send_command_to_xp = lambda cmd: sent.append(cmd) or True
        with patch(
            "src.ui.control_panel.DatasetManager.save_sample",
            side_effect=AssertionError("NG auto must never teach KNN"),
        ):
            result = ControlPanel.save_label(fake, "NG", source="production_auto")
        self.assertIsNone(result)
        self.assertEqual(sent, ["1"])
        self.assertIs(fake.last_decision_command_success, True)

    def test_ng_archive_hook_excludes_unverified_auto(self):
        from src.services.ng_image_archive import install_ng_image_archive
        class Plain:
            def save_label(self, label, source="button"):
                self.called = (label, source)
                return None
        install_ng_image_archive(Plain)
        p = Plain.__new__(Plain)
        p.current_ng = np.zeros((16, 16, 3), dtype=np.uint8)
        p.current_analysis = uncertain_three_light()
        p.current_analysis["detail"]["production_decision_policy"] = {
            "reason": "production_uncertain_auto_ng_quarantine_v1",
            "resolution": "automatic",
        }
        p.capture_cycle_source = "network"
        p.current_aoi_info = {"category": "FALTANDO"}
        p.ng_archive_enabled = True
        with patch(
            "src.services.ng_image_archive.archive_image_candidates",
            side_effect=AssertionError("must not save auto-NG into NG archive"),
        ):
            p.save_label("NG", source="production_auto")
        self.assertEqual(p.called, ("NG", "production_auto"))

    def test_anomaly_learning_routes_autonomous_decision_to_quarantine(self):
        from src.services.anomaly_learning import install_anomaly_learning
        class Plain:
            def save_label(self, label, source="button"):
                self.last_decision_command_success = True
                self.called = (label, source)
                return None
        install_anomaly_learning(Plain)
        p = Plain.__new__(Plain)
        p.current_ng = np.zeros((16, 16, 3), dtype=np.uint8)
        p.current_analysis = uncertain_three_light()
        p.current_analysis["detail"]["production_decision_policy"] = {
            "reason": "production_uncertain_auto_ng_quarantine_v1",
            "resolution": "automatic",
        }
        task = {
            "source": "production_auto",
            "analysis": p.current_analysis,
            "label": "NG",
        }
        with patch(
            "src.services.anomaly_learning._decision_task", return_value=task,
        ) as snapshot, patch(
            "src.services.anomaly_learning._submit_decision_persistence",
        ) as queued:
            p.save_label("NG", source="production_auto")
        snapshot.assert_called_once()
        queued.assert_called_once_with(p, task)
        self.assertEqual(p.called, ("NG", "production_auto"))


if __name__ == "__main__":
    unittest.main()
