import unittest
from unittest.mock import patch

import numpy as np

from src.services.anomaly_learning import install_anomaly_learning


class _Button:
    def setEnabled(self, value):
        self.enabled = value

    def setText(self, value):
        self.text = value


class _Orchestrator:
    def __init__(self):
        self.reloads = 0

    def reload_memory(self):
        self.reloads += 1


class _Panel:
    def __init__(self, ai_defect=False):
        self.current_ng = np.zeros((8, 8, 3), dtype=np.uint8)
        self.current_sample = self.current_ng.copy()
        self.current_aoi_info = {"category": "Much Adhesive"}
        self.current_analysis = {"is_defect": ai_defect, "detail": {}}
        self.btn_save_ok = _Button()
        self.btn_save_ng = _Button()
        self.btn_skip = _Button()
        self.btn_start = _Button()
        self.orchestrator = _Orchestrator()
        self.is_locked = True
        self.original_calls = 0

    def save_label(self, user_decision, source="button"):
        self.original_calls += 1
        return "original"

    def send_command_to_xp(self, value):
        self.command = value

    def update_brain_status(self, *args):
        self.brain_status = args

    def update_history_status(self, *args):
        self.history_status = args


class AnomalyLearningPolicyTests(unittest.TestCase):
    def setUp(self):
        class Panel(_Panel):
            pass

        self.panel_class = Panel
        install_anomaly_learning(self.panel_class)

    def test_human_agreement_submits_confirmed_memory_and_audit_task(self):
        panel = self.panel_class(ai_defect=True)
        saved = []
        confirmed = []
        panel._live_memory_writer = lambda task: (
            confirmed.append(task) or {"committed": True}
        )
        panel._anomaly_persistence_submitter = saved.append
        panel.save_label("NG", source="button")
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["label"], "NG")
        self.assertEqual(saved[0]["source"], "button")
        self.assertEqual(saved[0]["ai_decision"], "NG")
        self.assertEqual(panel.original_calls, 0)

    def test_human_disagreement_submits_difficult_case_with_images(self):
        panel = self.panel_class(ai_defect=False)
        saved = []
        panel._live_memory_writer = lambda task: {"committed": True}
        panel._anomaly_persistence_submitter = saved.append
        panel.save_label("NG", source="button")
        self.assertEqual(len(saved), 1)
        self.assertTrue(saved[0]["save_images"])
        self.assertEqual(saved[0]["ai_decision"], "OK")

    def test_test_mode_button_records_xp_ack_for_post_judgment_reinspection(self):
        from types import SimpleNamespace
        for acknowledged in (True, False):
            with self.subTest(acknowledged=acknowledged):
                panel = self.panel_class(ai_defect=True)
                panel.combo_mode = SimpleNamespace(
                    currentText=lambda: "Modo Teste",
                )
                commands = []
                def send(command):
                    commands.append(command)
                    return acknowledged
                panel.send_command_to_xp = send
                panel._live_memory_writer = lambda _task: {"committed": True}
                panel._anomaly_persistence_submitter = lambda _task: None

                panel.save_label("NG", source="button")

                self.assertEqual(commands, ["1"])
                self.assertIs(panel.last_decision_command_success, acknowledged)
                # Não alterar a semântica já existente do Teste.
                self.assertFalse(panel.is_locked)

    def test_reinspection_in_test_requires_second_xp_command_before_learning(self):
        from types import SimpleNamespace
        for acknowledged in (True, False):
            with self.subTest(acknowledged=acknowledged):
                panel = self.panel_class(ai_defect=True)
                panel.combo_mode = SimpleNamespace(
                    currentText=lambda: "Modo Teste",
                )
                panel.capture_cycle_source = "network"
                panel.network_reinspection_of_judged_event = True
                panel.network_intake_last_image_event_id = "new-event"
                commands = []
                panel.send_command_to_xp = (
                    lambda value: commands.append(value) or acknowledged
                )
                learned = []
                persisted = []
                panel._live_memory_writer = (
                    lambda task: learned.append(task) or {"committed": True}
                )
                panel._anomaly_persistence_submitter = persisted.append

                result = panel.save_label("OK", source="button")

                self.assertEqual(commands, ["0"])
                self.assertIs(panel.last_decision_command_success, acknowledged)
                if acknowledged:
                    self.assertEqual(len(learned), 1)
                    self.assertEqual(len(persisted), 1)
                    self.assertEqual(persisted[0]["event_id"], "new-event")
                    self.assertTrue(
                        persisted[0]["reinspection_of_judged_event"]
                    )
                    self.assertTrue(persisted[0]["save_images"])
                    self.assertFalse(panel.is_locked)
                else:
                    self.assertIs(result, False)
                    self.assertEqual(learned, [])
                    self.assertEqual(persisted, [])
                    self.assertTrue(panel.is_locked)

    def test_failed_production_manual_command_must_not_learn_or_advance(self):
        from types import SimpleNamespace
        panel = self.panel_class(ai_defect=True)
        panel.combo_mode = SimpleNamespace(
            currentText=lambda: "Modo Produção",
        )
        panel.send_command_to_xp = lambda _command: False
        with patch(
            "src.services.anomaly_learning.immediate_confirmed_memory"
        ) as persist, patch(
            "src.services.anomaly_learning._submit_decision_persistence"
        ) as async_submit:
            self.assertIs(panel.save_label("OK", source="button"), False)
        persist.assert_not_called()
        async_submit.assert_not_called()
        self.assertTrue(panel.is_locked)
        self.assertEqual(panel.original_calls, 0)

    def test_local_test_correction_uses_current_mss_frame_not_old_xp_id(self):
        from src.services.anomaly_learning import _decision_task
        panel = self.panel_class(ai_defect=True)
        panel.capture_cycle_source = "local"
        panel.adhesive_multilight_primary_event_id = "old-multilight-event"
        panel.network_intake_last_image_event_id = "old-network-event"
        panel.capture_debug_last_record = {
            "event_id": "local-event-02", "source": "local_mss",
        }
        panel.capture_debug_last_image_event_id = "local-event-02"
        panel.capture_debug_last_image = np.ones(
            (20, 30, 3), dtype=np.uint8,
        )
        result = _decision_task(
            panel, normalized="OK", source="button", ai_decision="NG",
        )
        self.assertEqual(result["event_id"], "local-event-02")
        self.assertTrue(np.array_equal(
            result["source_frame"], panel.capture_debug_last_image,
        ))

    def test_automatic_decision_does_not_self_train(self):
        panel = self.panel_class(ai_defect=True)
        saved = []
        panel._anomaly_persistence_submitter = saved.append
        self.assertEqual(panel.save_label("NG", source="auto"), "original")
        self.assertEqual(panel.original_calls, 1)
        self.assertEqual(saved, [])

    def test_production_auto_ok_does_not_self_train(self):
        panel = self.panel_class(ai_defect=False)
        saved = []
        panel._anomaly_persistence_submitter = saved.append
        self.assertEqual(
            panel.save_label("OK", source="production_auto"), "original"
        )
        self.assertEqual(panel.original_calls, 1)
        self.assertEqual(saved, [])


if __name__ == "__main__":
    unittest.main()
