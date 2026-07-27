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

    def test_human_agreement_saves_json_without_images(self):
        panel = self.panel_class(ai_defect=True)
        with patch(
            "src.services.anomaly_learning.DatasetManager.save_sample",
            return_value="memory.json",
        ) as save:
            panel.save_label("NG", source="button")

        self.assertFalse(save.call_args.kwargs["save_images"])
        self.assertEqual(save.call_args.kwargs["ai_decision"], "NG")
        self.assertEqual(panel.orchestrator.reloads, 1)

    def test_human_disagreement_adds_audit_images(self):
        panel = self.panel_class(ai_defect=False)
        with patch(
            "src.services.anomaly_learning.DatasetManager.save_sample",
            return_value="memory.json",
        ) as save:
            panel.save_label("NG", source="button")

        self.assertTrue(save.call_args.kwargs["save_images"])
        self.assertEqual(save.call_args.kwargs["ai_decision"], "OK")

    def test_automatic_decision_does_not_self_train(self):
        panel = self.panel_class(ai_defect=True)
        with patch(
            "src.services.anomaly_learning.DatasetManager.save_sample"
        ) as save:
            result = panel.save_label("NG", source="auto")

        save.assert_not_called()
        self.assertEqual(result, "original")
        self.assertEqual(panel.original_calls, 1)


if __name__ == "__main__":
    unittest.main()
