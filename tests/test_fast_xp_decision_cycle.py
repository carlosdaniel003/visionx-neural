import threading
import unittest

import numpy as np

from src.services.anomaly_learning import install_anomaly_learning
from src.services.decision_persistence import DecisionPersistenceQueue
from src.ui.network_image_cycle_gate import install_network_image_cycle_gate


class FakeButton:
    def __init__(self):
        self.enabled = True
        self.text = ""

    def setEnabled(self, value):
        self.enabled = bool(value)

    def setText(self, value):
        self.text = str(value)


class FakeReceiver:
    def __init__(self, events):
        self.events = events
        self.locked = True
        self.release_count = 0

    def lock_image_gate(self):
        self.locked = True

    def release_image_gate(self):
        self.locked = False
        self.release_count += 1
        self.events.append("gate_released")


class FakePresenter:
    def __init__(self, panel=None):
        self.panel = panel

    def sync(self, force=False):
        return force


class FastDecisionPanel:
    _anomaly_learning_installed = False
    _network_image_cycle_gate_installed = False

    def __init__(self):
        self.events = []
        self.network_receiver = FakeReceiver(self.events)
        self.current_sample = np.full((20, 20, 3), 30, dtype=np.uint8)
        self.current_ng = np.full((20, 20, 3), 80, dtype=np.uint8)
        self.current_aoi_info = {"category": "Shifted"}
        self.current_analysis = {"is_defect": True, "detail": {"final_score": 0.8}}
        self.capture_cycle_active = True
        self.capture_cycle_source = "network"
        self.capture_cycle_ignored_signals = 0
        self.is_locked = True

        self.btn_start = FakeButton()
        self.btn_save_ok = FakeButton()
        self.btn_save_ng = FakeButton()
        self.btn_skip = FakeButton()

        self.submitted_tasks = []
        self._anomaly_persistence_submitter = self._capture_task

    def _capture_task(self, task):
        self.submitted_tasks.append(task)
        self.events.append("persistence_queued")

    def send_command_to_xp(self, key):
        self.events.append(f"send_{key}")
        return True

    def prepare_for_next_network_image(self):
        self.events.append("ui_cleared")
        self.current_sample = None
        self.current_ng = None
        self.current_analysis = None
        self.current_aoi_info = {}

    def update_history_status(self, label, source):
        self.events.append(f"history_{label}_{source}")

    def update_brain_status(self, _message, _active=False):
        pass

    def handle_network_image(self, _image, _ip):
        return True

    def start_monitoring(self):
        return True

    def skip_image(self):
        self.is_locked = False

    def save_label(self, _decision, source="button"):
        self.events.append(f"legacy_save_{source}")
        self.is_locked = False
        return "legacy"


install_anomaly_learning(FastDecisionPanel)
install_network_image_cycle_gate(FastDecisionPanel, FakePresenter)


class FastDecisionCycleTests(unittest.TestCase):
    def test_zero_clears_old_capture_and_releases_gate_before_persistence_runs(self):
        panel = FastDecisionPanel()
        panel.capture_cycle_active = True
        panel.capture_cycle_source = "network"
        panel.network_receiver.locked = True
        original_ng = panel.current_ng.copy()

        panel.save_label("OK", source="button")

        self.assertEqual(
            panel.events[:4],
            ["send_0", "ui_cleared", "history_OK_button", "persistence_queued"],
        )
        self.assertEqual(panel.events[-1], "gate_released")
        self.assertFalse(panel.capture_cycle_active)
        self.assertFalse(panel.network_receiver.locked)
        self.assertFalse(panel.is_locked)
        self.assertIsNone(panel.current_ng)

        self.assertEqual(len(panel.submitted_tasks), 1)
        task = panel.submitted_tasks[0]
        self.assertTrue(np.array_equal(task["ng_image"], original_ng))
        self.assertEqual(task["label"], "OK")
        self.assertEqual(task["ai_decision"], "NG")
        self.assertTrue(task["save_images"])

    def test_xp_keyboard_decision_does_not_echo_press_command_back_to_xp(self):
        panel = FastDecisionPanel()
        panel.capture_cycle_active = True
        panel.capture_cycle_source = "network"
        panel.network_receiver.locked = True

        panel.save_label("NG", source="xp_keyboard")

        self.assertNotIn("send_1", panel.events)
        self.assertIn("ui_cleared", panel.events)
        self.assertIn("persistence_queued", panel.events)
        self.assertIn("gate_released", panel.events)


class FakeDatasetManager:
    calls = []
    completed = threading.Event()

    @classmethod
    def save_sample(cls, **task):
        cls.calls.append(task["label"])
        cls.completed.set()
        return "memory.json"


class FakeOrchestrator:
    def __init__(self):
        self.reload_count = 0

    def reload_memory(self):
        self.reload_count += 1


class DecisionPersistenceQueueTests(unittest.TestCase):
    def setUp(self):
        FakeDatasetManager.calls = []
        FakeDatasetManager.completed = threading.Event()

    def test_background_queue_saves_and_then_reloads_memory(self):
        orchestrator = FakeOrchestrator()
        queue = DecisionPersistenceQueue(
            orchestrator,
            dataset_manager=FakeDatasetManager,
        )
        queue.submit({"label": "OK"})

        self.assertTrue(FakeDatasetManager.completed.wait(timeout=2.0))
        queue.wait_until_idle()

        self.assertEqual(FakeDatasetManager.calls, ["OK"])
        self.assertEqual(orchestrator.reload_count, 1)
        self.assertEqual(queue.pending_count(), 0)


if __name__ == "__main__":
    unittest.main()
