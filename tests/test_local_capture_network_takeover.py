import unittest

from src.ui.local_capture_safety import install_local_capture_safety
from src.ui.network_image_cycle_gate import install_network_image_cycle_gate


class FakeSignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class FakeReceiver:
    def __init__(self):
        self.locked = False
        self.lock_count = 0
        self.release_count = 0

    def lock_image_gate(self):
        self.locked = True
        self.lock_count += 1

    def release_image_gate(self):
        self.locked = False
        self.release_count += 1


class FakeMonitor:
    def __init__(self):
        self.layout_detected = FakeSignal()
        self.log_updated = FakeSignal()
        self.finished = FakeSignal()
        self.running = False
        self.start_count = 0

    def start(self):
        self.running = True
        self.start_count += 1

    def isRunning(self):
        return self.running


class FakeButton:
    def __init__(self):
        self.enabled = True
        self.text_value = ""
        self.tooltip = ""

    def setEnabled(self, value):
        self.enabled = bool(value)

    def setText(self, value):
        self.text_value = str(value)

    def setToolTip(self, value):
        self.tooltip = str(value)


class FakeLabel:
    def __init__(self):
        self.value = ""

    def setText(self, value):
        self.value = str(value)


class FakeCombo:
    def currentText(self):
        return "Modo Teste"


class FakePanel:
    def __init__(self):
        self.network_receiver = FakeReceiver()
        self.combo_mode = FakeCombo()
        self.current_analysis = None
        self.current_sample = None
        self.current_ng = None
        self.current_aoi_info = {}
        self.is_locked = False
        self.production_review_pending = False
        self.last_xp_ip = None
        self.minimized = False
        self.start_calls = 0
        self.status_messages = []
        self.local_scheduled = []
        self.network_scheduled = []
        self.created_monitors = []

        self.btn_start = FakeButton()
        self.btn_skip = FakeButton()
        self.btn_save_ok = FakeButton()
        self.btn_save_ng = FakeButton()
        self.lbl_operation_hint = FakeLabel()

        self._local_capture_scheduler = (
            lambda delay, callback: self.local_scheduled.append((delay, callback))
        )
        self._network_cycle_scheduler = (
            lambda delay, callback: self.network_scheduled.append((delay, callback))
        )

        def factory():
            monitor = FakeMonitor()
            self.created_monitors.append(monitor)
            return monitor

        self._screen_monitor_factory = factory

    def handle_network_image(self, _image, ip):
        # Simula uma imagem recebida que não contém barras/recorte AOI válido.
        self.last_xp_ip = ip
        self.is_locked = True
        self.current_analysis = None
        self.current_sample = None
        self.current_ng = None

    def start_monitoring(self):
        self.start_calls += 1
        self.last_xp_ip = None
        self.is_locked = True
        self.minimized = True

    def _start_radar(self):
        raise AssertionError("A supervisão MSS deve substituir este método")

    def process_aoi_images(self, *_args):
        self.current_analysis = {"confidence": 0.8}
        self.current_sample = object()
        self.current_ng = object()
        self.is_locked = True
        self.minimized = False

    def skip_image(self):
        self.current_analysis = None
        self.current_sample = None
        self.current_ng = None
        self.current_aoi_info = {}
        self.is_locked = False

    def save_label(self, *_args, **_kwargs):
        self.is_locked = False

    def update_brain_status(self, message, _active=False):
        self.status_messages.append(str(message))

    def update_network_status(self, message):
        self.status_messages.append(str(message))

    def _reset_confidence_panel(self):
        pass

    def _reset_reference_panel(self):
        pass

    def _reset_aoi_info(self):
        self.current_analysis = None
        self.current_aoi_info = {}

    def _safe_maximize(self):
        self.minimized = False

    def closeEvent(self, event):
        event.accept()

    def change_lighting(self, *_args, **_kwargs):
        pass


class FakePresenter:
    def __init__(self, panel=None):
        self.panel = panel

    def sync(self, force=False):
        return force

    @staticmethod
    def _set_enabled(button, enabled):
        button.setEnabled(enabled)

    @staticmethod
    def _set_text(button, text):
        button.setText(text)


# Mesma ordem usada atualmente no main.py.
install_network_image_cycle_gate(FakePanel, FakePresenter)
install_local_capture_safety(FakePanel)


class LocalCaptureNetworkTakeoverTests(unittest.TestCase):
    def test_incomplete_network_cycle_can_be_replaced_by_local_mss(self):
        panel = FakePanel()

        self.assertTrue(panel.handle_network_image(object(), "192.168.0.10"))
        self.assertTrue(panel.capture_cycle_active)
        self.assertEqual(panel.capture_cycle_source, "network")
        self.assertTrue(panel.is_locked)
        self.assertIsNone(panel.current_analysis)

        self.assertTrue(panel.start_monitoring())

        self.assertEqual(panel.start_calls, 1)
        self.assertTrue(panel.minimized)
        self.assertTrue(panel.local_capture_pending)
        self.assertEqual(panel.capture_cycle_source, "local")
        self.assertTrue(panel.capture_cycle_active)
        self.assertTrue(panel.network_receiver.locked)

        self.assertTrue(panel._start_radar())
        self.assertEqual(len(panel.created_monitors), 1)
        self.assertEqual(panel.created_monitors[0].start_count, 1)

    def test_network_watchdog_releases_invalid_frame(self):
        panel = FakePanel()
        self.assertTrue(panel.handle_network_image(object(), "192.168.0.10"))
        self.assertEqual(len(panel.network_scheduled), 1)

        _delay, callback = panel.network_scheduled[0]
        callback()

        self.assertFalse(panel.capture_cycle_active)
        self.assertFalse(panel.network_receiver.locked)
        self.assertFalse(panel.is_locked)
        self.assertIsNone(panel.current_analysis)

    def test_valid_network_analysis_is_not_released_by_watchdog(self):
        panel = FakePanel()
        self.assertTrue(panel.handle_network_image(object(), "192.168.0.10"))
        panel.current_sample = object()
        panel.current_ng = object()
        panel.current_analysis = {"confidence": 0.8}

        _delay, callback = panel.network_scheduled[0]
        callback()

        self.assertTrue(panel.capture_cycle_active)
        self.assertTrue(panel.network_receiver.locked)
        self.assertTrue(panel.is_locked)


if __name__ == "__main__":
    unittest.main()
