import unittest

from src.ui.local_capture_safety import install_local_capture_safety
from src.ui.mode_selector_gate import install_mode_selector_gate
from src.ui.network_image_cycle_gate import install_network_image_cycle_gate
from src.ui.operational_controls_model import operational_state


class FakeSignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in list(self.callbacks):
            callback(*args)


class FakeMonitor:
    def __init__(self):
        self.layout_detected = FakeSignal()
        self.log_updated = FakeSignal()
        self.finished = FakeSignal()
        self.running = False
        self.start_count = 0
        self.stop_count = 0

    def start(self):
        self.running = True
        self.start_count += 1

    def stop(self):
        self.running = False
        self.stop_count += 1

    def isRunning(self):
        return self.running

    def emit_layout(self):
        self.layout_detected.emit(object(), object(), {"value": "FALTANDO"})
        self.running = False
        self.finished.emit()


class FakeButton:
    def __init__(self):
        self.enabled = True
        self.visible = True
        self.text_value = ""

    def setEnabled(self, value):
        self.enabled = bool(value)

    def isEnabled(self):
        return self.enabled

    def setVisible(self, value):
        self.visible = bool(value)

    def setText(self, value):
        self.text_value = str(value)

    def text(self):
        return self.text_value


class FakeLabel:
    def __init__(self):
        self.value = ""

    def setText(self, value):
        self.value = str(value)

    def text(self):
        return self.value

    def setProperty(self, *_args):
        pass


class FakeCombo:
    def __init__(self, value="Modo Teste"):
        self.value = value
        self.enabled = True
        self.cursor = None
        self.tooltip = ""

    def currentText(self):
        return self.value

    def setCurrentText(self, value):
        if self.enabled:
            self.value = str(value)

    def setEnabled(self, value):
        self.enabled = bool(value)

    def setCursor(self, value):
        self.cursor = value

    def setToolTip(self, value):
        self.tooltip = str(value)


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


class FakePanel:
    def __init__(self):
        self.combo_mode = FakeCombo()
        self.network_receiver = FakeReceiver()
        self.monitor = None
        self.current_analysis = None
        self.current_sample = None
        self.current_ng = None
        self.current_aoi_info = {}
        self.is_locked = False
        self.production_review_pending = False
        self.minimized = False
        self.closed = False
        self.start_calls = 0
        self.skip_calls = 0
        self.process_calls = 0
        self.save_calls = []
        self.light_calls = []
        self.status_messages = []
        self.scheduled = []
        self.created_monitors = []
        self.raise_in_start = False
        self.raise_in_process = False
        self.raise_in_lighting = False

        self.btn_start = FakeButton()
        self.btn_skip = FakeButton()
        self.btn_save_ok = FakeButton()
        self.btn_save_ng = FakeButton()
        self.btn_light_mid = FakeButton()
        self.btn_light_side = FakeButton()
        self.btn_light_top = FakeButton()
        self.lbl_operation_hint = FakeLabel()
        self.lbl_operation_state = FakeLabel()

        self._local_capture_scheduler = (
            lambda delay, callback: self.scheduled.append((delay, callback))
        )

        def factory():
            monitor = FakeMonitor()
            self.created_monitors.append(monitor)
            return monitor

        self._screen_monitor_factory = factory

    def start_monitoring(self):
        self.start_calls += 1
        if self.raise_in_start:
            raise RuntimeError("falha simulada ao iniciar MSS")
        self.is_locked = True
        self.minimized = True

    def _start_radar(self):
        raise AssertionError("A implementação segura deve substituir este método")

    def process_aoi_images(self, *_args):
        self.process_calls += 1
        if self.raise_in_process:
            raise RuntimeError("falha simulada na análise")
        self.current_analysis = {"confidence": 0.80}
        self.current_sample = object()
        self.current_ng = object()
        self.is_locked = True
        self.minimized = False

    def handle_network_image(self, *_args):
        self.current_analysis = None
        self.is_locked = True

    def skip_image(self):
        self.skip_calls += 1
        self.current_analysis = None
        self.current_sample = None
        self.current_ng = None
        self.current_aoi_info = {}
        self.is_locked = False

    def save_label(self, decision, source="button"):
        self.save_calls.append((decision, source))
        self.current_analysis = None
        self.current_sample = None
        self.current_ng = None
        self.is_locked = False

    def change_lighting(self, mode, source="local"):
        if self.raise_in_lighting:
            raise RuntimeError("falha simulada na iluminação")
        self.light_calls.append((mode, source))

    def update_brain_status(self, message, _active=False):
        self.status_messages.append(str(message))

    def update_network_status(self, message):
        self.status_messages.append(str(message))

    def _safe_maximize(self):
        self.minimized = False

    def _reset_confidence_panel(self):
        pass

    def _reset_reference_panel(self):
        pass

    def _reset_aoi_info(self):
        self.current_analysis = None
        self.current_aoi_info = {}

    def closeEvent(self, event):
        self.closed = True
        event.accept()


class FakePresenter:
    def __init__(self, panel=None):
        self.panel = panel

    @staticmethod
    def _set_enabled(button, enabled):
        button.setEnabled(enabled)

    @staticmethod
    def _set_text(button, text):
        button.setText(text)

    def sync(self, force=False):
        return force


class FakeCloseEvent:
    def __init__(self):
        self.accepted = False

    def accept(self):
        self.accepted = True


install_network_image_cycle_gate(FakePanel, FakePresenter)
install_local_capture_safety(FakePanel)
install_mode_selector_gate(FakePresenter)


class ButtonStateMatrixTests(unittest.TestCase):
    def test_idle_state_enables_capture_and_lighting(self):
        for mode in ("Modo Teste", "Modo Sombra", "Modo Produção"):
            with self.subTest(mode=mode):
                state = operational_state(
                    mode=mode,
                    is_locked=False,
                    has_analysis=False,
                )
                self.assertTrue(state.capture_enabled)
                self.assertTrue(state.lighting_enabled)
                self.assertFalse(state.discard_enabled)
                self.assertFalse(state.approve_enabled)
                self.assertFalse(state.reject_enabled)

    def test_processing_state_disables_every_action(self):
        state = operational_state(
            mode="Modo Teste",
            is_locked=True,
            has_analysis=False,
        )
        self.assertFalse(state.capture_enabled)
        self.assertFalse(state.discard_enabled)
        self.assertFalse(state.approve_enabled)
        self.assertFalse(state.reject_enabled)
        self.assertFalse(state.lighting_enabled)
        self.assertFalse(state.dataset_clear_enabled)

    def test_test_review_exposes_all_resolution_paths(self):
        state = operational_state(
            mode="Modo Teste",
            is_locked=True,
            has_analysis=True,
        )
        self.assertTrue(state.capture_enabled)
        self.assertTrue(state.discard_enabled)
        self.assertTrue(state.approve_enabled)
        self.assertTrue(state.reject_enabled)
        self.assertTrue(state.lighting_enabled)


class OperationalSequenceTests(unittest.TestCase):
    def _start_local_and_emit_analysis(self, panel):
        self.assertTrue(panel.start_monitoring())
        self.assertTrue(panel.minimized)
        self.assertTrue(panel.local_capture_pending)
        self.assertTrue(panel.capture_cycle_active)
        self.assertTrue(panel.network_receiver.locked)

        self.assertTrue(panel._start_radar())
        monitor = panel.created_monitors[-1]
        self.assertTrue(monitor.running)
        monitor.emit_layout()

        self.assertEqual(panel.process_calls, 1)
        self.assertFalse(panel.local_capture_pending)
        self.assertFalse(panel.minimized)
        self.assertIsNotNone(panel.current_analysis)
        self.assertTrue(panel.is_locked)
        return monitor

    def test_local_capture_still_minimizes_and_starts_mss(self):
        panel = FakePanel()
        monitor = self._start_local_and_emit_analysis(panel)
        self.assertEqual(panel.start_calls, 1)
        self.assertEqual(monitor.start_count, 1)

    def test_ok_then_new_capture_does_not_stick(self):
        panel = FakePanel()
        self._start_local_and_emit_analysis(panel)
        panel.save_label("OK", source="button")
        self.assertFalse(panel.capture_cycle_active)
        self.assertFalse(panel.network_receiver.locked)
        self.assertFalse(panel.is_locked)
        self.assertTrue(panel.start_monitoring())

    def test_ng_then_new_capture_does_not_stick(self):
        panel = FakePanel()
        self._start_local_and_emit_analysis(panel)
        panel.save_label("NG", source="button")
        self.assertFalse(panel.capture_cycle_active)
        self.assertFalse(panel.is_locked)
        self.assertTrue(panel.start_monitoring())

    def test_discard_then_new_capture_does_not_stick(self):
        panel = FakePanel()
        self._start_local_and_emit_analysis(panel)
        panel.skip_image()
        self.assertFalse(panel.capture_cycle_active)
        self.assertFalse(panel.network_receiver.locked)
        self.assertFalse(panel.is_locked)
        self.assertTrue(panel.start_monitoring())

    def test_capture_new_piece_discards_review_and_launches_mss_atomically(self):
        panel = FakePanel()
        panel.current_analysis = {"confidence": 0.70}
        panel.current_sample = object()
        panel.current_ng = object()
        panel.is_locked = True
        panel.capture_cycle_active = True
        panel.network_receiver.locked = True

        self.assertTrue(panel.start_monitoring())
        self.assertEqual(panel.skip_calls, 1)
        self.assertEqual(panel.start_calls, 1)
        self.assertTrue(panel.local_capture_pending)
        self.assertTrue(panel.capture_cycle_active)
        self.assertTrue(panel.network_receiver.locked)
        self.assertEqual(panel.network_receiver.release_count, 0)

    def test_capture_click_is_ignored_while_processing(self):
        panel = FakePanel()
        panel.current_analysis = None
        panel.is_locked = True
        panel.capture_cycle_active = True
        panel.network_receiver.locked = True
        self.assertFalse(panel.start_monitoring())
        self.assertEqual(panel.start_calls, 0)
        self.assertFalse(panel.local_capture_pending)

    def test_capture_is_blocked_during_production_review(self):
        panel = FakePanel()
        panel.combo_mode.setCurrentText("Modo Produção")
        panel.current_analysis = {"confidence": 0.70}
        panel.production_review_pending = True
        panel.is_locked = True
        panel.capture_cycle_active = True
        self.assertFalse(panel.start_monitoring())
        self.assertEqual(panel.skip_calls, 0)
        self.assertEqual(panel.start_calls, 0)

    def test_timeout_restores_window_and_unlocks_system(self):
        panel = FakePanel()
        self.assertTrue(panel.start_monitoring())
        timeout_callbacks = [
            callback for delay, callback in panel.scheduled if delay == 20_000
        ]
        self.assertEqual(len(timeout_callbacks), 1)
        timeout_callbacks[0]()
        self.assertFalse(panel.local_capture_pending)
        self.assertFalse(panel.capture_cycle_active)
        self.assertFalse(panel.network_receiver.locked)
        self.assertFalse(panel.is_locked)
        self.assertFalse(panel.minimized)

    def test_stale_monitor_signal_after_timeout_is_ignored(self):
        panel = FakePanel()
        self.assertTrue(panel.start_monitoring())
        self.assertTrue(panel._start_radar())
        monitor = panel.created_monitors[-1]
        timeout_callback = next(
            callback for delay, callback in panel.scheduled if delay == 20_000
        )
        timeout_callback()
        monitor.emit_layout()
        self.assertEqual(panel.process_calls, 0)
        self.assertFalse(panel.capture_cycle_active)

    def test_start_exception_restores_interface(self):
        panel = FakePanel()
        panel.raise_in_start = True
        self.assertFalse(panel.start_monitoring())
        self.assertFalse(panel.capture_cycle_active)
        self.assertFalse(panel.is_locked)
        self.assertFalse(panel.minimized)
        self.assertTrue(panel.status_messages)

    def test_analysis_exception_restores_interface(self):
        panel = FakePanel()
        panel.raise_in_process = True
        self.assertTrue(panel.start_monitoring())
        self.assertTrue(panel._start_radar())
        panel.created_monitors[-1].emit_layout()
        self.assertEqual(panel.process_calls, 1)
        self.assertFalse(panel.capture_cycle_active)
        self.assertFalse(panel.is_locked)
        self.assertFalse(panel.minimized)

    def test_complete_repeated_sequence_never_leaves_gate_stuck(self):
        panel = FakePanel()
        resolutions = (
            ("OK", "save"),
            ("NG", "save"),
            (None, "discard"),
            ("OK", "save"),
        )
        for decision, action in resolutions:
            self._start_local_and_emit_analysis(panel)
            if action == "discard":
                panel.skip_image()
            else:
                panel.save_label(decision, source="button")
            self.assertFalse(panel.capture_cycle_active)
            self.assertFalse(panel.network_receiver.locked)
            self.assertFalse(panel.is_locked)

    def test_lighting_buttons_do_not_change_capture_lock(self):
        panel = FakePanel()
        for light in ("MID", "SIDE", "TOP"):
            panel.change_lighting(light, source="local")
        self.assertEqual(
            panel.light_calls,
            [("MID", "local"), ("SIDE", "local"), ("TOP", "local")],
        )
        self.assertFalse(panel.capture_cycle_active)
        self.assertFalse(panel.is_locked)

    def test_lighting_exception_is_non_fatal(self):
        panel = FakePanel()
        panel.raise_in_lighting = True
        panel.change_lighting("MID", source="local")
        self.assertFalse(panel.capture_cycle_active)
        self.assertFalse(panel.is_locked)
        self.assertTrue(panel.status_messages)

    def test_mode_selector_is_locked_during_cycle_and_released_after_decision(self):
        panel = FakePanel()
        presenter = FakePresenter(panel)
        presenter.sync(force=True)
        self.assertTrue(panel.combo_mode.enabled)

        self._start_local_and_emit_analysis(panel)
        presenter.sync(force=True)
        self.assertFalse(panel.combo_mode.enabled)
        previous_mode = panel.combo_mode.currentText()
        panel.combo_mode.setCurrentText("Modo Produção")
        self.assertEqual(panel.combo_mode.currentText(), previous_mode)

        panel.save_label("OK", source="button")
        presenter.sync(force=True)
        self.assertTrue(panel.combo_mode.enabled)
        panel.combo_mode.setCurrentText("Modo Produção")
        self.assertEqual(panel.combo_mode.currentText(), "Modo Produção")

    def test_close_stops_running_local_monitor(self):
        panel = FakePanel()
        self.assertTrue(panel.start_monitoring())
        self.assertTrue(panel._start_radar())
        monitor = panel.created_monitors[-1]
        event = FakeCloseEvent()
        panel.closeEvent(event)
        self.assertEqual(monitor.stop_count, 1)
        self.assertTrue(event.accepted)
        self.assertTrue(panel.closed)


class SourceContractTests(unittest.TestCase):
    def test_main_installs_wrappers_in_safe_order(self):
        source = open("main.py", encoding="utf-8").read()
        cycle = source.index("install_network_image_cycle_gate(")
        local = source.index("install_local_capture_safety(")
        mode = source.index("install_mode_selector_gate(")
        self.assertLess(cycle, local)
        self.assertLess(local, mode)

    def test_dataset_button_has_exception_recovery(self):
        source = open(
            "src/ui/test_mode_dataset_controls.py", encoding="utf-8"
        ).read()
        self.assertIn("except Exception as exc", source)
        self.assertIn("finally:", source)
        self.assertIn("O sistema permanece ativo", source)

    def test_idle_button_copy_names_mss_explicitly(self):
        source = open("src/ui/capture_button_copy.py", encoding="utf-8").read()
        self.assertIn("Capturar local (MSS)", source)
        self.assertIn("Minimiza o VisionX", source)


if __name__ == "__main__":
    unittest.main()
