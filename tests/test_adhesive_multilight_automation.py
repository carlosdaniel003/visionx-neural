import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QWidget

from src.ui.adhesive_multilight_automation import (
    AdhesiveMultiLightAutomation,
    FRAME_TIMEOUT_MS,
    MAX_FRAME_RETRIES,
)


class _FakeReceiver:
    def __init__(self):
        self.enabled = False
        self.history = []

    def set_auxiliary_image_mode(self, enabled):
        self.enabled = bool(enabled)
        self.history.append(self.enabled)


class _FakePanel(QWidget):
    def __init__(self):
        super().__init__()
        self.last_xp_ip = "169.254.95.200"
        self.network_receiver = _FakeReceiver()
        self.commands = []
        self.visual_modes = []
        self.brain_status = []
        self.network_status = []
        self.adhesive_multilight_automation_active = False

    def send_command_to_xp(self, command):
        self.commands.append(str(command))
        return True

    def change_lighting(self, mode, source):
        self.visual_modes.append((str(mode), str(source)))
        return True

    def update_brain_status(self, message, active=False):
        self.brain_status.append((str(message), bool(active)))

    def update_network_status(self, message):
        self.network_status.append(str(message))


class AdhesiveMultiLightAutomationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel = _FakePanel()
        self.automation = AdhesiveMultiLightAutomation(self.panel)

    def _flush_events(self):
        self.app.processEvents()

    def test_full_sequence_is_side_top_mid_then_restore_side(self):
        self.assertTrue(self.automation.start())
        self.assertTrue(self.automation.active)
        self.assertEqual(self.automation.expected_mode, "TOP")
        self.assertEqual(self.automation.captured_modes, {"SIDE"})
        self.assertEqual(self.panel.commands, ["LEFT"])
        self.assertEqual(
            self.panel.visual_modes[-1],
            ("TOP", "adhesive_automation"),
        )
        self.assertTrue(self.panel.network_receiver.enabled)
        self.assertTrue(self.panel.adhesive_multilight_automation_active)

        self.assertTrue(self.automation.frame_stored("TOP"))
        self._flush_events()

        self.assertEqual(self.automation.expected_mode, "MID")
        self.assertEqual(self.panel.commands, ["LEFT", "RIGHT"])
        self.assertIn("TOP", self.automation.captured_modes)

        self.assertTrue(self.automation.frame_stored("MID"))
        self._flush_events()

        self.assertEqual(
            self.panel.commands,
            ["LEFT", "RIGHT", "DOWN"],
        )
        self.assertEqual(
            self.panel.visual_modes[-1],
            ("SIDE", "adhesive_automation"),
        )
        self.assertFalse(self.automation.active)
        self.assertTrue(self.automation.completed)
        self.assertFalse(self.panel.network_receiver.enabled)
        self.assertFalse(self.panel.adhesive_multilight_automation_active)
        self.assertEqual(
            self.automation.captured_modes,
            {"SIDE", "TOP", "MID"},
        )

    def test_unexpected_frame_does_not_advance_state(self):
        self.assertTrue(self.automation.start())

        self.assertFalse(self.automation.frame_stored("MID"))
        self.assertEqual(self.automation.expected_mode, "TOP")
        self.assertEqual(self.panel.commands, ["LEFT"])

    def test_timeout_repeats_same_absolute_lighting_once(self):
        self.assertTrue(self.automation.start())
        self.assertEqual(FRAME_TIMEOUT_MS, 8000)
        self.assertEqual(MAX_FRAME_RETRIES, 1)

        self.automation._timeout.stop()
        self.automation._handle_timeout()

        self.assertTrue(self.automation.active)
        self.assertEqual(self.automation.expected_mode, "TOP")
        self.assertEqual(self.automation.retry_count, 1)
        self.assertEqual(self.panel.commands, ["LEFT", "LEFT"])

    def test_second_timeout_aborts_and_restores_side(self):
        self.assertTrue(self.automation.start())

        self.automation._timeout.stop()
        self.automation._handle_timeout()
        self.automation._timeout.stop()
        self.automation._handle_timeout()

        self.assertFalse(self.automation.active)
        self.assertFalse(self.automation.completed)
        self.assertEqual(
            self.panel.commands,
            ["LEFT", "LEFT", "DOWN"],
        )
        # Falha automática mantém o canal auxiliar disponível para fallback
        # manual da mesma peça.
        self.assertTrue(self.panel.network_receiver.enabled)

    def test_cycle_end_mid_sequence_restores_side_and_closes_aux_mode(self):
        self.assertTrue(self.automation.start())
        self.assertEqual(self.panel.commands, ["LEFT"])

        self.automation.cancel_for_cycle_end()

        self.assertFalse(self.automation.active)
        self.assertFalse(self.panel.network_receiver.enabled)
        self.assertEqual(self.panel.commands, ["LEFT", "DOWN"])

    def test_missing_xp_ip_does_not_start(self):
        self.panel.last_xp_ip = ""

        self.assertFalse(self.automation.start())
        self.assertFalse(self.automation.active)
        self.assertEqual(self.panel.commands, [])


class AdhesiveAutomationSourceContractTests(unittest.TestCase):
    def test_inspection_uses_expected_automation_mode_for_aux_frame(self):
        source = open(
            "src/ui/adhesive_multilight_inspection.py",
            encoding="utf-8",
        ).read()

        self.assertIn("expected_frame_mode", source)
        self.assertIn("frame_stored(aux_mode)", source)
        self.assertIn("start_automation()", source)

    def test_manual_odin_lighting_is_blocked_during_sequence(self):
        source = open(
            "src/ui/control_panel.py",
            encoding="utf-8",
        ).read()

        self.assertIn(
            "adhesive_multilight_automation_active",
            source,
        )
        self.assertIn(
            "controle manual de iluminação temporariamente bloqueado",
            source,
        )

    def test_main_installs_automation_before_event_loop(self):
        source = open("main.py", encoding="utf-8").read()

        panel = source.index("panel = ControlPanel()")
        automation = source.index(
            "install_adhesive_multilight_automation(panel)"
        )
        event_loop = source.index("app.exec()")

        self.assertLess(panel, automation)
        self.assertLess(automation, event_loop)


if __name__ == "__main__":
    unittest.main()
