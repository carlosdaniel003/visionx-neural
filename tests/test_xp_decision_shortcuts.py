import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QPushButton, QWidget

from src.ui.xp_decision_shortcuts import (
    _activate_decision,
    install_xp_decision_shortcuts,
)


class XPDecisionShortcutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _panel():
        panel = QWidget()
        panel.btn_save_ok = QPushButton("OK", panel)
        panel.btn_save_ng = QPushButton("NG", panel)
        panel.ok_count = 0
        panel.ng_count = 0
        panel.feedback = []
        panel.show_decision_key_feedback = (
            lambda decision, source="": panel.feedback.append(
                (decision, source)
            )
        )
        panel.btn_save_ok.clicked.connect(
            lambda _checked=False: setattr(panel, "ok_count", panel.ok_count + 1)
        )
        panel.btn_save_ng.clicked.connect(
            lambda _checked=False: setattr(panel, "ng_count", panel.ng_count + 1)
        )
        return panel

    def test_zero_uses_same_ok_button_path(self):
        panel = self._panel()
        self.assertTrue(_activate_decision(panel, "OK"))
        self.assertEqual(panel.ok_count, 1)
        self.assertEqual(panel.ng_count, 0)
        self.assertEqual(panel.feedback, [("OK", "odin_keyboard")])

    def test_one_uses_same_ng_button_path(self):
        panel = self._panel()
        self.assertTrue(_activate_decision(panel, "NG"))
        self.assertEqual(panel.ok_count, 0)
        self.assertEqual(panel.ng_count, 1)
        self.assertEqual(panel.feedback, [("NG", "odin_keyboard")])

    def test_disabled_decision_is_not_bypassed(self):
        panel = self._panel()
        panel.btn_save_ok.setEnabled(False)
        self.assertFalse(_activate_decision(panel, "OK"))
        self.assertEqual(panel.ok_count, 0)
        self.assertEqual(panel.feedback, [])

    def test_window_shortcuts_include_main_and_numeric_keypad(self):
        panel = self._panel()
        install_xp_decision_shortcuts(panel)

        self.assertTrue(panel._xp_decision_shortcuts_installed)
        self.assertEqual(len(panel._xp_decision_shortcuts), 4)
        sequences = {
            shortcut.key().toString()
            for shortcut in panel._xp_decision_shortcuts
        }
        self.assertIn("0", sequences)
        self.assertIn("1", sequences)
        self.assertTrue(any("Num" in sequence and "0" in sequence for sequence in sequences))
        self.assertTrue(any("Num" in sequence and "1" in sequence for sequence in sequences))


class XPCommandBridgeSourceTests(unittest.TestCase):
    def test_ok_and_ng_buttons_still_route_to_press_commands(self):
        source = open("src/services/anomaly_learning.py", encoding="utf-8").read()
        self.assertIn(
            'self.send_command_to_xp("0" if normalized == "OK" else "1")',
            source,
        )

    def test_command_sender_uses_xp_port_and_reports_failures(self):
        source = open("src/ui/control_panel.py", encoding="utf-8").read()
        self.assertIn('command = f"PRESS_{str(tecla).strip().upper()}"', source)
        self.assertIn("s.connect((self.last_xp_ip, 5000))", source)
        self.assertIn('s.sendall(command.encode("utf-8"))', source)
        self.assertIn("Falha ao enviar {command}", source)


if __name__ == "__main__":
    unittest.main()
