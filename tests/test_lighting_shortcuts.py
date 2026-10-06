import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QPushButton, QWidget

from src.ui.lighting_shortcuts import (
    _activate_lighting,
    install_lighting_shortcuts,
)


class LightingShortcutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _panel():
        panel = QWidget()
        panel.btn_light_top = QPushButton("TOP", panel)
        panel.btn_light_side = QPushButton("SIDE", panel)
        panel.btn_light_mid = QPushButton("MID", panel)
        panel.calls = []

        def change_lighting(mode, source):
            panel.calls.append((mode, source))
            return True

        panel.change_lighting = change_lighting
        return panel

    def test_left_routes_to_top_using_keyboard_source(self):
        panel = self._panel()

        self.assertTrue(
            _activate_lighting(panel, "TOP", "btn_light_top")
        )
        self.assertEqual(panel.calls, [("TOP", "odin_keyboard")])

    def test_disabled_lighting_control_is_not_bypassed(self):
        panel = self._panel()
        panel.btn_light_top.setEnabled(False)

        self.assertFalse(
            _activate_lighting(panel, "TOP", "btn_light_top")
        )
        self.assertEqual(panel.calls, [])

    def test_window_shortcuts_cover_left_down_right(self):
        panel = self._panel()
        install_lighting_shortcuts(panel)

        self.assertTrue(panel._lighting_shortcuts_installed)
        self.assertEqual(len(panel._lighting_shortcuts), 3)

        sequences = {
            shortcut.key().toString()
            for shortcut in panel._lighting_shortcuts
        }
        self.assertIn("Left", sequences)
        self.assertIn("Down", sequences)
        self.assertIn("Right", sequences)
        self.assertTrue(
            all(not shortcut.autoRepeat() for shortcut in panel._lighting_shortcuts)
        )


class LightingShortcutSourceContractTests(unittest.TestCase):
    def test_main_installs_shortcuts_after_panel_creation(self):
        source = open("main.py", encoding="utf-8").read()

        panel = source.index("panel = ControlPanel()")
        shortcuts = source.index("install_lighting_shortcuts(panel)")
        self.assertLess(panel, shortcuts)

    def test_control_panel_does_not_depend_on_arrow_keypress_event(self):
        source = open("src/ui/control_panel.py", encoding="utf-8").read()

        self.assertNotIn("Qt.Key.Key_Left:", source)
        self.assertNotIn("Qt.Key.Key_Down:", source)
        self.assertNotIn("Qt.Key.Key_Right:", source)


if __name__ == "__main__":
    unittest.main()
