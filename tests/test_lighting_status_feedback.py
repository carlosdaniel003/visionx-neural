import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

from src.ui.lighting_status_feedback import (
    LIGHTING_STATUS_MARGIN,
    LIGHTING_STATUS_TOP_OFFSET,
    LIGHTING_STATUS_WIDTH,
    install_lighting_status_feedback,
    install_lighting_status_feedback_hooks,
)


class LightingStatusOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _panel(self):
        panel = QWidget()
        panel.resize(1000, 700)
        panel.lbl_light_value = QLabel("SIDE", panel)
        install_lighting_status_feedback(panel)
        return panel

    def test_persistent_card_starts_in_side(self):
        panel = self._panel()
        overlay = panel.lighting_status_feedback

        self.assertEqual(overlay.current_mode, "SIDE")
        self.assertEqual(overlay.value_label.text(), "SIDE")
        self.assertEqual(overlay.key_label.text(), "↓")
        self.assertFalse(overlay.isHidden())
        self.assertTrue(
            overlay.testAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents
            )
        )
        self.assertEqual(overlay.focusPolicy(), Qt.FocusPolicy.NoFocus)
        self.assertFalse(hasattr(overlay, "_hide_timer"))

    def test_card_maps_top_side_mid_to_correct_arrows(self):
        panel = self._panel()
        overlay = panel.lighting_status_feedback

        self.assertTrue(overlay.set_lighting("TOP"))
        self.assertEqual(overlay.key_label.text(), "←")

        self.assertTrue(overlay.set_lighting("SIDE"))
        self.assertEqual(overlay.key_label.text(), "↓")

        self.assertTrue(overlay.set_lighting("MID"))
        self.assertEqual(overlay.key_label.text(), "→")

    def test_card_is_anchored_below_verdict_area(self):
        panel = self._panel()
        overlay = panel.lighting_status_feedback
        target = overlay._top_right_position()

        self.assertEqual(
            target.x(),
            panel.width() - LIGHTING_STATUS_WIDTH - LIGHTING_STATUS_MARGIN,
        )
        self.assertEqual(target.y(), LIGHTING_STATUS_TOP_OFFSET)


class LightingStatusHookTests(unittest.TestCase):
    def test_odin_keyboard_updates_status_and_shows_actual_xp_key(self):
        class FakePanel:
            def __init__(self):
                self.events = []

            def change_lighting(self, mode, source):
                self.events.append(("change", mode, source))
                return True

            def update_lighting_status_feedback(self, mode):
                self.events.append(("status", mode))
                return True

            def show_operational_key_feedback(self, key, source=""):
                self.events.append(("key", key, source))
                return True

        install_lighting_status_feedback_hooks(FakePanel)
        panel = FakePanel()

        result = panel.change_lighting("TOP", "odin_keyboard")

        self.assertTrue(result)
        self.assertEqual(
            panel.events,
            [
                ("change", "TOP", "odin_keyboard"),
                ("status", "TOP"),
                ("key", "LEFT", "odin_keyboard"),
            ],
        )

    def test_failed_xp_command_does_not_change_visual_state(self):
        class FakePanel:
            def __init__(self):
                self.events = []

            def change_lighting(self, mode, source):
                self.events.append(("change", mode, source))
                return False

            def update_lighting_status_feedback(self, mode):
                self.events.append(("status", mode))

            def show_operational_key_feedback(self, key, source=""):
                self.events.append(("key", key, source))

        install_lighting_status_feedback_hooks(FakePanel)
        panel = FakePanel()

        result = panel.change_lighting("MID", "odin_control")

        self.assertFalse(result)
        self.assertEqual(
            panel.events,
            [("change", "MID", "odin_control")],
        )


if __name__ == "__main__":
    unittest.main()
