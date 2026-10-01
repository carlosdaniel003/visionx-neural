import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget

from src.ui.decision_key_feedback import (
    FEEDBACK_DURATION_MS,
    FEEDBACK_MARGIN,
    FEEDBACK_SIZE,
    install_decision_key_feedback,
    install_decision_key_feedback_hooks,
)


class DecisionKeyFeedbackOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _panel(self):
        panel = QWidget()
        panel.resize(1000, 700)
        install_decision_key_feedback(panel)
        return panel

    def test_ok_feedback_is_green_zero_and_click_through(self):
        panel = self._panel()

        shown = panel.show_decision_key_feedback(
            "OK",
            source="odin_keyboard",
        )

        overlay = panel.decision_key_feedback
        self.assertTrue(shown)
        self.assertEqual(overlay.width(), FEEDBACK_SIZE)
        self.assertEqual(overlay.height(), FEEDBACK_SIZE)
        self.assertEqual(overlay.digit_label.text(), "0")
        self.assertEqual(overlay.decision_label.text(), "OK")
        self.assertEqual(overlay.source_label.text(), "TECLADO ODIN")
        self.assertEqual(overlay.property("tone"), "ok")
        self.assertTrue(
            overlay.testAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents
            )
        )
        self.assertEqual(overlay.focusPolicy(), Qt.FocusPolicy.NoFocus)
        self.assertEqual(
            overlay.x(),
            panel.width() - FEEDBACK_SIZE - FEEDBACK_MARGIN,
        )
        self.assertEqual(
            overlay.y(),
            panel.height() - FEEDBACK_SIZE - FEEDBACK_MARGIN,
        )
        self.assertTrue(overlay._hide_timer.isActive())
        self.assertEqual(overlay._hide_timer.interval(), FEEDBACK_DURATION_MS)

    def test_overlay_uses_odin_visual_language(self):
        panel = self._panel()
        overlay = panel.decision_key_feedback

        source = open(
            "src/ui/decision_key_feedback.py",
            encoding="utf-8",
        ).read()

        self.assertEqual(overlay.header_label.text(), "DECISÃO RECEBIDA")
        self.assertIn("#101010", source)
        self.assertIn("#303030", source)
        self.assertIn("#f5c518", source)
        self.assertIn("#4ade80", source)
        self.assertIn("#ff6262", source)
        self.assertIn("_position_bottom_right", source)

    def test_ng_feedback_is_red_one(self):
        panel = self._panel()

        shown = panel.show_decision_key_feedback(
            "NG",
            source="xp_keyboard",
        )

        overlay = panel.decision_key_feedback
        self.assertTrue(shown)
        self.assertEqual(overlay.digit_label.text(), "1")
        self.assertEqual(overlay.decision_label.text(), "NG")
        self.assertEqual(
            overlay.source_label.text(),
            "TECLADO WINDOWS XP",
        )
        self.assertEqual(overlay.property("tone"), "ng")

    def test_same_decision_echo_is_suppressed(self):
        panel = self._panel()

        first = panel.show_decision_key_feedback(
            "OK",
            source="odin_keyboard",
        )
        echo = panel.show_decision_key_feedback(
            "OK",
            source="xp_keyboard",
        )

        self.assertTrue(first)
        self.assertFalse(echo)
        self.assertEqual(
            panel.decision_key_feedback.source_label.text(),
            "TECLADO ODIN",
        )

    def test_different_decision_can_replace_visible_feedback(self):
        panel = self._panel()

        self.assertTrue(
            panel.show_decision_key_feedback(
                "OK",
                source="odin_keyboard",
            )
        )
        self.assertTrue(
            panel.show_decision_key_feedback(
                "NG",
                source="xp_keyboard",
            )
        )
        self.assertEqual(panel.decision_key_feedback.digit_label.text(), "1")


class DecisionKeyFeedbackXPBridgeTests(unittest.TestCase):
    def test_xp_ok_ng_commands_show_feedback_only_with_active_capture(self):
        class FakeControlPanel:
            def __init__(self):
                self.current_ng = object()
                self.calls = []
                self.feedback = []

            def handle_physical_keyboard(self, command):
                self.calls.append(command)
                if command in {"OK", "NG"}:
                    self.current_ng = None
                return command

            def show_decision_key_feedback(self, decision, source=""):
                self.feedback.append((decision, source))

        install_decision_key_feedback_hooks(FakeControlPanel)
        panel = FakeControlPanel()

        result = panel.handle_physical_keyboard("OK")

        self.assertEqual(result, "OK")
        self.assertEqual(panel.calls, ["OK"])
        self.assertEqual(panel.feedback, [("OK", "xp_keyboard")])

        panel.feedback.clear()
        panel.handle_physical_keyboard("NG")
        self.assertEqual(panel.feedback, [])

    def test_non_decision_xp_command_never_shows_feedback(self):
        class FakeControlPanel:
            def __init__(self):
                self.current_ng = object()
                self.feedback = []

            def handle_physical_keyboard(self, command):
                return command

            def show_decision_key_feedback(self, decision, source=""):
                self.feedback.append((decision, source))

        install_decision_key_feedback_hooks(FakeControlPanel)
        panel = FakeControlPanel()

        panel.handle_physical_keyboard("MID")

        self.assertEqual(panel.feedback, [])


class DecisionKeyFeedbackSourceContractTests(unittest.TestCase):
    def test_main_installs_hook_before_panel_and_overlay_before_shortcuts(self):
        source = open("main.py", encoding="utf-8").read()

        hook = source.index("install_decision_key_feedback_hooks(ControlPanel)")
        panel = source.index("panel = ControlPanel()")
        overlay = source.index("install_decision_key_feedback(panel)")
        shortcuts = source.index("install_xp_decision_shortcuts(panel)")

        self.assertLess(hook, panel)
        self.assertLess(panel, overlay)
        self.assertLess(overlay, shortcuts)


if __name__ == "__main__":
    unittest.main()
