import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEasingCurve, Qt
from PyQt6.QtWidgets import QApplication, QWidget

from src.ui.decision_key_feedback import (
    FEEDBACK_DURATION_MS,
    FEEDBACK_FADE_IN_MS,
    FEEDBACK_FADE_OUT_MS,
    FEEDBACK_MARGIN,
    FEEDBACK_SIZE,
    FEEDBACK_SLIDE_PX,
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
        target = overlay._bottom_right_position()
        self.assertEqual(
            target.x(),
            panel.width() - FEEDBACK_SIZE - FEEDBACK_MARGIN,
        )
        self.assertEqual(
            target.y(),
            panel.height() - FEEDBACK_SIZE - FEEDBACK_MARGIN,
        )
        self.assertEqual(overlay._slide_in.endValue(), target)
        self.assertEqual(
            overlay._slide_in.startValue().y(),
            target.y() + FEEDBACK_SLIDE_PX,
        )
        self.assertTrue(overlay._hide_timer.isActive())
        self.assertEqual(
            overlay._hide_timer.interval(),
            FEEDBACK_DURATION_MS - FEEDBACK_FADE_OUT_MS,
        )

    def test_overlay_animation_is_short_non_looping_and_lightweight(self):
        panel = self._panel()
        overlay = panel.decision_key_feedback

        panel.show_decision_key_feedback(
            "OK",
            source="odin_keyboard",
        )

        self.assertEqual(overlay._fade_in.duration(), FEEDBACK_FADE_IN_MS)
        self.assertEqual(overlay._slide_in.duration(), FEEDBACK_FADE_IN_MS)
        self.assertEqual(overlay._fade_out.duration(), FEEDBACK_FADE_OUT_MS)
        self.assertEqual(
            overlay._fade_in.easingCurve().type(),
            QEasingCurve.Type.OutCubic,
        )
        self.assertEqual(
            overlay._fade_out.easingCurve().type(),
            QEasingCurve.Type.InOutQuad,
        )
        self.assertTrue(overlay._hide_timer.isSingleShot())
        self.assertLessEqual(
            FEEDBACK_FADE_IN_MS + FEEDBACK_FADE_OUT_MS,
            300,
        )

    def test_overlay_uses_odin_visual_language(self):
        panel = self._panel()
        overlay = panel.decision_key_feedback

        source = open(
            "src/ui/decision_key_feedback.py",
            encoding="utf-8",
        ).read()

        self.assertEqual(overlay.header_label.text(), "TECLA PRESSIONADA")
        self.assertIn("rgba(13, 13, 13, 248)", source)
        self.assertIn("border: 1px solid #f5c518", source)
        self.assertIn("#f5c518", source)
        self.assertIn("#4ade80", source)
        self.assertIn("#ff6262", source)
        self.assertIn("_bottom_right_position", source)

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

    def test_arrow_feedback_reuses_card_without_dismissing_verdict(self):
        panel = QWidget()
        panel.resize(1000, 700)
        events = []

        def start_verdict_fade():
            events.append("verdict_fade")
            return True

        panel.start_ai_verdict_feedback_fade_out = start_verdict_fade
        install_decision_key_feedback(panel)

        shown = panel.show_operational_key_feedback(
            "LEFT",
            source="odin_keyboard",
        )

        overlay = panel.decision_key_feedback
        self.assertTrue(shown)
        self.assertEqual(overlay.header_label.text(), "TECLA PRESSIONADA")
        self.assertEqual(overlay.digit_label.text(), "←")
        self.assertEqual(overlay.decision_label.text(), "ESQUERDA")
        self.assertEqual(overlay.source_label.text(), "TECLADO ODIN")
        self.assertEqual(overlay.property("tone"), "light")

        overlay._start_fade_out()
        self.assertEqual(events, [])

    def test_lighting_button_feedback_identifies_sent_key(self):
        panel = self._panel()

        shown = panel.show_operational_key_feedback(
            "RIGHT",
            source="odin_control",
        )

        overlay = panel.decision_key_feedback
        self.assertTrue(shown)
        self.assertEqual(overlay.header_label.text(), "TECLA ENVIADA")
        self.assertEqual(overlay.digit_label.text(), "→")
        self.assertEqual(overlay.decision_label.text(), "DIREITA")
        self.assertEqual(overlay.source_label.text(), "CONTROLE ODIN")

    def test_key_feedback_prepares_and_starts_verdict_fade_in_same_exit_event(self):
        panel = QWidget()
        panel.resize(1000, 700)
        events = []

        def prepare_verdict():
            events.append("prepare_verdict")
            return True

        def prepare_lighting():
            events.append("prepare_lighting")
            return True

        def start_verdict_fade():
            events.append("verdict_fade")
            return True

        def start_lighting_fade():
            events.append("lighting_fade")
            return True

        panel.prepare_ai_verdict_feedback_dismissal = prepare_verdict
        panel.prepare_lighting_status_feedback_dismissal = prepare_lighting
        panel.start_ai_verdict_feedback_fade_out = start_verdict_fade
        panel.start_lighting_status_feedback_fade_out = start_lighting_fade
        install_decision_key_feedback(panel)

        shown = panel.show_decision_key_feedback(
            "OK",
            source="odin_keyboard",
        )
        self.assertTrue(shown)
        self.assertEqual(
            events,
            ["prepare_verdict", "prepare_lighting"],
        )

        panel.decision_key_feedback._start_fade_out()

        self.assertEqual(
            events,
            [
                "prepare_verdict",
                "prepare_lighting",
                "verdict_fade",
                "lighting_fade",
            ],
        )
        self.assertEqual(
            panel.decision_key_feedback._fade_out.duration(),
            FEEDBACK_FADE_OUT_MS,
        )


class DecisionKeyFeedbackXPBridgeTests(unittest.TestCase):
    def test_xp_feedback_is_prepared_before_productive_handler_resets_cycle(self):
        class FakeControlPanel:
            def __init__(self):
                self.current_ng = object()
                self.events = []

            def handle_physical_keyboard(self, command):
                self.events.append(("handler", command))
                self.current_ng = None
                return command

            def show_decision_key_feedback(self, decision, source=""):
                self.events.append(("feedback", decision, source))
                return True

        install_decision_key_feedback_hooks(FakeControlPanel)
        panel = FakeControlPanel()

        panel.handle_physical_keyboard("OK")

        self.assertEqual(
            panel.events,
            [
                ("feedback", "OK", "xp_keyboard"),
                ("handler", "OK"),
            ],
        )

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
