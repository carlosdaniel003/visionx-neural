import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEasingCurve, Qt
from PyQt6.QtWidgets import QApplication, QWidget

from src.ui.decision_verdict_feedback import (
    VERDICT_FEEDBACK_DURATION_MS,
    VERDICT_FEEDBACK_FADE_IN_MS,
    VERDICT_FEEDBACK_FADE_OUT_MS,
    VERDICT_FEEDBACK_HEIGHT,
    VERDICT_FEEDBACK_MARGIN,
    VERDICT_FEEDBACK_SLIDE_PX,
    VERDICT_FEEDBACK_TOP_OFFSET,
    VERDICT_FEEDBACK_WIDTH,
    install_ai_verdict_feedback,
    install_ai_verdict_feedback_hooks,
    verdict_feedback_state,
)


class AIVerdictFeedbackStateTests(unittest.TestCase):
    def test_maps_only_final_supported_verdicts(self):
        self.assertEqual(
            verdict_feedback_state({"verdict": "FALHA FALSA"}),
            ("FALHA FALSA", "ok"),
        )
        self.assertEqual(
            verdict_feedback_state({"verdict": "DEFEITO REAL"}),
            ("DEFEITO REAL", "ng"),
        )
        self.assertEqual(
            verdict_feedback_state({"verdict": "DEFEITO"}),
            ("DEFEITO REAL", "ng"),
        )

    def test_review_state_does_not_invent_binary_verdict(self):
        analysis = {
            "is_defect": False,
            "verdict": "REVISÃO OBRIGATÓRIA",
            "detail": {
                "decision_trace": {
                    "operator_review_required": True,
                }
            },
        }
        self.assertEqual(verdict_feedback_state(analysis), ("", ""))

    def test_legacy_analysis_without_verdict_can_still_map_when_final(self):
        self.assertEqual(
            verdict_feedback_state({"is_defect": False}),
            ("FALHA FALSA", "ok"),
        )
        self.assertEqual(
            verdict_feedback_state({"is_defect": True}),
            ("DEFEITO REAL", "ng"),
        )


class AIVerdictFeedbackOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _panel(self):
        panel = QWidget()
        panel.resize(1200, 800)
        install_ai_verdict_feedback(panel)
        return panel

    def test_false_failure_uses_top_right_green_card_without_percentage(self):
        panel = self._panel()

        shown = panel.show_ai_verdict_feedback(
            {
                "verdict": "FALHA FALSA",
                "is_defect": False,
                "confidence": 0.997,
            }
        )

        overlay = panel.ai_verdict_feedback
        self.assertTrue(shown)
        self.assertEqual(overlay.header_label.text(), "VEREDITO DA IA")
        self.assertEqual(overlay.verdict_label.text(), "FALHA FALSA")
        self.assertEqual(overlay.verdict_label.property("tone"), "ok")
        self.assertNotIn("%", overlay.verdict_label.text())
        self.assertNotIn("99", overlay.verdict_label.text())
        self.assertEqual(overlay.width(), VERDICT_FEEDBACK_WIDTH)
        self.assertEqual(overlay.height(), VERDICT_FEEDBACK_HEIGHT)
        self.assertTrue(
            overlay.testAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents
            )
        )
        self.assertEqual(overlay.focusPolicy(), Qt.FocusPolicy.NoFocus)

        target = overlay._top_right_position()
        self.assertEqual(
            target.x(),
            panel.width() - VERDICT_FEEDBACK_WIDTH - VERDICT_FEEDBACK_MARGIN,
        )
        self.assertEqual(target.y(), VERDICT_FEEDBACK_TOP_OFFSET)
        self.assertEqual(overlay._slide_in.endValue(), target)
        self.assertEqual(
            overlay._slide_in.startValue().x(),
            target.x() + VERDICT_FEEDBACK_SLIDE_PX,
        )

    def test_real_defect_uses_red_state_text(self):
        panel = self._panel()

        shown = panel.show_ai_verdict_feedback(
            {
                "verdict": "DEFEITO REAL",
                "is_defect": True,
                "confidence": 0.99,
            }
        )

        overlay = panel.ai_verdict_feedback
        self.assertTrue(shown)
        self.assertEqual(overlay.verdict_label.text(), "DEFEITO REAL")
        self.assertEqual(overlay.verdict_label.property("tone"), "ng")

    def test_animation_is_short_single_shot_and_non_blocking(self):
        panel = self._panel()
        overlay = panel.ai_verdict_feedback

        panel.show_ai_verdict_feedback(
            {"verdict": "FALHA FALSA", "is_defect": False}
        )

        self.assertEqual(
            overlay._fade_in.duration(),
            VERDICT_FEEDBACK_FADE_IN_MS,
        )
        self.assertEqual(
            overlay._fade_out.duration(),
            VERDICT_FEEDBACK_FADE_OUT_MS,
        )
        self.assertEqual(
            overlay._slide_in.duration(),
            VERDICT_FEEDBACK_FADE_IN_MS,
        )
        self.assertEqual(
            overlay._fade_in.easingCurve().type(),
            QEasingCurve.Type.OutCubic,
        )
        self.assertEqual(
            overlay._fade_out.easingCurve().type(),
            QEasingCurve.Type.InOutQuad,
        )
        self.assertTrue(overlay._hide_timer.isSingleShot())
        self.assertEqual(
            overlay._hide_timer.interval(),
            VERDICT_FEEDBACK_DURATION_MS - VERDICT_FEEDBACK_FADE_OUT_MS,
        )

    def test_visual_language_is_dark_yellow_with_state_color_only_on_verdict(self):
        source = open(
            "src/ui/decision_verdict_feedback.py",
            encoding="utf-8",
        ).read()

        self.assertIn("from src.ui.theme import ACCENT", source)
        self.assertIn("background-color: {SURFACE}", source)
        self.assertIn("border: 1px solid {ACCENT}", source)
        self.assertIn('QLabel#aiVerdictText[tone="ok"]', source)
        self.assertIn('QLabel#aiVerdictText[tone="ng"]', source)
        self.assertNotIn("confidence", source[source.index("class AIVerdictFeedbackOverlay"):])

    def test_clear_hides_overlay_and_removes_text(self):
        panel = self._panel()
        panel.show_ai_verdict_feedback(
            {"verdict": "DEFEITO REAL", "is_defect": True}
        )

        panel.clear_ai_verdict_feedback()

        self.assertFalse(panel.ai_verdict_feedback.isVisible())
        self.assertEqual(panel.ai_verdict_feedback.verdict_label.text(), "")


class AIVerdictFeedbackHookTests(unittest.TestCase):
    def test_reference_update_shows_and_reset_clears_feedback(self):
        class FakePanel:
            def __init__(self):
                self.events = []

            def _update_reference_panel(self, analysis):
                self.events.append(("reference", analysis["verdict"]))
                return "updated"

            def _reset_confidence_panel(self):
                self.events.append(("reset", None))
                return "reset"

            def show_ai_verdict_feedback(self, analysis):
                self.events.append(("overlay", analysis["verdict"]))

            def clear_ai_verdict_feedback(self):
                self.events.append(("clear", None))

        install_ai_verdict_feedback_hooks(FakePanel)
        panel = FakePanel()

        result = panel._update_reference_panel(
            {"verdict": "FALHA FALSA", "is_defect": False}
        )
        reset = panel._reset_confidence_panel()

        self.assertEqual(result, "updated")
        self.assertEqual(reset, "reset")
        self.assertEqual(
            panel.events,
            [
                ("reference", "FALHA FALSA"),
                ("overlay", "FALHA FALSA"),
                ("reset", None),
                ("clear", None),
            ],
        )


class AIVerdictFeedbackSourceContractTests(unittest.TestCase):
    def test_main_installs_verdict_hook_after_background_and_overlay_on_panel(self):
        source = open("main.py", encoding="utf-8").read()

        background = source.index("install_decision_background(ControlPanel)")
        hook = source.index("install_ai_verdict_feedback_hooks(ControlPanel)")
        panel = source.index("panel = ControlPanel()")
        key_overlay = source.index("install_decision_key_feedback(panel)")
        verdict_overlay = source.index("install_ai_verdict_feedback(panel)")

        self.assertLess(background, hook)
        self.assertLess(hook, panel)
        self.assertLess(panel, key_overlay)
        self.assertLess(key_overlay, verdict_overlay)


if __name__ == "__main__":
    unittest.main()
