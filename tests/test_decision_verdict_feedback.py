import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEasingCurve, Qt
from PyQt6.QtWidgets import QApplication, QWidget

from src.ui.decision_verdict_feedback import (
    VERDICT_FEEDBACK_FADE_IN_MS,
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

    def test_overlay_enters_once_and_stays_visible_until_explicit_reset(self):
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
            overlay._slide_in.duration(),
            VERDICT_FEEDBACK_FADE_IN_MS,
        )
        self.assertEqual(
            overlay._fade_in.easingCurve().type(),
            QEasingCurve.Type.OutCubic,
        )
        self.assertTrue(overlay.isVisible())
        self.assertFalse(hasattr(overlay, "_hide_timer"))
        self.assertFalse(hasattr(overlay, "_fade_out"))

        source = open(
            "src/ui/decision_verdict_feedback.py",
            encoding="utf-8",
        ).read()
        self.assertNotIn("QTimer", source)
        self.assertNotIn("VERDICT_FEEDBACK_DURATION_MS", source)
        self.assertNotIn("VERDICT_FEEDBACK_FADE_OUT_MS", source)

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

        panel = self._panel()
        panel.show_ai_verdict_feedback(
            {
                "verdict": "FALHA FALSA",
                "is_defect": False,
                "confidence": 0.997,
            }
        )
        rendered_texts = {
            panel.ai_verdict_feedback.verdict_label.text(),
        }
        self.assertEqual(rendered_texts, {"FALHA FALSA"})
        self.assertFalse(any("%" in value for value in rendered_texts))
        self.assertFalse(any("99" in value for value in rendered_texts))
        self.assertNotIn("aiVerdictHeader", source)
        self.assertNotIn("aiVerdictHint", source)

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
                self.current_analysis = object()
                self.is_locked = True

            def _update_reference_panel(self, analysis):
                self.current_analysis = analysis
                self.is_locked = True
                self.events.append(("reference", analysis["verdict"]))
                return "updated"

            def _reset_confidence_panel(self):
                self.current_analysis = None
                self.is_locked = False
                self.events.append(("reset", None))
                return "reset"

            def save_label(self, decision, source="button"):
                self.current_analysis = None
                self.is_locked = False
                self.events.append(("save", decision))
                return source

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

        panel.events.clear()
        panel._update_reference_panel(
            {"verdict": "DEFEITO REAL", "is_defect": True}
        )
        panel.save_label("NG")
        self.assertEqual(
            panel.events,
            [
                ("reference", "DEFEITO REAL"),
                ("overlay", "DEFEITO REAL"),
                ("save", "NG"),
                ("clear", None),
            ],
        )


class AIVerdictFeedbackSourceContractTests(unittest.TestCase):
    def test_main_installs_persistent_verdict_overlay_without_dynamic_background(self):
        source = open("main.py", encoding="utf-8").read()

        hook = source.index("install_ai_verdict_feedback_hooks(ControlPanel)")
        panel = source.index("panel = ControlPanel()")
        key_overlay = source.index("install_decision_key_feedback(panel)")
        verdict_overlay = source.index("install_ai_verdict_feedback(panel)")

        self.assertNotIn("install_decision_background", source)
        self.assertLess(hook, panel)
        self.assertLess(panel, key_overlay)
        self.assertLess(key_overlay, verdict_overlay)


if __name__ == "__main__":
    unittest.main()
