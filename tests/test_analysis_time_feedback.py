"""Regressões do novo cartão temporário de tempo final."""
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget

from src.ui.analysis_time_feedback import (
    ANALYSIS_TIME_HEIGHT,
    ANALYSIS_TIME_TOP,
    ANALYSIS_TIME_WIDTH,
    analysis_time_feedback_state,
    install_analysis_time_feedback,
    install_analysis_time_feedback_hooks,
)
from src.ui.lighting_status_feedback import (
    LIGHTING_STATUS_HEIGHT,
    LIGHTING_STATUS_TOP_OFFSET,
)

ROOT = Path(__file__).resolve().parents[1]


def final_analysis(seconds=5.51855369983241, *, side=False):
    result = {
        "verdict": "DEFEITO REAL",
        "is_defect": True,
        "detail": {
            "analysis_time_seconds": seconds,
            "analysis_time_contract": "primeira imagem SIDE -> resultado pintado",
        },
    }
    if side:
        result["multilight_final"] = True
        result["detail"]["side_early_exit"] = {
            "analyzed_lights": ["SIDE"],
            "skipped_lights": ["TOP", "MID"],
        }
    return result


class AnalysisTimeStateTests(unittest.TestCase):
    def test_reads_actual_final_duration_for_side_and_three_lights(self):
        self.assertEqual(analysis_time_feedback_state(final_analysis()), "5.52 s")
        self.assertEqual(
            analysis_time_feedback_state(final_analysis(1.9, side=True)),
            "1.90 s",
        )
        self.assertEqual(analysis_time_feedback_state({
            "verdict": "FALHA FALSA", "is_defect": False,
            "detail": {"analysis_time_seconds": 0.0},
        }), "0.00 s")

    def test_missing_invalid_or_infinite_values_do_not_produce_card(self):
        for seconds in (-1, None, float("nan"), float("inf"), "invalid"):
            with self.subTest(seconds=seconds):
                self.assertEqual(analysis_time_feedback_state(final_analysis(seconds)), "")
        self.assertEqual(analysis_time_feedback_state(None), "")
        self.assertEqual(analysis_time_feedback_state({
            "verdict": "DEFEITO REAL", "detail": {},
        }), "")
        self.assertEqual(analysis_time_feedback_state({
            "detail": {"analysis_time_seconds": 2.5},
        }), "")


class AnalysisTimeOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _panel(self, width=1366, height=768):
        panel = QWidget()
        panel.resize(width, height)
        install_analysis_time_feedback(panel)
        return panel

    def test_starts_hidden_and_shows_only_when_final_time_exists(self):
        panel = self._panel()
        overlay = panel.analysis_time_feedback
        self.assertTrue(overlay.isHidden())
        self.assertFalse(overlay.show_analysis({"verdict": "DEFEITO REAL"}))
        self.assertTrue(overlay.isHidden())
        panel.last_analysis_time_seconds = 900.0
        self.assertTrue(panel.show_analysis_time_feedback(final_analysis()))
        self.assertEqual(overlay.value_label.text(), "5.52 s")
        self.assertFalse(overlay.isHidden())
        self.assertNotIn("900", overlay.value_label.text())
        self.assertTrue(overlay.testAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents))
        self.assertEqual(overlay.focusPolicy(), Qt.FocusPolicy.NoFocus)

    def test_intermediate_multilight_side_never_shows_final_card(self):
        panel = self._panel()
        panel.adhesive_multilight_pending_start = True
        self.assertFalse(panel.show_analysis_time_feedback(final_analysis()))
        self.assertTrue(panel.analysis_time_feedback.isHidden())
        self.assertTrue(panel.show_analysis_time_feedback(
            final_analysis(2.31, side=True)))
        self.assertEqual(panel.analysis_time_feedback.value_label.text(), "2.31 s")

    def test_fade_synchronizes_with_verdict_and_decision(self):
        panel = self._panel()
        overlay = panel.analysis_time_feedback
        panel.show_analysis_time_feedback(final_analysis())
        self.assertTrue(panel.prepare_analysis_time_feedback_dismissal())
        self.assertFalse(panel.clear_analysis_time_feedback())
        self.assertFalse(overlay.isHidden())
        self.assertTrue(panel.start_analysis_time_feedback_fade_out())
        self.assertEqual(overlay._fade_out.endValue(), 0.0)
        overlay._finish_hide()
        self.assertTrue(overlay.isHidden())
        self.assertEqual(overlay.value_label.text(), "")

    def test_discard_immediately_clears_without_waiting_for_decision_fade(self):
        panel = self._panel()
        panel.show_analysis_time_feedback(final_analysis())
        self.assertTrue(panel.clear_analysis_time_feedback())
        self.assertTrue(panel.analysis_time_feedback.isHidden())

    def test_same_right_edge_and_no_overlap_on_notebook_and_large_monitor(self):
        self.assertEqual(
            ANALYSIS_TIME_TOP,
            LIGHTING_STATUS_TOP_OFFSET + LIGHTING_STATUS_HEIGHT + 12,
        )
        for width, height in ((1024, 768), (1366, 768), (1920, 1080),
                              (2560, 1440), (340, 700)):
            with self.subTest(width=width):
                panel = self._panel(width, height)
                overlay = panel.analysis_time_feedback
                target = overlay._position()
                self.assertLessEqual(overlay.width(), ANALYSIS_TIME_WIDTH)
                self.assertEqual(overlay.height(), ANALYSIS_TIME_HEIGHT)
                self.assertGreaterEqual(target.x(), 0)
                self.assertGreaterEqual(target.y(), LIGHTING_STATUS_TOP_OFFSET
                                        + LIGHTING_STATUS_HEIGHT)
                self.assertLessEqual(target.x() + overlay.width(), width)
                self.assertLessEqual(target.y() + overlay.height(), height)


class AnalysisTimeLifecycleHookTests(unittest.TestCase):
    def test_reset_hides_on_discard_and_preserves_pending_decision(self):
        class Panel(QWidget):
            def __init__(self):
                super().__init__()
                self.resets = 0

            def _reset_confidence_panel(self):
                self.resets += 1

        install_analysis_time_feedback_hooks(Panel)
        app = QApplication.instance() or QApplication([])
        panel = Panel()
        install_analysis_time_feedback(panel)
        panel.show_analysis_time_feedback(final_analysis())
        panel._reset_confidence_panel()
        self.assertTrue(panel.analysis_time_feedback.isHidden())
        self.assertEqual(panel.resets, 1)

        panel.show_analysis_time_feedback(final_analysis())
        panel.prepare_analysis_time_feedback_dismissal()
        panel._reset_confidence_panel()
        self.assertFalse(panel.analysis_time_feedback.isHidden())
        self.assertTrue(panel.start_analysis_time_feedback_fade_out())
        panel.analysis_time_feedback._finish_hide()
        self.assertTrue(panel.analysis_time_feedback.isHidden())


class AnalysisTimeIntegrationContractTests(unittest.TestCase):
    def test_no_permanent_timer_widget_or_legacy_timer_writes(self):
        ui = (ROOT / "src/ui/control_panel_ui.py").read_text(encoding="utf-8")
        panel = (ROOT / "src/ui/control_panel.py").read_text(encoding="utf-8")
        multi = (ROOT / "src/ui/adhesive_multilight_inspection.py").read_text(
            encoding="utf-8")
        self.assertNotIn('setObjectName("latencyCard")', ui)
        self.assertNotIn("lbl_timer", ui + panel + multi)
        self.assertIn("self.header_widgets = [title_block, mode_card]", ui)
        self.assertIn("show_analysis_time_feedback", panel)
        self.assertIn("show_analysis_time_feedback", multi)

    def test_registered_in_main_and_synchronized_with_decision_overlay(self):
        entry = (ROOT / "main.py").read_text(encoding="utf-8")
        feedback = (ROOT / "src/ui/decision_key_feedback.py").read_text(
            encoding="utf-8")
        self.assertIn("install_analysis_time_feedback_hooks(ControlPanel)", entry)
        self.assertIn("install_analysis_time_feedback(panel)", entry)
        self.assertIn('"prepare_analysis_time_feedback_dismissal"', feedback)
        self.assertIn('"start_analysis_time_feedback_fade_out"', feedback)


if __name__ == "__main__":
    unittest.main()
