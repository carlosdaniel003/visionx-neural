"""ODIN: veredito e estado KNN no MESMO card, sincronizados em 0/1."""
from __future__ import annotations

import os
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget

from src.ui.decision_verdict_feedback import (
    install_ai_verdict_feedback, VERDICT_FEEDBACK_HEIGHT,
)
from src.ui.decision_key_feedback import install_decision_key_feedback
from src.ui.lighting_status_feedback import (
    install_lighting_status_feedback, LIGHTING_STATUS_TOP_OFFSET,
)


def report(route, *, verdict="FALHA FALSA", label=None, light_routes=None):
    detail = {"recognition_route": route}
    if label:
        detail["recognition_known_label"] = label
    if light_routes:
        detail["recognition_light_routes"] = light_routes
    return {"is_defect": verdict == "DEFEITO REAL",
            "verdict": verdict, "detail": detail}


class CombinedVerdictMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def panel(self):
        panel = QWidget()
        panel.resize(1100, 700)
        install_decision_key_feedback(panel)
        install_ai_verdict_feedback(panel)
        install_lighting_status_feedback(panel)
        return panel

    def test_new_memory_is_subtitle_of_verdict_not_independent_overlay(self):
        p = self.panel()
        self.addCleanup(p.deleteLater)
        self.assertTrue(p.show_ai_verdict_feedback(report("NEW_CNN")))
        card = p.ai_verdict_feedback
        self.assertEqual(card.verdict_label.text(), "FALHA FALSA")
        self.assertEqual(card.memory_state_label.text(), "CASO NOVO • SEM MATCH EXATO")
        self.assertFalse(card.memory_state_label.isHidden())
        self.assertEqual(card.height(), VERDICT_FEEDBACK_HEIGHT)
        self.assertFalse(hasattr(p, "inspection_memory_feedback"))
        self.assertTrue(card.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents))

    def test_exact_known_memory_shares_verdict_card(self):
        p = self.panel()
        self.addCleanup(p.deleteLater)
        self.assertTrue(p.show_ai_verdict_feedback(report(
            "KNOWN_KNN", verdict="DEFEITO REAL", label="NG",
        )))
        card = p.ai_verdict_feedback
        self.assertEqual(card.verdict_label.text(), "DEFEITO REAL")
        self.assertEqual(card.memory_state_label.text(), "JÁ VISTO • KNN EXATO")
        self.assertIn("rótulo NG", card.toolTip())

    def test_mixed_memory_and_absent_route_are_not_misrepresented(self):
        p = self.panel()
        self.addCleanup(p.deleteLater)
        p.show_ai_verdict_feedback(report(
            "MULTILIGHT_MIXED",
            light_routes={"SIDE": "KNOWN_KNN", "TOP": "NEW_CNN", "MID": "KNOWN_KNN"},
        ))
        card = p.ai_verdict_feedback
        self.assertEqual(card.memory_state_label.text(), "MEMÓRIA MISTA • 3 LUZES")
        p.show_ai_verdict_feedback(report(""))
        self.assertEqual(card.memory_state_label.text(), "")
        self.assertTrue(card.memory_state_label.isHidden())
        self.assertNotIn("CASO NOVO", card.toolTip())

    def test_known_and_new_subtitles_follow_exact_same_opacity_timeline(self):
        p = self.panel()
        self.addCleanup(p.deleteLater)
        p.show_ai_verdict_feedback(report("NEW_CNN"))
        card = p.ai_verdict_feedback
        self.assertEqual(card.memory_state_label.graphicsEffect(), None)
        self.assertTrue(p.prepare_ai_verdict_feedback_dismissal())
        # Reset during keypress must not wipe the subtitle before fade.
        self.assertFalse(p.clear_ai_verdict_feedback())
        self.assertEqual(card.memory_state_label.text(), "CASO NOVO • SEM MATCH EXATO")
        self.assertTrue(p.show_decision_key_feedback("OK", source="production_auto"))
        self.assertTrue(p.decision_key_feedback._sync_verdict_on_exit)
        p.decision_key_feedback._start_fade_out()
        self.assertEqual(card._fade_out.endValue(), 0.0)
        self.assertEqual(card._fade_out.duration(),
                         p.decision_key_feedback._fade_out.duration())
        # One _finish_hide() clears both messages in one QWidget.
        card._finish_hide()
        self.assertEqual(card.verdict_label.text(), "")
        self.assertEqual(card.memory_state_label.text(), "")
        self.assertTrue(card.memory_state_label.isHidden())

    def test_combined_verdict_avoids_lighting_card_overlap(self):
        p = self.panel()
        self.addCleanup(p.deleteLater)
        p.show_ai_verdict_feedback(report("KNOWN_KNN", label="OK"))
        p.lighting_status_feedback.set_lighting("SIDE")
        p.show_lighting_status_feedback(report("KNOWN_KNN", label="OK"))
        rect = p.ai_verdict_feedback.geometry()
        lighting_rect = p.lighting_status_feedback.geometry()
        self.assertEqual(lighting_rect.y(), LIGHTING_STATUS_TOP_OFFSET)
        self.assertLessEqual(rect.bottom(), lighting_rect.top())
        self.assertFalse(hasattr(p, "inspection_memory_feedback"))

    def test_production_review_still_overrides_model_approval(self):
        p = self.panel()
        self.addCleanup(p.deleteLater)
        a = report("NEW_CNN")
        a["production_review_required"] = True
        p.show_ai_verdict_feedback(a)
        self.assertEqual(p.ai_verdict_feedback.verdict_label.text(), "REVISÃO OBRIGATÓRIA")
        self.assertEqual(p.ai_verdict_feedback.memory_state_label.text(),
                         "CASO NOVO • SEM MATCH EXATO")

    def test_entrypoint_has_no_separate_memory_overlay_installers(self):
        main = Path("main.py").read_text(encoding="utf-8")
        keys = Path("src/ui/decision_key_feedback.py").read_text(encoding="utf-8")
        self.assertNotIn("install_inspection_memory_feedback(panel)", main)
        self.assertNotIn("install_inspection_memory_feedback_hooks(ControlPanel)", main)
        self.assertNotIn("start_inspection_memory_feedback_fade_out", keys)
        self.assertIn("install_ai_verdict_feedback(panel)", main)


if __name__ == "__main__":
    unittest.main()
