"""Regressões binárias: SIDE/TOP conhecidos + MID novo = JÁ VI.

O status é só telemetria do histórico KNN; NÃO altera roteamento, CNN,
consenso, liberação 0/1 nem persistência da memória.
"""
from __future__ import annotations

import os
import unittest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QWidget
from src.ui.neural_telemetry_model import memory_seen_state, memory_panel_text
from src.ui.inspection_memory_feedback import memory_feedback_state
from src.ui.decision_verdict_feedback import install_ai_verdict_feedback
from src.ui.widgets.knn_spectrum import KNNSpectrumWidget


def make_case(side="KNOWN_KNN", top="KNOWN_KNN", mid="NEW_CNN"):
    routes = {"SIDE": side, "TOP": top, "MID": mid}
    return {
        "verdict": "FALHA FALSA", "is_defect": False,
        "detail": {
            "recognition_route": "MULTILIGHT_MIXED"
                if len(set(routes.values())) > 1 else next(iter(routes.values())),
            "recognition_match": "NOT_FOUND",
            "recognition_memory_verified": False,
            "recognition_light_routes": routes,
            "cnn_v2_light_diagnostics": {
                "SIDE": {
                    "route": side, "human_memory_label": "OK",
                    "cnn_active": side == "NEW_CNN",
                },
                "TOP": {
                    "route": top, "human_memory_label": "OK",
                    "cnn_active": top == "NEW_CNN",
                },
                "MID": {
                    "route": mid, "human_memory_label": "",
                    "cnn_active": mid == "NEW_CNN",
                },
            },
        },
    }


class BinarySeenLogicTests(unittest.TestCase):
    def test_uploaded_debug_side_top_seen_mid_new_is_seen(self):
        analysis = make_case()
        status = memory_seen_state(analysis)
        self.assertEqual(status["status"], "JA_VI")
        self.assertEqual(status["label"], "JÁ VI")
        self.assertEqual(status["recognized_lights"], ["SIDE", "TOP"])
        self.assertEqual(status["new_lights"], ["MID"])
        self.assertEqual(memory_feedback_state(analysis)[0], "JÁ VI")
        title, explanation = memory_panel_text(analysis)
        self.assertIn("JÁ VI", title)
        self.assertIn("2/3", title)
        self.assertIn("SIDE, TOP", explanation)
        self.assertIn("MID", explanation)
        self.assertNotIn("MEMÓRIA MISTA", title)

    def test_any_single_recognized_light_means_seen(self):
        for known in ("SIDE", "TOP", "MID"):
            lights = ["NEW_CNN"]*3
            lights[("SIDE", "TOP", "MID").index(known)] = "KNOWN_KNN"
            with self.subTest(known=known):
                state = memory_seen_state(make_case(*lights))
                self.assertEqual(state["status"], "JA_VI")
                self.assertEqual(state["recognized_lights"], [known])

    def test_no_known_illumination_means_never_seen(self):
        analysis = make_case("NEW_CNN", "NEW_CNN", "NEW_CNN")
        state = memory_seen_state(analysis)
        self.assertEqual(state["status"], "NUNCA_VI")
        self.assertEqual(memory_feedback_state(analysis)[0], "NUNCA VI")
        self.assertIn("0/3", self._caption(state))
        self.assertIn("NUNCA VI", memory_panel_text(analysis)[0])

    @staticmethod
    def _caption(state):
        return f"{len(state['recognized_lights'])}/3"

    def test_unavailable_memory_does_not_claim_never_seen(self):
        analysis = {"verdict": "FALHA FALSA", "detail": {}}
        self.assertIsNone(memory_seen_state(analysis)["status"])
        self.assertEqual(memory_feedback_state(analysis), ("", "", ""))
        self.assertIn("CONSULTA KNN", memory_panel_text(analysis)[0])

    def test_single_exact_memory_pair_is_seen(self):
        analysis = {"verdict": "DEFEITO REAL", "detail": {
            "recognition_route": "KNOWN_KNN",
            "recognition_known_label": "NG",
            "recognition_memory_verified": True,
        }}
        self.assertEqual(memory_seen_state(analysis)["status"], "JA_VI")
        self.assertEqual(memory_feedback_state(analysis)[0], "JÁ VI")
        self.assertIn("Rótulo NG", memory_panel_text(analysis)[1])

    def test_conflict_without_valid_exact_pair_does_not_claim_absence(self):
        analysis = {"verdict": "REVISÃO OBRIGATÓRIA", "detail": {
            "recognition_route": "MEMORY_CONFLICT",
        }}
        self.assertEqual(memory_feedback_state(analysis), ("", "", ""))

    def test_binary_label_is_independent_of_classification(self):
        a = make_case()
        a["verdict"] = "DEFEITO REAL"
        a["is_defect"] = True
        self.assertEqual(memory_feedback_state(a)[0], "JÁ VI")


class SeenVisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_verdict_overlay_shows_only_two_memory_words(self):
        p = QWidget()
        self.addCleanup(p.deleteLater)
        p.resize(1100, 780)
        install_ai_verdict_feedback(p)
        shown = p.show_ai_verdict_feedback(make_case())
        self.assertTrue(shown)
        overlay = p.ai_verdict_feedback
        self.assertEqual(overlay.verdict_label.text(), "FALHA FALSA")
        self.assertEqual(overlay.memory_state_label.text(), "JÁ VI")
        self.assertNotIn("MEMÓRIA MISTA", overlay.memory_state_label.text())
        self.assertIn("SIDE", overlay.toolTip())
        self.assertTrue(p.prepare_ai_verdict_feedback_dismissal())
        self.assertFalse(p.clear_ai_verdict_feedback())
        self.assertTrue(p.start_ai_verdict_feedback_fade_out())
        overlay._finish_hide()
        self.assertEqual(overlay.memory_state_label.text(), "")

    def test_memory_panel_uses_colored_cards_for_all_three_lights(self):
        widget = KNNSpectrumWidget()
        self.addCleanup(widget.deleteLater)
        widget.resize(930, 310)
        widget.update_data(make_case()["detail"])
        self.assertEqual(widget.seen_state["recognized_lights"], ["SIDE","TOP"])
        image = widget.grab().toImage()
        self.assertEqual(image.width(), 930)
        # Sampling card background, outside text/borders.
        self.assertEqual(image.pixelColor(18, 115).name(), "#152c20")
        self.assertEqual(image.pixelColor(324, 115).name(), "#152c20")
        self.assertEqual(image.pixelColor(630, 115).name(), "#2a2418")

    def test_narrow_panel_is_rendered_not_empty_black(self):
        widget = KNNSpectrumWidget()
        self.addCleanup(widget.deleteLater)
        widget.resize(345, 338)
        widget.update_data(make_case()["detail"])
        self.assertEqual(widget.grab().toImage().height(), 338)
        self.assertIn("JÁ VI", widget.route_header)

    def test_unknown_route_still_has_panel_explanation(self):
        widget = KNNSpectrumWidget()
        self.addCleanup(widget.deleteLater)
        widget.resize(700, 300)
        widget.update_data({"recognition_route": "MEMORY_CONFLICT"})
        self.assertIsNone(widget.seen_state["status"])
        self.assertTrue(widget.route_header)


if __name__ == "__main__":
    unittest.main()
