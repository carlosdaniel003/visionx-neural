"""Regressão da memória KNN visual do evento real 6a62411f... (16/10).

O roteador exato informou NOT_FOUND, mas a KNN visual recuperou NG muito
semelhante nas três iluminações. UI não pode anunciar NUNCA VI nesse caso.
Não treina CNN, não altera KNN, não autoriza comando ao XP.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QWidget

from src.ui.neural_telemetry_model import memory_panel_text, memory_seen_state
from src.ui.inspection_memory_feedback import memory_feedback_state
from src.ui.decision_verdict_feedback import install_ai_verdict_feedback
from src.ui.widgets.knn_spectrum import KNNSpectrumWidget


LIGHTS = ("SIDE", "TOP", "MID")
NG_SCORES = {"SIDE": 0.999124, "TOP": 0.999605, "MID": 0.999427}
OK_SCORES = {"SIDE": 0.926033, "TOP": 0.77638, "MID": 0.876728}
SUPPORTS = {"SIDE": 31, "TOP": 12, "MID": 18}


def reported_event():
    return {
        "verdict": "DEFEITO REAL", "is_defect": True,
        "multilight_final": True,
        "detail": {
            "recognition_route": "NEW_CNN",
            "recognition_match": "NOT_FOUND",
            "recognition_memory_verified": False,
            "recognition_known_label": "",
            "multilight_category": "FALTANDO",
            "cnn_v2_active": True,
            "cnn_v2_consensus": "REVIEW",
            "cnn_v2_checkpoint_sha256":
                "6e4a31e8826d7b2afa18fbecb579a4d8979067713faa032d329f37f54729b599",
            "recognition_light_routes": {light: "NEW_CNN" for light in LIGHTS},
            "knn_visual_corroboration": {
                "status": "NG_CORROBORATED",
                "strong_ng_lights": list(LIGHTS),
                "trend_ng_lights": list(LIGHTS),
                "auto_decision_authorized": False,
                "lights": {
                    light: {
                        "status": "NG_SIMILAR",
                        "ng_support": SUPPORTS[light],
                        "ng": {"count": 94, "best_similarity": NG_SCORES[light]},
                        "ok": {"count": 30, "best_similarity": OK_SCORES[light]},
                    }
                    for light in LIGHTS
                },
            },
        },
    }


class VisualMemoryMeaningTests(unittest.TestCase):
    def test_real_three_light_ng_archive_is_seen_visually(self):
        case = reported_event()
        state = memory_seen_state(case)
        self.assertEqual(state["status"], "VI_SEMELHANTE")
        self.assertTrue(state["approximate_ng_seen"])
        self.assertFalse(state["exact_pair_seen"])
        self.assertEqual(state["recognized_lights"], [])
        self.assertEqual(state["approximate_ng_lights"], list(LIGHTS))
        self.assertEqual(state["routes_by_light"]["SIDE"], "NEW_CNN")
        self.assertEqual(memory_feedback_state(case)[0], "VI SEMELHANTE")
        self.assertEqual(memory_feedback_state(case)[2], "similar")
        title, description = memory_panel_text(case)
        self.assertIn("VI SEMELHANTE", title)
        self.assertIn("3/3", title)
        for light in LIGHTS:
            self.assertIn(light, description)
            self.assertIn(f"{NG_SCORES[light]*100:.2f}%", description)
        self.assertIn("não pares exatos", description)
        self.assertIn("CNN", description)
        self.assertEqual(case["detail"]["recognition_route"], "NEW_CNN")
        self.assertFalse(case["detail"]["knn_visual_corroboration"]["auto_decision_authorized"])

    def test_json_human_ok_visual_recall_is_visible_but_not_exact(self):
        case = reported_event()
        case["detail"].pop("knn_visual_corroboration", None)
        case["detail"]["confirmed_visual_recall"] = {
            "schema": "visionx.human_visual_recall.v1",
            "source": "PERSISTENT_HUMAN_LABELS",
            "diagnostic_only": True,
            "auto_ok_authorized": False,
            "status": "HUMAN_OK_VISUAL_CANDIDATE",
            "best_ok_similarity": .991,
            "best_ng_similarity": .70,
            "cross_light_candidate": True,
        }
        state = memory_seen_state(case)
        self.assertEqual(state["status"], "VI_SEMELHANTE")
        self.assertFalse(state["exact_pair_seen"])
        self.assertTrue(state["human_visual_seen"])
        self.assertEqual(state["human_visual_label"], "OK")
        self.assertEqual(memory_feedback_state(case)[0], "VI SEMELHANTE")
        title, body = memory_panel_text(case)
        self.assertIn("JULGAMENTO HUMANO OK", title)
        self.assertIn("99.10%", body)
        self.assertIn("não autoriza 0/OK", body)
        self.assertEqual(case["verdict"], "DEFEITO REAL")

        # Metadado visual arbitrário não pode fingir memória humana.
        case["detail"]["confirmed_visual_recall"]["source"] = "PUBLIC_ARCHIVE_PNG"
        self.assertEqual(memory_seen_state(case)["status"], "NUNCA_VI")

    def test_exact_human_match_remains_higher_priority_than_approximate(self):
        case = reported_event()
        case["detail"]["recognition_route"] = "KNOWN_KNN"
        case["detail"]["recognition_known_label"] = "NG"
        case["detail"]["recognition_light_routes"]["SIDE"] = "KNOWN_KNN"
        state = memory_seen_state(case)
        self.assertEqual(state["status"], "JA_VI")
        self.assertFalse(state["approximate_ng_seen"])
        self.assertEqual(memory_feedback_state(case)[0], "JÁ VI")

    def test_visual_status_alone_does_not_invent_seen_history(self):
        for bad in [
            {"status": "NG_CORROBORATED", "strong_ng_lights": list(LIGHTS)},
            {"status": "NG_CORROBORATED",
             "strong_ng_lights": ["SIDE"],
             "lights": {"SIDE": {"status": "NG_SIMILAR"}}},
            {"status": "NG_TREND_THREE_LIGHTS",
             "trend_ng_lights": list(LIGHTS),
             "lights": {"SIDE": {"status": "NG_LEANING"}}},
        ]:
            case = reported_event()
            case["detail"]["knn_visual_corroboration"] = bad
            with self.subTest(bad=bad):
                self.assertEqual(memory_seen_state(case)["status"], "NUNCA_VI")
                self.assertEqual(memory_feedback_state(case)[0], "SEM PAR EXATO")

    def test_new_without_neighbors_is_only_no_exact_match(self):
        case = reported_event()
        del case["detail"]["knn_visual_corroboration"]
        self.assertEqual(memory_seen_state(case)["status"], "NUNCA_VI")
        self.assertIn("SEM PAR EXATO", memory_panel_text(case)[0])
        self.assertIn("não significa", memory_panel_text(case)[1])

    def test_three_light_ng_trend_has_same_visual_memory_display(self):
        case = reported_event()
        visual = case["detail"]["knn_visual_corroboration"]
        visual["status"] = "NG_TREND_THREE_LIGHTS"
        visual["strong_ng_lights"] = []
        for light in LIGHTS:
            visual["lights"][light]["status"] = "NG_LEANING"
        self.assertEqual(memory_seen_state(case)["status"], "VI_SEMELHANTE")

    def test_missing_memory_route_with_real_visual_never_claims_never_seen(self):
        case = reported_event()
        case["detail"]["recognition_route"] = ""
        case["detail"]["recognition_light_routes"] = {}
        self.assertEqual(memory_seen_state(case)["status"], "VI_SEMELHANTE")


class VisualMemoryQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_live_verdict_card_shows_seen_similar_but_preserves_ng(self):
        widget = QWidget()
        self.addCleanup(widget.deleteLater)
        widget.resize(1100, 720)
        install_ai_verdict_feedback(widget)
        self.assertTrue(widget.show_ai_verdict_feedback(reported_event()))
        card = widget.ai_verdict_feedback
        self.assertEqual(card.verdict_label.text(), "DEFEITO REAL")
        self.assertEqual(card.memory_state_label.text(), "VI SEMELHANTE")
        self.assertIn("NG", card.toolTip())
        self.assertIn("não pares exatos", card.toolTip())
        self.assertNotIn("SEM PAR EXATO", card.toolTip())

    def test_knn_spectrum_shows_similar_ng_in_all_three_light_cards(self):
        card = KNNSpectrumWidget()
        self.addCleanup(card.deleteLater)
        card.resize(900, 260)
        card.update_data(reported_event()["detail"])
        self.assertEqual(card.seen_state["status"], "VI_SEMELHANTE")
        self.assertEqual(card.seen_state["approximate_ng_lights"], list(LIGHTS))
        self.assertIn("VI SEMELHANTE", card.route_header)
        self.assertFalse(card.grab().isNull())


if __name__ == "__main__":
    unittest.main()
