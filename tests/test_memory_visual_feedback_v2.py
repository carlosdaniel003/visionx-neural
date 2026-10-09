"""Correções: memória KNN sempre explicada; influência visual; overlay de ocorrência."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QLabel, QWidget

from src.ui.neural_telemetry_model import memory_panel_text
from src.ui.inspection_memory_feedback import (
    InspectionMemoryFeedbackOverlay, memory_feedback_state,
)
from src.ui.decision_panel import _render_panel
from src.ui.widgets.knn_spectrum import KNNSpectrumWidget
from src.ui.widgets.decision_influence import DecisionInfluenceWidget
from src.ui.neural_influence_model import neural_influence_rows


def case(route, *, label="", detail=None):
    d={"recognition_route": route}
    if label:
        d["recognition_known_label"] = label
    d.update(detail or {})
    return {"verdict": "FALHA FALSA", "is_defect": False, "detail": d}


def cnn_case():
    return case("NEW_CNN",detail={
        "cnn_v2_active": True, "cnn_v2_ng_score_uncalibrated": .00025,
        "decision_trace": {},
    })


class MemoryFeedbackTests(unittest.TestCase):
    def test_exact_human_seen_has_label_but_not_approximate_similarity(self):
        a=case("KNOWN_KNN",label="OK")
        title, message, tone=memory_feedback_state(a)
        self.assertIn("JÁ VISTO",title)
        self.assertIn("OK",message)
        self.assertEqual(tone,"known")

    def test_new_cnn_is_new_pair_not_new_type_of_defect(self):
        a=cnn_case()
        title,message,tone=memory_feedback_state(a)
        self.assertIn("CASO NOVO",title)
        self.assertIn("par",message)
        self.assertEqual(tone,"new")
        self.assertNotIn("primeiro defeito",message.lower())

    def test_new_physical_specialist_is_new_pair(self):
        self.assertIn("CASO NOVO",memory_feedback_state(case("NEW_EXPERTS"))[0])

    def test_unknown_route_does_not_claim_first_occurrence(self):
        self.assertEqual(memory_feedback_state(case("")),("","",""))
        self.assertEqual(memory_feedback_state({}),("","",""))

    def test_multilight_mixed_not_mislabeled_first_occurrence(self):
        a=case("MULTILIGHT_MIXED", detail={
            "recognition_light_routes":{
                "SIDE":"KNOWN_KNN","TOP":"NEW_CNN","MID":"KNOWN_KNN"
            }
        })
        title,message,tone=memory_feedback_state(a)
        self.assertIn("MISTA",title)
        self.assertIn("SIDE",message)
        self.assertIn("TOP",message)
        self.assertEqual(tone,"mixed")

    def test_conflicting_memory_requires_review(self):
        title,message,tone=memory_feedback_state(case("MEMORY_CONFLICT"))
        self.assertIn("CONFLITANTE",title)
        self.assertEqual(tone,"review")


class MemoryPanelRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt=QApplication.instance() or QApplication([])

    def make_panel(self):
        class MockPanel:
            def __init__(self):
                self.frame_decision_influence=DecisionInfluenceWidget()
                self.frame_knn=KNNSpectrumWidget()
                self.lbl_decision_summary=QLabel()
                self.lbl_decision_rule=QLabel()
                self.lbl_db_info=QLabel()
                self.lbl_memory_role=QLabel()
                self.lbl_verdict=QLabel()
        return MockPanel()

    def test_no_trace_still_populates_memory_area_with_actual_route(self):
        panel=self.make_panel()
        self.addCleanup(panel.frame_knn.deleteLater)
        self.addCleanup(panel.frame_decision_influence.deleteLater)
        _render_panel(panel,cnn_case())
        self.assertIn("KNN SEM MATCH EXATO",panel.lbl_memory_role.text())
        self.assertIn("par humano exato não encontrado",panel.lbl_db_info.text())
        self.assertIn("NEW_CNN",panel.frame_knn.recognition_route)
        self.assertTrue(panel.frame_knn.isVisible() or not panel.frame_knn.isHidden())

    def test_known_knn_not_replaced_by_legacy_empty_text(self):
        panel=self.make_panel()
        self.addCleanup(panel.frame_knn.deleteLater)
        self.addCleanup(panel.frame_decision_influence.deleteLater)
        _render_panel(panel,case("KNOWN_KNN",label="NG",detail={
            "decision_trace":{"fusion_rule":"knn_verified_pair_exact",
                              "dominant_engine":"knn_verified_exact"}
        }))
        self.assertIn("KNN EXATO",panel.lbl_memory_role.text())
        self.assertIn("NG",panel.lbl_db_info.text())
        self.assertNotIn("Dataset sem memória",panel.lbl_db_info.text())

    def test_actual_gui_overlay_remains_until_decision_fade(self):
        panel=QWidget()
        self.addCleanup(panel.deleteLater)
        panel.resize(1100,720)
        overlay=InspectionMemoryFeedbackOverlay(panel)
        self.assertTrue(overlay.show_analysis(cnn_case()))
        self.assertIn("CASO NOVO",overlay.title_label.text())
        self.assertTrue(overlay.prepare_decision_dismissal())
        self.assertFalse(overlay.clear())
        self.assertTrue(overlay.start_synchronized_fade_out())
        self.assertTrue(overlay.clear(force=True))

    def test_influence_card_has_per_light_scores_and_knn_label(self):
        a=case("NEW_CNN",detail={
            "cnn_v2_light_diagnostics":{
                "SIDE":{"cnn_active":True,"ng_score_uncalibrated":.01,"verdict":"FALHA FALSA"},
                "TOP":{"cnn_active":True,"ng_score_uncalibrated":.98,"verdict":"DEFEITO REAL"},
                "MID":{"cnn_active":True,"ng_score_uncalibrated":.10,"verdict":"FALHA FALSA"},
            },
        })
        w=DecisionInfluenceWidget()
        self.addCleanup(w.deleteLater)
        w.resize(850,315)
        w.update_data(a)
        self.assertEqual(len(w.rows),3)
        self.assertEqual(w.rows[0]["ng_score_uncalibrated"],.01)
        self.assertEqual(w.rows[1]["ng_score_uncalibrated"],.98)
        self.assertTrue(w.grab().toImage().width() > 0)
        self.assertIn("Verde",w.toolTip())

    def test_influence_knn_does_not_claim_cnn_probability(self):
        rows=neural_influence_rows(case("KNOWN_KNN",label="OK"))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["known_memory_label"],"OK")
        self.assertIsNone(rows[0]["ng_score_uncalibrated"])

    def test_inconclusive_memory_falls_back_to_explicit_message(self):
        title,message=memory_panel_text(case(""))
        self.assertTrue(title)
        self.assertTrue(message)
        self.assertNotIn("CASO NOVO",title)


if __name__=="__main__":
    unittest.main()
