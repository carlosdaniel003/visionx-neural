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


def auto_side_case(label="NG"):
    """Prova de telemetria da rota exata, sem consultar banco SQLite."""
    from src.core.production_auto_side_recall import SCHEMA, RULE, _proof_digest

    analysis = case("NEW_CNN")
    analysis["is_defect"] = label == "NG"
    analysis["verdict"] = "DEFEITO REAL" if label == "NG" else "FALHA FALSA"
    analysis["lighting_mode"] = "SIDE"
    analysis["multilight_final"] = True
    analysis["eligible_for_final_decision"] = True
    d = analysis["detail"]
    d.update({
        "fusion_rule": RULE, "multilight_final": True,
        "eligible_for_final_decision": True,
        "multilight_category": "FALTANDO",
    })
    proof = {
        "schema": SCHEMA, "record_id": 12,
        "event_id": "auto-event-01", "recorded_at_ns": 1,
        "category": "FALTANDO",
        "reference_sha256": "a" * 64,
        "test_sha256": "b" * 64,
        "source_frame_sha256": "c" * 64,
        "label": label, "source_reason": "production_binary_unverified_ng_v1",
        "analyzed_lights": "SIDE,TOP,MID",
        "original_side_defect": int(label == "NG"),
        "physical_probe": (
            "REJECTION_ONLY" if label == "NG" else "NO_STRONG_EVIDENCE"
        ),
    }
    proof["integrity"] = _proof_digest(proof)
    d["production_auto_side_recall"] = proof
    return analysis


def cnn_case():
    return case("NEW_CNN",detail={
        "cnn_v2_active": True, "cnn_v2_ng_score_uncalibrated": .00025,
        "decision_trace": {},
    })


class MemoryFeedbackTests(unittest.TestCase):
    def test_exact_human_seen_has_label_but_not_approximate_similarity(self):
        a=case("KNOWN_KNN",label="OK")
        title, message, tone=memory_feedback_state(a)
        self.assertEqual(title,"JÁ VI")
        self.assertIn("OK",message)
        self.assertEqual(tone,"known")

    def test_new_cnn_is_new_pair_not_new_type_of_defect(self):
        a=cnn_case()
        title,message,tone=memory_feedback_state(a)
        self.assertEqual(title,"SEM PAR EXATO")
        self.assertIn("par",message.lower())
        self.assertEqual(tone,"new")
        self.assertNotIn("primeiro defeito",message.lower())

    def test_new_physical_specialist_is_new_pair(self):
        self.assertEqual(memory_feedback_state(case("NEW_EXPERTS"))[0],"SEM PAR EXATO")

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
        self.assertEqual(title,"JÁ VI")
        self.assertIn("SIDE",message)
        self.assertIn("TOP",message)
        self.assertEqual(tone,"known")

    def test_automatic_exact_side_is_distinct_from_human_knn(self):
        from src.ui.neural_telemetry_model import memory_seen_state
        for label in ("OK", "NG"):
            with self.subTest(label=label):
                a = auto_side_case(label)
                state = memory_seen_state(a)
                self.assertTrue(state["auto_side_exact_seen"])
                self.assertEqual(state["status"], "JA_VI")
                self.assertFalse(state["exact_pair_seen"])
                title, description, tone = memory_feedback_state(a)
                self.assertEqual(title, "JÁ VI • DECISÃO AUTOMÁTICA " + label)
                self.assertIn("memória operacional automática", description)
                self.assertIn("TOP/MID dispensadas", description)
                self.assertEqual(tone, "known")

    def test_no_false_seen_for_invalid_or_unselected_auto_proof(self):
        from src.ui.neural_telemetry_model import memory_seen_state
        from copy import deepcopy
        original = auto_side_case("NG")
        changes = (
            lambda a: a["detail"].update(fusion_rule="other"),
            lambda a: a["detail"]["production_auto_side_recall"].update(label="OK"),
            lambda a: a["detail"]["production_auto_side_recall"].update(integrity="broken"),
            lambda a: a["detail"].update(multilight_final=False),
            lambda a: a.update(is_defect=False),
            lambda a: a["detail"].update(capture_error=True),
        )
        for change in changes:
            with self.subTest(change=str(change)):
                a = deepcopy(original)
                change(a)
                self.assertFalse(memory_seen_state(a)["auto_side_exact_seen"])
                self.assertNotIn("DECISÃO AUTOMÁTICA", memory_panel_text(a)[0])

    def test_auto_side_proof_remains_visible_when_only_category_changes(self):
        from copy import deepcopy
        from src.ui.neural_telemetry_model import memory_seen_state
        analysis = deepcopy(auto_side_case("NG"))
        analysis["detail"]["multilight_category"] = "UNKNOWN"
        self.assertTrue(memory_seen_state(analysis)["auto_side_exact_seen"])

    def test_operator_known_pair_keeps_precedence_over_auto_marker(self):
        from src.ui.neural_telemetry_model import memory_seen_state
        a = auto_side_case("NG")
        a["detail"]["recognition_route"] = "KNOWN_KNN"
        a["detail"]["recognition_known_label"] = "NG"
        state = memory_seen_state(a)
        self.assertTrue(state["exact_pair_seen"])
        self.assertFalse(state["auto_side_exact_seen"])
        self.assertEqual(memory_feedback_state(a)[0], "JÁ VI")
        self.assertIn("KNN EXATO", memory_panel_text(a)[0])

    def test_conflicting_memory_requires_review(self):
        title,message,tone=memory_feedback_state(case("MEMORY_CONFLICT"))
        self.assertEqual((title,message,tone),("","",""))


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
        self.assertIn("SEM PAR EXATO",panel.lbl_memory_role.text())
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

    def test_auto_side_memory_in_real_overlay_and_bottom_panel(self):
        panel = self.make_panel()
        self.addCleanup(panel.frame_knn.deleteLater)
        self.addCleanup(panel.frame_decision_influence.deleteLater)
        a = auto_side_case("NG")
        _render_panel(panel, a)
        self.assertIn("DECISÃO AUTOMÁTICA NG", panel.lbl_memory_role.text())
        self.assertIn("memória operacional automática", panel.lbl_db_info.text())

        widget_parent = QWidget()
        self.addCleanup(widget_parent.deleteLater)
        widget_parent.resize(1100, 720)
        overlay = InspectionMemoryFeedbackOverlay(widget_parent)
        self.assertTrue(overlay.show_analysis(a))
        self.assertEqual(
            overlay.title_label.text(), "JÁ VI • DECISÃO AUTOMÁTICA NG",
        )
        self.assertIn("TOP/MID dispensadas", overlay.description_label.text())

    def test_actual_gui_overlay_remains_until_decision_fade(self):
        panel=QWidget()
        self.addCleanup(panel.deleteLater)
        panel.resize(1100,720)
        overlay=InspectionMemoryFeedbackOverlay(panel)
        self.assertTrue(overlay.show_analysis(cnn_case()))
        self.assertEqual(overlay.title_label.text(),"SEM PAR EXATO")
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
