"""Painéis CNN/KNN: scores reais por luz, rota e debug sem métricas falsas."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.core.multilight_fusion import fuse_multilight
from src.services.capture_debug_payload import decision_record
from src.ui.neural_telemetry_model import (
    cnn_panel_text, memory_panel_text, neural_summary,
)
from src.ui.neural_influence_model import neural_influence_rows
from src.ui.network_xp_debug import format_network_debug_report
from src.ui.widgets.neural_specialist import (
    NeuralSpecialistWidget, VerifiedMemorySpecialistWidget,
)
from src.ui.widgets.decision_influence import DecisionInfluenceWidget
from src.ui.widgets.knn_spectrum import KNNSpectrumWidget
from src.ui.adhesive_multilight_analysis import _LightingExpertLane


SHA = "6e4a31e8826d7b2afa18fbecb579a4d8979067713faa032d329f37f54729b599"
SCORES = {"SIDE": .0002535293169785291,
          "TOP": .0001943337410921231,
          "MID": .00002622978536237497}


def make_light(light):
    score = SCORES[light]
    return {
        "lighting_mode": light, "is_defect": False,
        "verdict": "FALHA FALSA", "confidence": 1-score,
        "active_engines": ["faltando_cnn_v2.py"],
        "production_review_required": False,
        "detail": {
            "recognition_route": "NEW_CNN",
            "cnn_v2_aoi_category": "INVERTIDO",
            "cnn_v2_active": True, "cnn_v2_experimental": True,
            "cnn_v2_status": "INFERENCE_OK",
            "cnn_v2_checkpoint_sha256": SHA,
            "cnn_v2_checkpoint_verified": True,
            "cnn_v2_checkpoint_best_epoch": 18,
            "cnn_v2_lighting_mode": light,
            "cnn_v2_ng_score_uncalibrated": score,
            "final_score": score, "physical_score": 0.0,
            "dominant_engine": "faltando_cnn_v2",
            "fusion_rule": "cnn_v2_dualscale_direct",
            "decision_trace": {
                "cnn_ng_score_uncalibrated": score,
                "fusion_rule": "cnn_v2_dualscale_direct",
                "operator_review_required": False,
                "final_score": score, "physical_score": 0.0,
                "weights": {"cnn": 1.0, "knn": 0.0, "physical": 0.0},
            },
        },
    }


def combined():
    return fuse_multilight(
        {light: make_light(light) for light in SCORES}, "INVERTIDO"
    )


class TelemetryModelTests(unittest.TestCase):
    def test_original_three_scores_preserved_not_copied_from_side(self):
        final = combined()
        diag = final["detail"]["cnn_v2_light_diagnostics"]
        self.assertEqual(set(diag), {"SIDE", "TOP", "MID"})
        for light, value in SCORES.items():
            self.assertAlmostEqual(diag[light]["ng_score_uncalibrated"], value)
            self.assertEqual(diag[light]["checkpoint_sha256"], SHA)
        self.assertEqual(final["detail"]["cnn_v2_consensus"], "OK")

    def test_cnn_panel_multilight_has_real_values_per_light(self):
        data = cnn_panel_text(combined())
        self.assertIn("CNN FALTANDO v2", data["header"])
        for light in SCORES:
            self.assertTrue(any(light + ":" in item for item in data["lines"]))
        self.assertTrue(any("não calibrad" in s for s in data["lines"]))

    def test_multilight_influence_has_three_cnn_rows_not_physical(self):
        rows = neural_influence_rows(combined())
        self.assertEqual([x["id"] for x in rows],
                         ["cnn_side", "cnn_top", "cnn_mid"])
        self.assertTrue(all(row["telemetry_row"] for row in rows))
        self.assertTrue(all(row["fusion_weight"] == 0 for row in rows))
        self.assertTrue(all("score NG" in row["display_text"] for row in rows))

    def test_knn_new_case_does_not_fabricate_similarity_zero(self):
        header, description = memory_panel_text(combined())
        self.assertIn("KNN SEM MATCH EXATO", header)
        self.assertIn("não existe similaridade knn medida", description.lower())

    def test_known_knn_label_is_not_a_cnn_score(self):
        known = {
            "verdict": "DEFEITO REAL",
            "detail": {
                "recognition_route": "KNOWN_KNN",
                "recognition_known_label": "NG",
                "recognition_memory_verified": True,
                "recognition_memory_path": "human/2026-10-09.json",
                "decision_trace": {"dominant_engine": "knn_verified_exact"},
            },
        }
        self.assertFalse(neural_summary(known)["cnn_active"])
        self.assertIn("NG", memory_panel_text(known)[1])
        self.assertEqual(neural_influence_rows(known)[0]["id"], "knn_verified")

    def test_missing_score_is_not_treated_as_zero(self):
        frame = make_light("SIDE")
        del frame["detail"]["cnn_v2_ng_score_uncalibrated"]
        del frame["detail"]["decision_trace"]["cnn_ng_score_uncalibrated"]
        state = neural_summary(frame)
        self.assertIsNone(state["cnn_ng_score"])
        self.assertIn("N/D", "\n".join(cnn_panel_text(frame)["lines"]))

    def test_debug_record_exposes_model_and_consensus(self):
        a = combined()
        payload = decision_record(a, {"category": "INVERTIDO"})
        self.assertTrue(payload["cnn_v2"]["active"])
        self.assertEqual(payload["cnn_v2"]["consensus"], "OK")
        self.assertEqual(payload["cnn_v2"]["light_diagnostics"]["TOP"]
                         ["ng_score_uncalibrated"], SCORES["TOP"])
        self.assertEqual(payload["recognition"]["route"], "NEW_CNN")

    def test_debug_report_has_dedicated_neural_section_and_knn_route(self):
        a = combined()
        record = {
            "schema": "visionx.network_xp_debug.v1",
            "source": "windows_xp", "source_ip": "169.254.95.200",
            "decision": decision_record(a, {"category": "INVERTIDO"}),
        }
        output = format_network_debug_report(record)
        self.assertIn("REDE NEURAL CNN FALTANDO v2", output)
        self.assertIn("MEMÓRIA KNN (ROTA EFETIVA)", output)
        self.assertIn("SIDE: NEW_CNN", output)
        self.assertIn("TOP: NEW_CNN", output)
        self.assertIn("MID: NEW_CNN", output)
        self.assertIn("0.019433%", output)
        self.assertNotIn("Missing similaridade direta:", output)
        self.assertNotIn("INVERTIDO transformação ganho:", output)

    def test_debug_known_knn_without_cnn_explicitly_shows_human_label(self):
        a = {
            "verdict": "DEFEITO REAL", "is_defect": True,
            "detail": {
                "recognition_route": "KNOWN_KNN",
                "recognition_match": "EXACT_PAIR",
                "recognition_memory_verified": True,
                "recognition_known_label": "NG",
                "recognition_reason": "Par humano idêntico",
            },
        }
        record = {
            "source": "windows_xp",
            "decision": decision_record(a, {"category": "INVERTIDO"}),
        }
        debug = format_network_debug_report(record)
        self.assertIn("MEMÓRIA KNN (ROTA EFETIVA)", debug)
        self.assertIn("Rota: KNOWN_KNN", debug)
        self.assertIn("Rótulo humano recuperado: NG", debug)
        self.assertNotIn("REDE NEURAL CNN FALTANDO v2", debug)

    def test_physical_engine_legacy_not_recast_as_neural(self):
        analysis = {"verdict": "FALHA FALSA", "detail": {
            "recognition_route": "NEW_EXPERTS", "decision_trace": {}
        }}
        self.assertFalse(neural_summary(analysis)["cnn_active"])
        self.assertEqual(neural_influence_rows(analysis), [])


class TelemetryWidgetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def test_expert_lane_renders_cnn_instead_of_looking_empty(self):
        lane = _LightingExpertLane("SIDE")
        self.addCleanup(lane.deleteLater)
        self.assertTrue(lane.set_analysis(make_light("SIDE")))
        widget = lane.frames["faltando_cnn_v2.py"]
        self.assertIn("CNN FALTANDO v2", widget.title.text())
        self.assertTrue(any("0.0254%" in line.text() for line in widget.lines))
        self.assertFalse(widget.isHidden())
        self.assertTrue(lane.frames["ssim_expert.py"].isHidden())

    def test_known_knn_has_visible_expert(self):
        lane = _LightingExpertLane("TOP")
        self.addCleanup(lane.deleteLater)
        analysis = {
            "verdict": "FALHA FALSA",
            "active_engines": ["knn_expert.py"],
            "detail": {"recognition_route": "KNOWN_KNN",
                       "recognition_known_label": "OK"},
        }
        self.assertTrue(lane.set_analysis(analysis))
        self.assertIn("KNN EXATO", lane.frames["knn_expert.py"].title.text())

    def test_influence_widget_uses_real_per_light_rows(self):
        w = DecisionInfluenceWidget()
        self.addCleanup(w.deleteLater)
        w.update_data(combined())
        self.assertEqual(len(w.rows), 3)

    def test_memory_spectrum_shows_route_not_zero_bars(self):
        w = KNNSpectrumWidget()
        self.addCleanup(w.deleteLater)
        w.update_data(combined()["detail"])
        self.assertEqual(w.recognition_route, "NEW_CNN")
        self.assertIn("KNN SEM MATCH EXATO", w.route_header)

    def test_neural_card_uses_unmodified_original_score(self):
        w = NeuralSpecialistWidget()
        self.addCleanup(w.deleteLater)
        local = make_light("MID")
        w.update_data(local["detail"], local)
        self.assertTrue(any("0.0026%" in lbl.text() for lbl in w.lines))

    def test_memory_card_known_without_network_read(self):
        w = VerifiedMemorySpecialistWidget()
        self.addCleanup(w.deleteLater)
        analysis = {"detail": {
            "recognition_route": "KNOWN_KNN", "recognition_known_label": "OK"
        }}
        w.update_data(analysis["detail"], analysis)
        self.assertIn("Rótulo OK", w.lines[0].text())


if __name__ == "__main__":
    unittest.main()
