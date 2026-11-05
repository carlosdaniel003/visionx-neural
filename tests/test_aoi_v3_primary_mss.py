"""Regressões de troca V2→V3 primária e paridade visual MSS/XP."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.core.neural.aoi_quality_cnn_v3_primary import AOIQualityV3Primary
from src.core.neural.aoi_v3_trial_contract import POINTER_SCHEMA, SNAPSHOT_SHA256
from src.services.archive_visual_presence import ArchiveVisualPresenceIndex
from src.services.capture_debug_payload import decision_record
from src.ui.archive_visual_presence_feedback import submit_local_mss_archive_presence
from src.ui.production_confidence_gate import production_decision_policy


def vote(light: str, score: float) -> dict:
    return {
        "schema": POINTER_SCHEMA, "status": "READY",
        "epoch": 3, "light": light,
        "checkpoint_sha256": SNAPSHOT_SHA256,
        "ng_score_uncalibrated": score,
        "predicted": "NG" if score >= .5 else "OK",
        "ng_veto": score >= .5,
        "automatic_ok_from_v3_authorized": False,
    }


class Engine:
    def __init__(self, score):
        self.score = score
        self.calls = []

    def predict(self, reference, test, mode):
        self.calls.append(mode)
        if isinstance(self.score, Exception):
            raise self.score
        if self.score is None:
            return None
        return vote(mode, self.score)


class PrimaryRoutingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((64, 80, 3), dtype=np.uint8)

    def test_v3_is_primary_for_both_classes_without_v2_fields(self):
        for score, expected in ((.02, "FALHA FALSA"), (.98, "DEFEITO REAL")):
            with self.subTest(score=score):
                e = Engine(score)
                result = AOIQualityV3Primary(engine=e).inspect(
                    self.image, self.image, "SIDE",
                )
                self.assertEqual(result["verdict"], expected)
                self.assertTrue(result["detail"]["cnn_v3_primary"])
                self.assertNotIn("cnn_v2_active", result["detail"])
                self.assertNotIn("faltando_cnn_v2.py", result["active_engines"])
                self.assertEqual(result["detail"]["aoi_v3_trial"]["status"], "READY")
                self.assertFalse(result["detail"]["cnn_v3_automatic_ok_authorized"])
                self.assertEqual(e.calls, ["SIDE"])
                debug = decision_record(result, {"category": "FALTANDO"})
                self.assertTrue(debug["cnn_v3"]["active"])
                self.assertEqual(debug["cnn_v3"]["role"], "PRIMARY")
                self.assertFalse(debug["cnn_v2"]["active"])

    def test_missing_or_corrupt_v3_fails_closed_no_legacy_fallback(self):
        for value in (None, ValueError("checkpoint inválido")):
            with self.subTest(value=value):
                result = AOIQualityV3Primary(engine=Engine(value)).inspect(
                    self.image, self.image, "SIDE",
                )
                self.assertEqual(result["verdict"], "REVISÃO OBRIGATÓRIA")
                self.assertEqual(result["detail"]["aoi_v3_trial"]["status"], "ERROR")
                self.assertNotIn("cnn_v2_active", result["detail"])

    def test_primary_ok_cannot_release_unverified_product(self):
        result = AOIQualityV3Primary(engine=Engine(.01)).inspect(
            self.image, self.image, "SIDE",
        )
        result["multilight_final"] = True
        result["detail"]["multilight_final"] = True
        result["detail"]["eligible_for_final_decision"] = True
        result["eligible_for_final_decision"] = True
        result["detail"]["production_continuous_auto"] = True
        result["detail"]["production_binary_auto"] = True
        result["detail"]["recognition_route"] = "NEW_CNN"
        policy = production_decision_policy(result)
        self.assertFalse(policy["auto_allowed"])
        self.assertEqual(policy["proposed_decision"], "")
        self.assertEqual(
            policy["reason"], "cnn_v3_primary_ok_requires_independent_confirmation",
        )

    def test_legacy_production_mode_cannot_bypass_v3_gate(self):
        result = AOIQualityV3Primary(engine=Engine(.03)).inspect(
            self.image, self.image, "SIDE",
        )
        result["detail"]["production_continuous_auto"] = True
        result["detail"]["production_binary_auto"] = False
        result["detail"]["recognition_route"] = "NEW_CNN"
        policy = production_decision_policy(result)
        self.assertTrue(policy["operational_error"])
        self.assertFalse(policy["auto_allowed"])
        self.assertEqual(
            policy["reason"], "cnn_v3_primary_requires_verified_binary_gate",
        )

    def test_ui_never_labels_v3_primary_as_v2(self):
        from src.ui.neural_telemetry_model import cnn_panel_text, neural_summary
        result = AOIQualityV3Primary(engine=Engine(.01)).inspect(
            self.image, self.image, "SIDE",
        )
        result["detail"]["recognition_route"] = "NEW_CNN"
        summary = neural_summary(result)
        self.assertEqual(summary["cnn_model_name"], "CNN AOI V3")
        self.assertEqual(summary["cnn_ng_score"], .01)
        self.assertIn("CNN AOI V3", cnn_panel_text(result)["header"])
        self.assertNotIn("v2", cnn_panel_text(result)["header"])

    def test_primary_ng_still_respects_final_capture(self):
        result = AOIQualityV3Primary(engine=Engine(.98)).inspect(
            self.image, self.image, "SIDE",
        )
        result["detail"]["production_continuous_auto"] = True
        result["detail"]["production_binary_auto"] = True
        result["detail"]["recognition_route"] = "NEW_CNN"
        # Uma só SIDE antes de concluir o ciclo não dispara o XP.
        p = production_decision_policy(result)
        self.assertFalse(p["auto_allowed"])
        self.assertTrue(p["operational_error"])

    def test_v3_runtime_does_not_queue_v2_retraining(self):
        from src.services.neural_online_learning import eligible_online_case

        task = {
            "aoi_info": {"category": "FALTANDO"},
            "analysis": {"detail": {
                "recognition_route": "NEW_CNN",
                "cnn_v3_primary": True,
            }},
            "source": "button", "label": "OK",
        }
        self.assertFalse(eligible_online_case(task))
        # Mesmo com rota KNOWN_KNN (correção manual recente), o
        # contexto da execução V3 impede lançamento de um treinador V2.
        task["analysis"] = {"detail": {"recognition_route": "KNOWN_KNN"}}
        task["reinspection_of_judged_event"] = True
        task["v3_primary_runtime"] = True
        self.assertFalse(eligible_online_case(task))
        # A API de treinamento V2 isolado permanece testável sem que
        # o ponteiro de outra instalação contamine a elegibilidade.
        task.pop("v3_primary_runtime")
        self.assertTrue(eligible_online_case(task))

    def test_main_installs_primary_instead_of_v2_predictor(self):
        main = (Path(__file__).resolve().parents[1] / "main.py").read_text(
            encoding="utf-8",
        )
        self.assertIn("predictor=AOIQualityV3Primary()", main)
        self.assertNotIn("install_faltando_cnn_live(MoEOrchestrator)\n", main)


class LocalMSSArchiveTests(unittest.TestCase):
    def test_mss_submits_original_full_frame_against_three_archive_lights(self):
        class Collector:
            def inspect(self, analysis, image, category, event_id, *,
                        frames, candidate_paths):
                self.args = (
                    analysis, image, category, event_id, frames, candidate_paths,
                )
                return True

        collector = Collector()
        original = np.full((840, 1165, 3), 55, dtype=np.uint8)
        a = {
            "verdict": "DEFEITO REAL",
            "detail": {
                "recognition_route": "NEW_CNN",
                "knn_c": {"neighbors_B": [{
                    "path": "public/ok_archive/2026-10-02_1029_FALTANDO.png",
                }]},
            },
        }
        panel = SimpleNamespace(
            local_capture_debug_event_id="mss-123",
            capture_debug_last_record={
                "event_id": "mss-123", "source": "local_mss",
            },
            capture_debug_last_image=original.copy(),
            capture_debug_last_image_event_id="mss-123",
            _side_archive_presence_feedback=collector,
        )
        self.assertTrue(submit_local_mss_archive_presence(panel, a, "FALTANDO"))
        analysis, image, cat, event, frames, candidates = collector.args
        self.assertEqual(candidates, [
            "public/ok_archive/2026-10-02_1029_FALTANDO.png"
        ])
        self.assertIs(analysis, a)
        self.assertEqual(cat, "FALTANDO")
        self.assertEqual(event, "mss-123")
        self.assertEqual(set(frames), {"SIDE", "TOP", "MID"})
        self.assertTrue(np.array_equal(image, original))
        for frame in frames.values():
            self.assertTrue(np.array_equal(frame, original))

    def test_mss_exact_archive_frame_does_not_forge_aoi_exact_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ok = root / "ok_archive"
            ng = root / "ng_archive"
            ok.mkdir()
            ng.mkdir()
            frame = np.full((840, 1165, 3), 90, dtype=np.uint8)
            cv2.line(frame, (20, 20), (1000, 720), (200, 20, 20), 15)
            self.assertTrue(cv2.imwrite(
                str(ok / "2026-10-21_1441_FALTANDO_TOP.png"), frame,
            ))
            index = ArchiveVisualPresenceIndex(ok_dir=ok, ng_dir=ng)
            results = {
                light: index.inspect(frame, "FALTANDO", light)
                for light in ("SIDE", "TOP", "MID")
            }
            self.assertEqual(results["TOP"]["status"], "EXACT_FRAME")
            self.assertEqual(results["TOP"]["label"], "OK")
            self.assertFalse(results["TOP"]["decision_authorized"])
            # O mesmo PNG de TOP não cria um par KNN verificado na SIDE.
            self.assertNotEqual(results["SIDE"]["status"], "EXACT_FRAME")
            self.assertNotIn("recognition_match", results["TOP"])


if __name__ == "__main__":
    unittest.main()
