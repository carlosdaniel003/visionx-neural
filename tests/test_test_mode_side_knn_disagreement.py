"""Contraexemplo real 22/10: CNN NG forte × KNN A/B OK sem sugestão seletiva."""
from __future__ import annotations

from copy import deepcopy
import unittest

from src.core.test_mode_side_knn_disagreement import (
    SCHEMA, test_mode_side_knn_disagreement,
)
from src.services.capture_debug_payload import decision_record
from src.ui.network_xp_debug import format_network_debug_report


def disputed_side():
    return {
        "verdict": "DEFEITO REAL", "is_defect": True, "lighting_mode": "SIDE",
        "confidence": 0.9998650550842285,
        "detail": {
            "recognition_route": "NEW_CNN",
            "cnn_v2_active": True,
            "cnn_v2_status": "INFERENCE_OK",
            "cnn_v2_checkpoint_verified": True,
            "cnn_v2_ng_score_uncalibrated": 0.9998650550842285,
            "knn_c": {
                "bank_status": "READY", "status": "REVISAO",
                "diagnostic_only": True, "auto_decision_authorized": False,
                "vote_A": "OK", "vote_B": "OK",
                "suggestion_A": "REVISAO", "suggestion_B": "REVISAO",
                "reason": "SEM_SUGESTAO_CONFIAVEL",
                "reason_A": "VIZINHOS_VISUAIS_DIVERGENTES",
                "reason_B": "SEM_CONTRAPROVA_VISUAL",
                "neighbors_A": [
                    {"path": "public/ok_archive/exemplo_1.png",
                     "label": "OK", "similarity": .99309},
                ],
                "neighbors_B": [
                    {"path": "public/ok_archive/exemplo_2.png",
                     "label": "OK", "similarity": .98159},
                ],
            },
        },
    }


class TestOnlyDisagreementTests(unittest.TestCase):
    def test_case_defers_only_in_test_mode_without_changing_verdict(self):
        analysis = disputed_side()
        before = deepcopy(analysis)
        event = test_mode_side_knn_disagreement(analysis, "Modo Teste")
        self.assertEqual(event["schema"], SCHEMA)
        self.assertEqual(event["status"], "DEFER_TO_TOP_MID")
        self.assertEqual(event["extra_lights"], ["TOP", "MID"])
        self.assertFalse(event["auto_ok_authorized"])
        self.assertFalse(event["xp_command_authorized"])
        self.assertEqual(analysis, before)  # pure; no mutation of CNN result

    def test_shadow_and_production_keep_existing_side_shortcut(self):
        analysis = disputed_side()
        for mode in ("Modo Produção", "Modo Sombra", "", "test", "Modo TESTE"):
            with self.subTest(mode=mode):
                self.assertIsNone(test_mode_side_knn_disagreement(analysis, mode))

    def test_no_trigger_for_unavailable_or_nonselective_knn_without_two_ok_votes(self):
        mutations = [
            ("bank_status", "LOADING"), ("bank_status", "UNAVAILABLE"),
            ("status", "SUGERE_OK"), ("status", "SUGERE_NG"),
            ("vote_A", "NG"), ("vote_B", None),
            ("suggestion_A", "SUGERE_OK"), ("suggestion_B", "SUGERE_NG"),
            ("diagnostic_only", False), ("auto_decision_authorized", True),
        ]
        for key,value in mutations:
            with self.subTest(key=key,value=value):
                image = disputed_side()
                image["detail"]["knn_c"][key] = value
                self.assertIsNone(test_mode_side_knn_disagreement(image,"Modo Teste"))

    def test_no_trigger_for_cnn_unverified_or_known_memory_or_ok_verdict(self):
        changes=[
            {"verdict": "FALHA FALSA", "is_defect": False},
            {"lighting_mode": "MID"},
            {"detail": {"recognition_route": "KNOWN_KNN"}},
            {"detail": {"cnn_v2_status": "NOT_READY"}},
            {"detail": {"cnn_v2_checkpoint_verified": False}},
            {"detail": {"cnn_v2_active": False}},
            {"detail": {"cnn_v2_ng_score_uncalibrated": .969}},
            {"detail": {"cnn_v2_ng_score_uncalibrated": float("nan")}},
        ]
        for change in changes:
            with self.subTest(change=str(change)):
                case=disputed_side()
                for key,value in change.items():
                    if key=="detail":
                        case["detail"].update(value)
                    else:
                        case[key]=value
                self.assertIsNone(test_mode_side_knn_disagreement(case,"Modo Teste"))

    def test_debug_contains_exact_neighbor_paths_and_revision_reasons(self):
        analysis = disputed_side()
        analysis["detail"]["test_mode_side_knn_disagreement"] = (
            test_mode_side_knn_disagreement(analysis, "Modo Teste")
        )
        record = decision_record(analysis, {"category": "FALTANDO"})
        self.assertEqual(record["knn_c"]["vote_A"], "OK")
        self.assertEqual(record["test_mode_side_knn_disagreement"]["status"],
                         "DEFER_TO_TOP_MID")
        raw = {
            "schema": "visionx.network_xp_debug.v1",
            "event_id": "d0ce716762e64f66a81eef73b9b09296",
            "decision": record,
        }
        text = format_network_debug_report(raw)
        for term in ("KNN ESTRATÉGIA C", "Voto A visual: OK",
                     "Voto B geométrico: OK",
                     "VIZINHOS_VISUAIS_DIVERGENTES",
                     "SEM_CONTRAPROVA_VISUAL",
                     "public/ok_archive/exemplo_1.png",
                     "public/ok_archive/exemplo_2.png",
                     "ARBITRAGEM EXPERIMENTAL", "DEFER_TO_TOP_MID"):
            self.assertIn(term, text)


if __name__ == "__main__":
    unittest.main()
