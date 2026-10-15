"""KNN por semelhança (não exata) e sua participação na fusão CNN/ROI."""
from __future__ import annotations

from copy import deepcopy
from threading import RLock
from types import SimpleNamespace
import unittest

import numpy as np

from src.core.anomaly_signature import build_anomaly_signature
from src.core.knn_visual_corroboration import (
    evaluate_knn_visual_corroboration, apply_knn_visual_corroboration,
)
from src.ui.neural_influence_model import neural_influence_rows
from src.ui.neural_telemetry_model import memory_seen_state, memory_panel_text
from src.ui.production_confidence_gate import production_decision_policy

LIGHTS = ("SIDE", "TOP", "MID")


def sample(light):
    ref = np.full((96, 112, 3), 100, dtype=np.uint8)
    test = ref.copy()
    test[22:78, 36:86] = (12, 170, 12)
    return {"lighting_mode": light, "sample_image": ref, "test_image": test}


def signature(light, delta=0.0, category="FALTANDO"):
    item = sample(light)
    sig = build_anomaly_signature(
        item["sample_image"], item["test_image"], {},
        {"category": category, "lighting_mode": light},
    )
    sig["vector"][155] += delta
    return sig


def record(light, label, *, delta=0, category="FALTANDO"):
    return {
        "mode": "anomaly", "label": label, "folder_label": label,
        "lighting_mode": light, "category": category,
        "path": f"archive/{category}/{label}/{light}/{delta}",
        "anomaly_signature": signature(light, delta, category),
        "archive_persistent": True,
    }


def knn(ok=(), ng=()):
    return SimpleNamespace(
        _memory_lock=RLock(),
        signatures_ok=list(ok), signatures_ng=list(ng),
    )


def samples():
    return {light: sample(light) for light in LIGHTS}


def fused(*, physical="SUSPECT_ABSENCE", category="FALTANDO"):
    return {
        "verdict": "REVISÃO OBRIGATÓRIA" if physical else "FALHA FALSA",
        "is_defect": False, "multilight_final": True,
        "eligible_for_final_decision": bool(not physical),
        "production_review_required": bool(physical),
        "confidence": .5 if physical else .999,
        "detail": {
            "recognition_route": "NEW_CNN",
            "multilight_category": category,
            "cnn_v2_active": True, "cnn_v2_experimental": True,
            "cnn_v2_consensus": "OK",
            "cnn_v2_supervised_auto_eligible": not bool(physical),
            "physical_missing_veto": {"status": physical} if physical else {},
            "cnn_v2_light_diagnostics": {
                light: {"cnn_active": True,
                        "ng_score_uncalibrated": .001,
                        "verdict": "FALHA FALSA", "route": "NEW_CNN"}
                for light in LIGHTS
            },
            "decision_trace": {},
        },
    }


class ApproximateKNNConsensusTests(unittest.TestCase):
    def strong_ng_memory(self, lights=("SIDE", "TOP", "MID")):
        return knn(ng=[
            record(light, "NG", delta=variation)
            for light in lights for variation in (0.0, .015)
        ])

    def test_similar_ng_from_two_lights_corrob_in_ram_without_writing(self):
        memory = self.strong_ng_memory()
        before = len(memory.signatures_ng)
        result = evaluate_knn_visual_corroboration(samples(), "FALTANDO", memory)
        self.assertEqual(result["status"], "NG_CORROBORATED")
        self.assertEqual(result["strong_ng_lights"], list(LIGHTS))
        self.assertFalse(result["auto_decision_authorized"])
        self.assertEqual(len(memory.signatures_ng), before)
        for light in LIGHTS:
            self.assertEqual(result["lights"][light]["ng_support"], 2)
            self.assertGreater(result["lights"][light]["ng"]["best_similarity"], .93)

    def test_saved_same_photo_ten_times_is_not_ten_votes(self):
        repeated = [record("SIDE", "NG")] * 10
        memory = knn(ng=repeated)
        outcome = evaluate_knn_visual_corroboration(samples(), "FALTANDO", memory)
        self.assertEqual(outcome["lights"]["SIDE"]["ng"]["count"], 1)
        self.assertEqual(outcome["status"], "NO_CONCLUSION")

    def test_near_ok_prevents_ng_vote_even_with_two_ng(self):
        memory = self.strong_ng_memory()
        memory.signatures_ok.append(record("SIDE", "OK"))
        memory.signatures_ok.append(record("TOP", "OK"))
        outcome = evaluate_knn_visual_corroboration(samples(), "FALTANDO", memory)
        self.assertEqual(outcome["lights"]["SIDE"]["status"], "CONFLICT")
        self.assertEqual(outcome["lights"]["TOP"]["status"], "CONFLICT")
        self.assertEqual(outcome["status"], "CONFLICT")

    def test_light_and_category_filter_exclude_unrelated_ok_or_ng(self):
        memory = knn(ng=[
            record("TOP", "NG", delta=d, category="MUITO ADESIVO")
            for d in (0, .015)
        ])
        outcome = evaluate_knn_visual_corroboration(samples(), "FALTANDO", memory)
        self.assertEqual(outcome["strong_ng_lights"], [])
        self.assertEqual(outcome["lights"]["SIDE"]["ng"]["count"], 0)
        self.assertEqual(outcome["lights"]["TOP"]["ng"]["count"], 0)

    def test_cross_category_ng_can_be_found_with_stronger_support(self):
        memory = knn(ng=[
            record(light, "NG", category="FALTANDO", delta=delta)
            for light in ("SIDE", "TOP")
            for delta in (0.0, .015, .03)
        ])
        outcome = evaluate_knn_visual_corroboration(samples(), "DESLOCADO", memory)
        self.assertEqual(outcome["status"], "NG_CORROBORATED")
        for light in ("SIDE", "TOP"):
            self.assertTrue(outcome["lights"][light]["cross_category_ng"])
            self.assertEqual(outcome["lights"][light]["ng_support"], 3)

    def test_cross_category_similar_ok_blocks_false_ng(self):
        memory = knn(
            ng=[
                record(light, "NG", category="FALTANDO", delta=delta)
                for light in ("SIDE", "TOP")
                for delta in (0.0, .015, .03)
            ],
            ok=[
                record(light, "OK", category="EMBORCADO")
                for light in ("SIDE", "TOP")
            ],
        )
        outcome = evaluate_knn_visual_corroboration(
            samples(), "DESLOCADO", memory,
        )
        self.assertEqual(outcome["status"], "CONFLICT")
        self.assertEqual(outcome["strong_ng_lights"], [])
        self.assertEqual(outcome["lights"]["SIDE"]["status"], "CONFLICT")
        self.assertGreater(outcome["lights"]["SIDE"]["ok"]["count"], 0)

    def test_empty_or_corrupt_data_is_not_a_vote(self):
        item = samples()
        item["TOP"]["test_image"] = None
        item["MID"]["test_image"] = np.zeros((2, 2, 3), dtype=np.uint8)
        outcome = evaluate_knn_visual_corroboration(
            item, "FALTANDO", self.strong_ng_memory(),
        )
        self.assertEqual(outcome["status"], "NO_CONCLUSION")
        self.assertEqual(outcome["lights"]["TOP"]["status"], "UNAVAILABLE")

    def test_strong_ng_and_physical_veto_produce_ng_with_manual_gate(self):
        entry = apply_knn_visual_corroboration(
            fused(), samples(), "FALTANDO", self.strong_ng_memory(),
        )
        self.assertEqual(entry["verdict"], "DEFEITO REAL")
        self.assertTrue(entry["is_defect"])
        self.assertTrue(entry["production_review_required"])
        self.assertFalse(entry["eligible_for_final_decision"])
        self.assertFalse(production_decision_policy(entry)["auto_allowed"])
        self.assertEqual(
            entry["detail"]["knn_visual_corroboration"]["status"], "NG_CORROBORATED"
        )

    def test_ng_vizinhos_without_physical_cannot_approve_ng(self):
        entry = apply_knn_visual_corroboration(
            fused(physical=""), samples(), "FALTANDO", self.strong_ng_memory(),
        )
        self.assertEqual(entry["verdict"], "REVISÃO OBRIGATÓRIA")
        self.assertFalse(entry["is_defect"])
        self.assertFalse(production_decision_policy(entry)["auto_allowed"])

    def test_nonmatching_category_and_exact_knn_are_preserved(self):
        for category in ("MUITO ADESIVO", "DESCONHECIDO"):
            item = fused(category=category)
            before = deepcopy(item)
            self.assertEqual(
                apply_knn_visual_corroboration(item, samples(), category, self.strong_ng_memory()),
                before,
            )
        item = fused()
        item["detail"]["recognition_route"] = "KNOWN_KNN"
        before = deepcopy(item)
        self.assertEqual(
            apply_knn_visual_corroboration(item, samples(), "FALTANDO", self.strong_ng_memory()),
            before,
        )

    def test_fail_closed_even_when_widget_rewrites_verdict(self):
        entry = apply_knn_visual_corroboration(
            fused(physical=""), samples(), "FALTANDO", self.strong_ng_memory(),
        )
        entry["verdict"] = "FALHA FALSA"
        entry["production_review_required"] = False
        entry["eligible_for_final_decision"] = True
        entry["detail"]["operator_review_required"] = False
        entry["detail"]["cnn_v2_supervised_auto_eligible"] = True
        entry["detail"]["decision_trace"]["operator_review_required"] = False
        # A sonda física concluiu sem suspeita: quem deve vetar é a KNN.
        entry["detail"]["physical_missing_veto"] = {
            "status": "NO_STRONG_EVIDENCE"
        }
        policy = production_decision_policy(entry)
        self.assertFalse(policy["auto_allowed"])
        self.assertEqual(policy["reason"], "cnn_knn_visual_ng_disagreement")

    def test_dashboard_marks_cnn_contradicted_and_separates_knn(self):
        entry = apply_knn_visual_corroboration(
            fused(), samples(), "FALTANDO", self.strong_ng_memory(),
        )
        rows = neural_influence_rows(entry)
        cnn = [x for x in rows if x["id"].startswith("cnn_")]
        neighbor = [x for x in rows if x.get("knn_approximate")]
        self.assertEqual(len(cnn), 3)
        self.assertTrue(all(x["contradicted"] for x in cnn))
        self.assertGreaterEqual(len(neighbor), 1)
        self.assertTrue(all(x["fusion_weight"] == 0 for x in neighbor))
        self.assertTrue(all(x["ng_score_uncalibrated"] is None for x in neighbor))
        memory = memory_seen_state(entry)
        self.assertEqual(memory["status"], "NUNCA_VI")  # Sem par exato.
        self.assertTrue(memory["approximate_ng_seen"])
        self.assertEqual(memory["label"], "VI SEMELHANTE")
        self.assertIn("VI SEMELHANTE", memory_panel_text(entry)[0])


if __name__ == "__main__":
    unittest.main()
