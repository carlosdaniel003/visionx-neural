import unittest

import numpy as np

from src.core.anomaly_memory_integration import (
    CROSS_CATEGORY_ABSENCE_GUARD_CATEGORIES,
    _dynamic_fusion,
    install_anomaly_memory_integration,
)
from src.core.anomaly_signature import valid_anomaly_signature


class _FakeKNN:
    def __init__(self):
        self.signature = None

    def analyze(self, *args, **kwargs):
        self.signature = kwargs.get("anomaly_signature")
        return {
            "has_memory": True,
            "vote_defect": 1.0,
            "best_similarity": 0.91,
            "n_neighbors": 1,
            "best_match_label": "NG",
            "memory_mode": "anomaly",
        }


class _FakeOrchestrator:
    def __init__(self):
        self.routing_table = {"Much Adhesive": ["shift", "semantic", "knn"]}
        self.experts = {"knn": _FakeKNN()}
        self.knn_was_in_original_route = None

    def inspect(
        self,
        full_gab,
        full_test,
        raw_anomalies,
        aoi_info,
        global_box_info,
        aoi_epicenters,
    ):
        self.knn_was_in_original_route = "knn" in self.routing_table[
            "Much Adhesive"
        ]
        return {
            "is_defect": True,
            "confidence": 0.8,
            "verdict": "DEFEITO REAL",
            "reason": "físico",
            "active_engines": ["shift_expert.py"],
            "bounding_box": None,
            "detail": {
                "shift_active": True,
                "adhesive_score": 0.90,
                "adhesive_tolerance": 0.32,
                "adhesive_is_defect": True,
                "adhesive_reason": "adesivo excedente",
                "semantic_delta": [0.0] * 128,
            },
        }

    def _master_fusion_score(self, shift, silk, semantic, ssim, knn):
        self.fusion_knn = knn
        trace = {
            "physical_score": 0.90,
            "cutoff": 0.45,
            "dominant_engine": "adhesive",
            "fusion_rule": "weighted_physical",
        }
        return 0.93, True, 0.90, "fusão", trace


class AnomalyMemoryIntegrationTests(unittest.TestCase):
    def test_signature_is_built_before_knn_and_original_knn_is_skipped(self):
        class Orchestrator(_FakeOrchestrator):
            pass

        install_anomaly_memory_integration(Orchestrator)
        orchestrator = Orchestrator()
        reference = np.full((40, 50, 3), 100, dtype=np.uint8)
        test = reference.copy()
        test[20:30, 25:35] = (20, 40, 140)

        result = orchestrator.inspect(
            reference,
            test,
            [],
            {"category": "Much Adhesive", "parts": "R1"},
            {},
            [(20, 15, 20, 20)],
        )

        self.assertFalse(orchestrator.knn_was_in_original_route)
        self.assertTrue(valid_anomaly_signature(orchestrator.experts["knn"].signature))
        self.assertEqual(result["detail"]["memory_mode"], "anomaly")
        self.assertEqual(result["detail"]["best_match_label"], "NG")
        self.assertIn("knn_expert.py", result["active_engines"])
        self.assertIn("anomaly_signature", result["detail"])




class _FusionOrchestrator:
    DECISION_CUTOFF = 0.45
    DECISION_SCHEMA = "test"

    @staticmethod
    def _engine_entry(
        engine_id,
        label,
        active,
        triggered,
        raw_score,
        effective_score,
        threshold,
        summary,
    ):
        return {
            "id": engine_id,
            "label": label,
            "active": bool(active),
            "triggered": bool(triggered),
            "raw_score": float(raw_score),
            "effective_score": float(effective_score),
            "threshold": float(threshold),
            "selected": False,
            "final_influence": 0.0,
            "summary": str(summary),
        }


class CrossCategoryAbsenceFusionTests(unittest.TestCase):
    def test_guard_categories_are_explicit_and_do_not_include_adhesive(self):
        self.assertEqual(
            CROSS_CATEGORY_ABSENCE_GUARD_CATEGORIES,
            frozenset({"EMBORCADO", "DESLOCADO", "INVERTIDO"}),
        )

    def test_base_fusion_never_allows_knn_veto_after_hard_absence(self):
        score, defect, confidence, _reason, trace = _dynamic_fusion(
            _FusionOrchestrator(),
            {
                "silk_error_pct": 0.54,
                "semantic_loss": 0.71,
            },
            "EMBORCADO",
            {
                "missing_active": True,
                "missing_is_defect": True,
                "missing_score": 0.85,
                "missing_tolerance": 0.36,
                "missing_reason": "ausência transversal confirmada",
                "missing_hard_absence": True,
                "missing_cross_category_guard": True,
            },
            {
                "has_memory": True,
                "vote_defect": 0.0,
                "best_similarity": 0.906,
                "n_neighbors": 1,
                "best_match_label": "OK",
                "operator_review_required": False,
            },
        )

        self.assertEqual(score, 1.0)
        self.assertTrue(defect)
        self.assertEqual(confidence, 0.99)
        self.assertEqual(trace["fusion_rule"], "missing_hard_absence")
        self.assertEqual(trace["dominant_engine"], "missing")
        self.assertEqual(trace["weights"], {"physical": 1.0, "knn": 0.0})
        self.assertTrue(trace["hard_missing_evidence"])
        self.assertTrue(trace["memory"]["suppressed_by_hard_missing"])


if __name__ == "__main__":
    unittest.main()
