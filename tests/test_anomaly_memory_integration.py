import unittest

import numpy as np

from src.core.anomaly_memory_integration import install_anomaly_memory_integration
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


if __name__ == "__main__":
    unittest.main()
