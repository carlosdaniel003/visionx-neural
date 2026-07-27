import unittest

import cv2
import numpy as np

from src.core.anomaly_signature import build_anomaly_signature
from src.core.experts.knn_expert import KNNExpert


class KNNAnomalyMemoryTests(unittest.TestCase):
    def test_knn_uses_anomaly_signature_without_legacy_model(self):
        reference = np.full((72, 96, 3), 110, dtype=np.uint8)
        stored_test = reference.copy()
        query_test = reference.copy()
        cv2.rectangle(stored_test, (52, 40), (68, 57), (20, 40, 135), -1)
        cv2.rectangle(query_test, (53, 41), (69, 58), (22, 42, 138), -1)

        stored_signature = build_anomaly_signature(reference, stored_test, {}, {})
        query_signature = build_anomaly_signature(reference, query_test, {}, {})

        expert = KNNExpert.__new__(KNNExpert)
        expert.net = None
        expert.signatures_ok = []
        expert.signatures_ng = [
            {
                "part": "R307S",
                "category": "MUCHADHESIVE",
                "path": "memory_ng.json",
                "label": "NG",
                "mode": "anomaly",
                "anomaly_signature": stored_signature,
                "sig": None,
            }
        ]

        result = expert.analyze(
            reference,
            query_test,
            aoi_info={"parts": "R307S", "category": "Much Adhesive"},
            anomaly_signature=query_signature,
        )

        self.assertTrue(result["has_memory"])
        self.assertEqual(result["memory_mode"], "anomaly")
        self.assertEqual(result["best_match_label"], "NG")
        self.assertEqual(result["vote_defect"], 1.0)
        self.assertGreater(result["best_similarity"], 0.80)
        self.assertEqual(result["query_embedding"], [])
        self.assertIsNone(expert.net)

    def test_category_memory_can_match_another_component(self):
        reference = np.full((72, 96, 3), 110, dtype=np.uint8)
        test = reference.copy()
        cv2.rectangle(test, (52, 40), (68, 57), (20, 40, 135), -1)
        signature = build_anomaly_signature(reference, test, {}, {})

        expert = KNNExpert.__new__(KNNExpert)
        expert.net = None
        expert.signatures_ok = []
        expert.signatures_ng = [
            {
                "part": "R100",
                "category": "MUCHADHESIVE",
                "path": "memory_ng.json",
                "label": "NG",
                "mode": "anomaly",
                "anomaly_signature": signature,
                "sig": None,
            }
        ]

        result = expert.analyze(
            reference,
            test,
            aoi_info={"parts": "R999", "category": "Much Adhesive"},
            anomaly_signature=signature,
        )

        self.assertTrue(result["has_memory"])
        self.assertEqual(result["memory_scope"], "categoria")


if __name__ == "__main__":
    unittest.main()
