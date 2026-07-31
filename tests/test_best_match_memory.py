import unittest
from unittest.mock import patch

import src.core.anomaly_memory_integration as fusion_module
import src.core.best_match_memory as memory_module
from src.core.anomaly_signature import VECTOR_SIZE
from src.core.experts.knn_expert import KNNExpert


class OrchestratorStub:
    DECISION_CUTOFF = 0.45
    DECISION_SCHEMA = "test"

    @staticmethod
    def _engine_entry(engine_id, label, active, triggered, raw, effective, threshold, summary):
        return {
            "id": engine_id,
            "label": label,
            "active": active,
            "triggered": triggered,
            "raw_score": raw,
            "effective_score": effective,
            "threshold": threshold,
            "selected": False,
            "final_influence": 0.0,
            "summary": summary,
        }


def sig(similarity=1.0):
    return {
        "schema": "visionx.anomaly.v1",
        "vector": [0.25] * VECTOR_SIZE,
        "test_similarity": similarity,
    }


def rec(label, similarity, index):
    return {
        "path": f"{label}_{index}.json",
        "anomaly_signature": sig(similarity),
    }


def compare(_query, stored):
    value = float(stored["test_similarity"])
    return value, {"value": value}


class MemorySelectionTests(unittest.TestCase):
    def analyze(self, ok_items, ng_items):
        expert = KNNExpert.__new__(KNNExpert)
        with patch.object(memory_module, "compare_anomaly_signatures", side_effect=compare):
            return memory_module._analyze_anomaly_memory_best_match(
                expert, sig(), ok_items, ng_items, 5, "categoria"
            )

    def test_single_closest_ng_is_not_overruled_by_many_ok(self):
        result = self.analyze(
            [rec("OK", 0.98, i) for i in range(99)],
            [rec("NG", 0.99, 0)],
        )
        self.assertTrue(result["has_memory"])
        self.assertEqual(result["best_match_label"], "NG")
        self.assertEqual(result["memory_score"], 1.0)
        self.assertEqual(result["memory_label_counts"], {"OK": 4, "NG": 1})
        self.assertFalse(result["quantity_influence"])

    def test_single_closest_ok_is_not_overruled_by_many_ng(self):
        result = self.analyze(
            [rec("OK", 0.99, 0)],
            [rec("NG", 0.98, i) for i in range(99)],
        )
        self.assertEqual(result["best_match_label"], "OK")
        self.assertEqual(result["memory_score"], 0.0)
        self.assertEqual(result["memory_label_counts"], {"OK": 1, "NG": 4})

    def test_weak_match_has_no_influence(self):
        result = self.analyze([rec("OK", 0.73, 0)], [rec("NG", 0.74, 0)])
        self.assertFalse(result["has_memory"])
        self.assertEqual(result["memory_score"], 0.5)

    def test_conflicting_near_tie_is_inconclusive(self):
        result = self.analyze([rec("OK", 0.989, 0)], [rec("NG", 0.990, 0)])
        self.assertTrue(result["conflicting_tie"])
        self.assertFalse(result["has_memory"])


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.fusion = memory_module._best_match_dynamic_fusion_factory(
            fusion_module._dynamic_fusion
        )
        self.orchestrator = OrchestratorStub()

    def run_fusion(self, label, similarity, legacy_vote, detail=None):
        knn = {
            "has_memory": True,
            "memory_available": True,
            "match_reliable": True,
            "best_match_label": label,
            "best_similarity": similarity,
            "vote_defect": legacy_vote,
            "n_neighbors": 5,
            "memory_mode": "anomaly",
            "memory_scope": "categoria",
        }
        return self.fusion(
            self.orchestrator, detail or {}, "DESLOCADO", None, knn
        )

    def test_strong_ng_ignores_old_low_vote(self):
        score, defect, confidence, reason, trace = self.run_fusion("NG", 0.99, 0.01)
        self.assertEqual(score, 1.0)
        self.assertTrue(defect)
        self.assertEqual(confidence, 0.99)
        self.assertEqual(trace["memory"]["memory_score"], 1.0)
        self.assertFalse(trace["memory"]["quantity_influence"])
        self.assertIn("quantidade de exemplos ignorada", reason)

    def test_strong_ok_ignores_old_high_vote(self):
        score, defect, confidence, _reason, trace = self.run_fusion(
            "OK", 0.99, 0.99, {"silk_error_pct": 0.20}
        )
        self.assertEqual(score, 0.0)
        self.assertFalse(defect)
        self.assertEqual(confidence, 0.99)
        self.assertEqual(trace["memory"]["memory_score"], 0.0)

    def test_intermediate_uses_best_label_with_partial_weight(self):
        score, defect, confidence, _reason, trace = self.run_fusion("NG", 0.80, 0.01)
        self.assertTrue(defect)
        self.assertGreater(score, self.orchestrator.DECISION_CUTOFF)
        self.assertLess(score, 1.0)
        self.assertLess(confidence, 0.99)
        self.assertEqual(trace["fusion_rule"], "best_match_intermediate")


class InstallationTests(unittest.TestCase):
    def test_installation_order(self):
        source = open("main.py", encoding="utf-8").read()
        self.assertLess(
            source.index("install_anomaly_memory_integration("),
            source.index("install_best_match_memory("),
        )


if __name__ == "__main__":
    unittest.main()
