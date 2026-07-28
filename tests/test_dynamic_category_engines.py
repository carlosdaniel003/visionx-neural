import unittest

from src.core.anomaly_memory_integration import (
    _dynamic_fusion,
    is_adhesive_category,
    is_missing_category,
    routes_for_category,
)
from src.ui.decision_model import influence_rows


class FakeOrchestrator:
    DECISION_CUTOFF = 0.45
    DECISION_SCHEMA = "visionx.decision.v1"

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


def standard_detail():
    return {
        "silk_error_pct": 0.12,
        "semantic_loss": 0.30,
        "local_score": 0.40,
        "ctx_score": 0.30,
        "decision_threshold": 0.45,
        "ssim": 0.72,
        "pct_changed": 0.18,
    }


class DynamicCategoryEngineTests(unittest.TestCase):
    def setUp(self):
        self.orchestrator = FakeOrchestrator()
        self.no_memory = {
            "has_memory": False,
            "vote_defect": 0.5,
            "best_similarity": 0.0,
            "n_neighbors": 0,
        }

    def test_routes_are_category_specific(self):
        self.assertTrue(is_missing_category("FALTANDO"))
        self.assertTrue(is_adhesive_category("MUITO ADESIVO"))
        self.assertIn("missing", routes_for_category("FALTANDO"))
        self.assertNotIn("shift", routes_for_category("FALTANDO"))
        self.assertIn("shift", routes_for_category("MUITO ADESIVO"))
        self.assertNotIn("missing", routes_for_category("MUITO ADESIVO"))
        self.assertNotIn("missing", routes_for_category("INVERTIDO"))
        self.assertNotIn("shift", routes_for_category("INVERTIDO"))

    def test_missing_trace_contains_only_applicable_specialist(self):
        missing = {
            "missing_active": True,
            "missing_score": 0.91,
            "missing_tolerance": 0.42,
            "missing_is_defect": True,
            "missing_reason": "Componente ausente",
        }
        _, _, _, _, trace = _dynamic_fusion(
            self.orchestrator,
            standard_detail(),
            "FALTANDO",
            missing,
            self.no_memory,
        )
        ids = [engine["id"] for engine in trace["engines"]]
        self.assertEqual(
            ids,
            ["missing", "structural", "semantic", "texture", "knn"],
        )
        self.assertEqual(trace["dominant_engine"], "missing")
        self.assertNotIn("adhesive", ids)

    def test_adhesive_trace_does_not_show_missing_motor(self):
        detail = standard_detail()
        detail.update(
            {
                "shift_active": True,
                "adhesive_score": 0.94,
                "adhesive_tolerance": 0.32,
                "adhesive_is_defect": True,
                "adhesive_reason": "Adesivo excedente",
            }
        )
        _, _, _, _, trace = _dynamic_fusion(
            self.orchestrator,
            detail,
            "MUITO ADESIVO",
            None,
            self.no_memory,
        )
        ids = [engine["id"] for engine in trace["engines"]]
        self.assertEqual(
            ids,
            ["adhesive", "structural", "semantic", "texture", "knn"],
        )
        self.assertNotIn("missing", ids)

    def test_standard_trace_has_only_four_standard_engines(self):
        _, _, _, _, trace = _dynamic_fusion(
            self.orchestrator,
            standard_detail(),
            "INVERTIDO",
            None,
            self.no_memory,
        )
        ids = [engine["id"] for engine in trace["engines"]]
        self.assertEqual(ids, ["structural", "semantic", "texture", "knn"])

    def test_influence_rows_follow_trace_dynamically(self):
        missing = {
            "missing_active": True,
            "missing_score": 0.88,
            "missing_tolerance": 0.42,
            "missing_is_defect": True,
            "missing_reason": "Componente ausente",
        }
        _, _, _, _, trace = _dynamic_fusion(
            self.orchestrator,
            standard_detail(),
            "FALTANDO",
            missing,
            self.no_memory,
        )
        rows = influence_rows(trace)
        self.assertEqual([row["id"] for row in rows], [
            "missing",
            "structural",
            "semantic",
            "texture",
            "knn",
        ])
        self.assertEqual(rows[0]["label"], "Presença do componente")


if __name__ == "__main__":
    unittest.main()
