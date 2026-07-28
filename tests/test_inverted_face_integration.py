import unittest

from src.core.inverted_face_integration import (
    _fusion_with_inverted,
    is_inverted_category,
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
        "silk_error_pct": 0.21,
        "semantic_loss": 0.34,
        "local_score": 0.30,
        "ctx_score": 0.26,
        "decision_threshold": 0.45,
        "ssim": 0.70,
        "pct_changed": 0.12,
    }


class InvertedFaceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.orchestrator = FakeOrchestrator()
        self.inverted = {
            "inverted_active": True,
            "inverted_score": 0.92,
            "inverted_tolerance": 0.43,
            "inverted_is_defect": True,
            "inverted_reason": "Assinatura invertida forte",
        }
        self.no_memory = {
            "has_memory": False,
            "vote_defect": 0.5,
            "best_similarity": 0.0,
            "n_neighbors": 0,
        }

    def test_category_aliases_are_recognized(self):
        self.assertTrue(is_inverted_category("INVERTIDO"))
        self.assertTrue(is_inverted_category("Reverse"))
        self.assertTrue(is_inverted_category("Up Side Down"))
        self.assertFalse(is_inverted_category("FALTANDO"))
        self.assertFalse(is_inverted_category("MUITO ADESIVO"))

    def test_inverted_trace_contains_exclusive_engine_and_standard_engines(self):
        _, is_defect, _, _, trace = _fusion_with_inverted(
            self.orchestrator,
            standard_detail(),
            self.inverted,
            self.no_memory,
        )
        ids = [engine["id"] for engine in trace["engines"]]
        self.assertEqual(
            ids,
            ["inverted", "structural", "semantic", "texture", "knn"],
        )
        self.assertTrue(is_defect)
        self.assertEqual(trace["dominant_engine"], "inverted")
        self.assertNotIn("adhesive", ids)
        self.assertNotIn("missing", ids)

    def test_influence_rows_show_face_signature_as_dominant(self):
        _, _, _, _, trace = _fusion_with_inverted(
            self.orchestrator,
            standard_detail(),
            self.inverted,
            self.no_memory,
        )
        rows = influence_rows(trace)
        self.assertEqual(rows[0]["id"], "inverted")
        self.assertEqual(rows[0]["label"], "Assinatura da face")
        self.assertTrue(rows[0]["selected"])
        self.assertTrue(rows[0]["participates"])

    def test_knn_fusion_rules_are_preserved(self):
        memory = {
            "has_memory": True,
            "vote_defect": 0.80,
            "best_similarity": 0.78,
            "n_neighbors": 3,
            "best_match_label": "NG",
            "memory_mode": "anomaly",
            "memory_scope": "category",
        }
        final_score, is_defect, _, _, trace = _fusion_with_inverted(
            self.orchestrator,
            standard_detail(),
            self.inverted,
            memory,
        )
        self.assertTrue(is_defect)
        self.assertEqual(trace["fusion_rule"], "weighted_physical")
        self.assertAlmostEqual(trace["weights"]["physical"], 0.70)
        self.assertAlmostEqual(trace["weights"]["knn"], 0.30)
        self.assertAlmostEqual(final_score, 0.90 * 0.70 + 0.80 * 0.30)


if __name__ == "__main__":
    unittest.main()
