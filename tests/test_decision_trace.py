import unittest

from src.core.moe_orchestrator import MoEOrchestrator


class DecisionTraceTests(unittest.TestCase):
    def setUp(self):
        self.orchestrator = MoEOrchestrator.__new__(MoEOrchestrator)

    def test_adhesive_uses_adhesive_score_instead_of_legacy_shift_pct(self):
        adhesive = {
            "shift_active": True,
            "is_defect": True,
            "adhesive_is_defect": True,
            "adhesive_score": 0.72,
            "adhesive_tolerance": 0.32,
            "shift_pct": 0.001,
            "adhesive_reason": "Adesivo excedente",
        }

        final_score, is_defect, _, _, trace = (
            self.orchestrator._master_fusion_score(
                adhesive,
                None,
                None,
                None,
                None,
            )
        )

        self.assertTrue(is_defect)
        self.assertGreaterEqual(final_score, 0.80)
        self.assertEqual(trace["dominant_engine"], "adhesive")
        adhesive_row = next(
            item for item in trace["engines"] if item["id"] == "adhesive"
        )
        self.assertAlmostEqual(adhesive_row["raw_score"], 0.72)
        self.assertTrue(adhesive_row["selected"])

    def test_high_similarity_memory_veto_is_explicit(self):
        structural = {
            "is_defect": True,
            "silk_error_pct": 0.50,
            "tolerance": 0.08,
            "reason": "Divergência estrutural",
        }
        memory = {
            "has_memory": True,
            "vote_defect": 0.0,
            "best_similarity": 1.0,
            "n_neighbors": 2,
        }

        final_score, is_defect, _, _, trace = (
            self.orchestrator._master_fusion_score(
                None,
                structural,
                None,
                None,
                memory,
            )
        )

        self.assertFalse(is_defect)
        self.assertEqual(final_score, 0.0)
        self.assertEqual(trace["fusion_rule"], "memory_veto")
        self.assertEqual(trace["weights"], {"physical": 0.0, "knn": 1.0})
        self.assertEqual(trace["dominant_engine"], "knn")
        self.assertEqual(trace["memory"]["role"], "VETO DA MEMÓRIA")

    def test_weighted_fusion_records_both_contributions(self):
        texture = {
            "local_score": 0.70,
            "ctx_score": 0.50,
            "ssim": 0.30,
            "pct_changed": 0.40,
        }
        memory = {
            "has_memory": True,
            "vote_defect": 0.60,
            "best_similarity": 0.60,
            "n_neighbors": 3,
        }

        final_score, _, _, _, trace = self.orchestrator._master_fusion_score(
            None,
            None,
            None,
            texture,
            memory,
        )

        physical = 0.70 * 0.65 + 0.50 * 0.35
        expected = physical * 0.70 + 0.60 * 0.30
        self.assertAlmostEqual(final_score, expected)
        self.assertEqual(trace["fusion_rule"], "weighted_physical")
        self.assertEqual(trace["weights"], {"physical": 0.70, "knn": 0.30})

    def test_trace_lists_all_current_motors(self):
        _, _, _, _, trace = self.orchestrator._master_fusion_score(
            None,
            None,
            None,
            None,
            None,
        )
        ids = [item["id"] for item in trace["engines"]]
        self.assertEqual(
            ids,
            ["adhesive", "structural", "semantic", "texture", "knn"],
        )
        self.assertEqual(trace["schema"], "visionx.decision.v1")


if __name__ == "__main__":
    unittest.main()
