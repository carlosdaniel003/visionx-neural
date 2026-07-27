import unittest

from src.ui.decision_model import (
    decision_summary,
    fusion_summary,
    influence_rows,
    memory_summary,
)


class DecisionModelTests(unittest.TestCase):
    def setUp(self):
        self.trace = {
            "final_score": 0.18,
            "physical_score": 0.76,
            "cutoff": 0.45,
            "confidence": 0.79,
            "dominant_engine": "knn",
            "fusion_rule": "memory_veto",
            "weights": {"physical": 0.0, "knn": 1.0},
            "memory": {
                "has_memory": True,
                "vote_defect": 0.18,
                "best_similarity": 0.96,
                "n_neighbors": 3,
                "role": "VETO DA MEMÓRIA",
            },
            "engines": [
                {
                    "id": "texture",
                    "label": "Laboratório de textura",
                    "active": True,
                    "triggered": True,
                    "raw_score": 0.76,
                    "effective_score": 0.76,
                    "threshold": 0.45,
                    "selected": True,
                    "final_influence": 0.0,
                    "summary": "diferença forte",
                },
                {
                    "id": "knn",
                    "label": "Memória local KNN",
                    "active": True,
                    "triggered": False,
                    "raw_score": 0.18,
                    "effective_score": 0.18,
                    "threshold": 0.45,
                    "selected": True,
                    "final_influence": 0.18,
                    "summary": "memória OK",
                },
            ],
        }

    def test_summary_exposes_score_cutoff_confidence_and_dominant_engine(self):
        text = decision_summary(self.trace)
        self.assertIn("Score final 18%", text)
        self.assertIn("corte 45%", text)
        self.assertIn("confiança 79%", text)
        self.assertIn("Memória local KNN", text)

    def test_fusion_summary_exposes_real_weights(self):
        text = fusion_summary(self.trace)
        self.assertIn("Memória substituiu", text)
        self.assertIn("peso físico 0%", text)
        self.assertIn("peso KNN 100%", text)

    def test_memory_summary_exposes_vote_similarity_and_neighbors(self):
        primary, role = memory_summary(self.trace)
        self.assertIn("Voto 18% NG", primary)
        self.assertIn("similaridade 96%", primary)
        self.assertIn("3 vizinho", primary)
        self.assertEqual(role, "VETO DA MEMÓRIA")

    def test_dominant_is_not_confused_with_participation(self):
        rows = influence_rows(self.trace)
        texture, knn = rows

        self.assertFalse(texture["selected"])
        self.assertFalse(texture["participates"])
        self.assertEqual(texture["fusion_weight"], 0.0)

        self.assertTrue(knn["selected"])
        self.assertTrue(knn["participates"])
        self.assertEqual(knn["fusion_weight"], 1.0)
        self.assertAlmostEqual(knn["score_contribution"], 0.18)
        self.assertAlmostEqual(knn["effect_vs_physical"], -0.58)

    def test_zero_ng_vote_can_still_have_thirty_percent_fusion_weight(self):
        trace = {
            "final_score": 0.686,
            "physical_score": 0.98,
            "cutoff": 0.45,
            "dominant_engine": "adhesive",
            "fusion_rule": "weighted_physical",
            "weights": {"physical": 0.70, "knn": 0.30},
            "memory": {
                "has_memory": True,
                "vote_defect": 0.0,
                "best_similarity": 0.68,
                "n_neighbors": 1,
                "role": "FUSÃO 70/30",
            },
            "engines": [
                {
                    "id": "adhesive",
                    "label": "Fluxo de adesivo",
                    "active": True,
                    "triggered": True,
                    "raw_score": 0.98,
                    "effective_score": 0.98,
                    "threshold": 0.32,
                },
                {
                    "id": "structural",
                    "label": "Comparador estrutural",
                    "active": True,
                    "triggered": True,
                    "raw_score": 0.59,
                    "effective_score": 0.85,
                    "threshold": 0.08,
                },
                {
                    "id": "knn",
                    "label": "Memória local KNN",
                    "active": True,
                    "triggered": False,
                    "raw_score": 0.0,
                    "effective_score": 0.0,
                    "threshold": 0.45,
                },
            ],
        }

        rows = {row["id"]: row for row in influence_rows(trace)}
        adhesive = rows["adhesive"]
        structural = rows["structural"]
        knn = rows["knn"]

        self.assertTrue(adhesive["selected"])
        self.assertAlmostEqual(adhesive["fusion_weight"], 0.70)
        self.assertAlmostEqual(adhesive["score_contribution"], 0.686)

        self.assertFalse(structural["participates"])
        self.assertEqual(structural["fusion_weight"], 0.0)

        self.assertFalse(knn["selected"])
        self.assertTrue(knn["participates"])
        self.assertAlmostEqual(knn["fusion_weight"], 0.30)
        self.assertAlmostEqual(knn["score_contribution"], 0.0)
        self.assertAlmostEqual(knn["effect_vs_physical"], -0.294)


if __name__ == "__main__":
    unittest.main()
