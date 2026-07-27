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

    def test_influence_rows_preserve_selected_and_final_influence(self):
        rows = influence_rows(self.trace)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["selected"])
        self.assertEqual(rows[0]["final_influence"], 0.0)
        self.assertEqual(rows[1]["final_influence"], 0.18)


if __name__ == "__main__":
    unittest.main()
