import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DecisionInfluenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            ROOT / "src" / "ui" / "widgets" / "decision_influence.py"
        ).read_text(encoding="utf-8")

    def test_knn_line_separates_match_vote_weight_and_effect(self):
        self.assertIn('f"match {match:.0%} • voto {vote_label} • "', self.source)
        self.assertIn('f"peso {weight:.0%} • efeito {effect_text}"', self.source)

    def test_footer_exposes_the_actual_fusion_formula(self):
        self.assertIn('f"Fusão: físico {physical:.0%}×{physical_weight:.0%} + "', self.source)
        self.assertIn('f"KNN {knn_vote:.0%}×{knn_weight:.0%} = {final_score:.0%}"', self.source)

    def test_visual_has_separate_weight_indicator(self):
        self.assertIn('row.get("fusion_weight", 0.0)', self.source)
        self.assertIn('QColor("#f5c518")', self.source)
        self.assertIn("barra amarela fina = peso", self.source)
        self.assertIn('row.get("evidence_score", score)', self.source)
        self.assertIn("KNN = match", self.source)

    def test_participation_is_not_labeled_as_dominance(self):
        self.assertIn('return "DOMINANTE"', self.source)
        self.assertIn('return "PARTICIPA"', self.source)


if __name__ == "__main__":
    unittest.main()
