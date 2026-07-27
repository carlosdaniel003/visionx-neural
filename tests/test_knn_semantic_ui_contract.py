import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class KNNSemanticUIContractTests(unittest.TestCase):
    def test_knn_widget_displays_best_neighbor_label(self):
        source = (ROOT / "src" / "ui" / "widgets" / "knn_spectrum.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("best_match_label", source)
        self.assertIn("Melhor vizinho:", source)
        self.assertIn("VOTO DOS RÓTULOS PARA DEFEITO NG", source)
        self.assertIn("SIMILARIDADE VISUAL DO MELHOR VIZINHO", source)

    def test_semantic_calibration_is_installed_before_panel_creation(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        panel_position = source.index("panel = ControlPanel()")
        self.assertLess(
            source.index("install_semantic_calibration(SemanticExpert)"),
            panel_position,
        )
        self.assertLess(
            source.index("install_semantic_widget_calibration(SemanticDNAWidget)"),
            panel_position,
        )

    def test_semantic_telemetry_exposes_global_local_and_calibrated_scores(self):
        source = (
            ROOT / "src" / "core" / "experts" / "semantic_calibration.py"
        ).read_text(encoding="utf-8")
        self.assertIn("score={calibrated:.1%}", source)
        self.assertIn("global={global_loss:.1%}", source)
        self.assertIn("local={local_evidence:.1%}", source)
        self.assertIn("corte={threshold:.0%}", source)

    def test_decision_summary_mentions_best_neighbor_label(self):
        source = (ROOT / "src" / "ui" / "decision_panel.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("best_match_label", source)
        self.assertIn("melhor vizinho", source)


if __name__ == "__main__":
    unittest.main()
