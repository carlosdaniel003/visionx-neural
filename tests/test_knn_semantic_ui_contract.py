import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class KNNSemanticUIContractTests(unittest.TestCase):
    def test_knn_widget_displays_anomaly_memory_context(self):
        source = (
            ROOT / "src" / "ui" / "widgets" / "knn_spectrum.py"
        ).read_text(encoding="utf-8")
        self.assertIn("best_match_label", source)
        self.assertIn("Melhor anomalia:", source)
        self.assertIn("memory_mode", source)
        self.assertIn("memory_scope", source)
        self.assertIn("VOTO DOS RÓTULOS PARA DEFEITO NG", source)
        self.assertIn("SIMILARIDADE DA ANOMALIA COM A MEMÓRIA", source)
        self.assertIn("SIMILARIDADE DA IMAGEM COMPLETA (LEGADO)", source)

    def test_semantic_calibration_is_installed_before_panel_creation(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        panel_position = source.index("panel = ControlPanel()")
        self.assertLess(
            source.index("install_semantic_calibration(SemanticExpert)"),
            panel_position,
        )
        self.assertLess(
            source.index(
                "install_semantic_widget_calibration(SemanticDNAWidget)"
            ),
            panel_position,
        )
        self.assertLess(
            source.index("install_anomaly_memory_integration(MoEOrchestrator)"),
            panel_position,
        )
        self.assertLess(
            source.index("install_anomaly_learning(ControlPanel)"),
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

    def test_decision_summary_mentions_anomaly_neighbor_and_mode(self):
        source = (ROOT / "src" / "ui" / "decision_panel.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("best_match_label", source)
        self.assertIn("melhor vizinho de anomalia", source)
        self.assertIn("assinatura de anomalia", source)
        self.assertIn("MEMÓRIA DE ANOMALIAS", source)


if __name__ == "__main__":
    unittest.main()
