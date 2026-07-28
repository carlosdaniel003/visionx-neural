import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class InvertedFaceUIContractTests(unittest.TestCase):
    def test_engine_integration_is_installed_after_anomaly_memory(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("install_inverted_face_integration", source)
        self.assertLess(
            source.index("install_anomaly_memory_integration(MoEOrchestrator)"),
            source.index("install_inverted_face_integration(MoEOrchestrator)"),
        )

    def test_debugger_is_installed_before_decision_panel(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("install_inverted_face_panel", source)
        self.assertLess(
            source.index("install_inverted_face_panel(panel)"),
            source.index("install_decision_panel(panel)"),
        )

    def test_debugger_exposes_three_explainable_views(self):
        source = (
            ROOT / "src" / "ui" / "widgets" / "inverted_face_debugger.py"
        ).read_text(encoding="utf-8")
        self.assertIn("ASSINATURA DA FACE", source)
        self.assertIn("ASSINATURA ESPERADA", source)
        self.assertIn("ASSINATURA OBSERVADA", source)
        self.assertIn("EVIDÊNCIA DE INVERSÃO", source)
        self.assertIn("inverted_feature_loss", source)
        self.assertIn("inverted_topology_mismatch", source)
        self.assertIn("inverted_orientation_mismatch", source)
        self.assertIn("inverted_best_transform", source)

    def test_panel_visibility_follows_exclusive_engine(self):
        source = (
            ROOT / "src" / "ui" / "inverted_face_panel.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"inverted_expert.py" in active_engines', source)
        self.assertIn("frame_inverted.setVisible(active)", source)
        self.assertIn("frame_inverted.setVisible(False)", source)

    def test_decision_model_has_inverted_label(self):
        source = (
            ROOT / "src" / "ui" / "decision_model.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"inverted": "Assinatura da face"', source)


if __name__ == "__main__":
    unittest.main()
