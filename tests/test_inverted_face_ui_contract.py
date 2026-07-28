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

    def test_debugger_exposes_witness_views_and_metrics(self):
        source = (
            ROOT / "src" / "ui" / "widgets" / "inverted_face_debugger.py"
        ).read_text(encoding="utf-8")
        self.assertIn("MARCA TESTEMUNHA", source)
        self.assertIn("MARCA ESPERADA", source)
        self.assertIn("MARCA OBSERVADA", source)
        self.assertIn("RETIDA / PERDIDA / NOVA", source)
        self.assertIn("inverted_witness_retention", source)
        self.assertIn("inverted_witness_loss", source)
        self.assertIn("inverted_relocation_similarity", source)
        self.assertIn("inverted_relocation_dx", source)
        self.assertIn("inverted_feature_loss", source)
        self.assertIn("inverted_orientation_mismatch", source)
        self.assertIn("inverted_best_transform", source)

    def test_panel_visibility_follows_exclusive_engine(self):
        source = (
            ROOT / "src" / "ui" / "inverted_face_panel.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"inverted_expert.py" in active_engines', source)
        self.assertIn("frame_inverted.setVisible(active)", source)
        self.assertIn("frame_inverted.setVisible(False)", source)
        self.assertIn("MARCA TESTEMUNHA • MOTOR INVERTIDO", source)

    def test_decision_model_keeps_inverted_engine_label(self):
        source = (
            ROOT / "src" / "ui" / "decision_model.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"inverted": "Assinatura da face"', source)


if __name__ == "__main__":
    unittest.main()
