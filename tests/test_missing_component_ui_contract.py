import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MissingComponentUIContractTests(unittest.TestCase):
    def test_missing_debugger_is_installed_before_decision_panel(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("install_missing_component_panel", source)
        self.assertLess(
            source.index("install_missing_component_panel(panel)"),
            source.index("install_decision_panel(panel)"),
        )

    def test_debugger_explains_roi_expectation_and_three_views(self):
        source = (
            ROOT / "src" / "ui" / "widgets" / "missing_debugger.py"
        ).read_text(encoding="utf-8")
        self.assertIn("EXPECTATIVA DA ROI", source)
        self.assertIn("CONTEÚDO RECEBIDO NA ROI", source)
        self.assertIn("QUEBRA DA EXPECTATIVA", source)
        self.assertIn("missing_expectation_mode", source)
        self.assertIn("missing_classification", source)
        self.assertIn("missing_direct_similarity", source)
        self.assertIn("missing_best_similarity", source)
        self.assertIn("missing_displacement_pixels", source)

    def test_panel_visibility_follows_active_engine(self):
        source = (
            ROOT / "src" / "ui" / "missing_component_panel.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"missing_expert.py" in active_engines', source)
        self.assertIn("frame_missing.setVisible(active)", source)
        self.assertIn("frame_missing.setVisible(False)", source)

    def test_decision_model_keeps_missing_engine_label(self):
        source = (
            ROOT / "src" / "ui" / "decision_model.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"missing": "Presença do componente"', source)


if __name__ == "__main__":
    unittest.main()
