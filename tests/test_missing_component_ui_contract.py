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

    def test_debugger_explains_patch_expectation_and_localized_difference(self):
        source = (
            ROOT / "src" / "ui" / "widgets" / "missing_debugger.py"
        ).read_text(encoding="utf-8")
        self.assertIn("EXPECTATIVA DO PATCH", source)
        self.assertIn("PATCH ESPERADO", source)
        self.assertIn("PATCH RECEBIDO", source)
        self.assertIn("SOMENTE PIXELS INCOMPATÍVEIS", source)
        self.assertIn("roi_patch_expectation", source)
        self.assertIn("missing_patch_type", source)
        self.assertIn("missing_residual_mean", source)
        self.assertIn("missing_residual_p90", source)
        self.assertIn("missing_edge_mismatch", source)
        self.assertIn("missing_direct_similarity", source)
        self.assertIn("missing_best_similarity", source)

    def test_panel_visibility_follows_active_engine(self):
        source = (
            ROOT / "src" / "ui" / "missing_component_panel.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"missing_expert.py" in active_engines', source)
        self.assertIn("frame_missing.setVisible(active)", source)
        self.assertIn("frame_missing.setVisible(False)", source)
        self.assertIn("EXPECTATIVA DO PATCH", source)

    def test_decision_model_keeps_missing_engine_label(self):
        source = (
            ROOT / "src" / "ui" / "decision_model.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"missing": "Presença do componente"', source)


if __name__ == "__main__":
    unittest.main()
