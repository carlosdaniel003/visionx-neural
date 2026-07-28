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

    def test_debugger_uses_exact_three_view_reconstruction(self):
        source = (
            ROOT / "src" / "ui" / "widgets" / "missing_debugger.py"
        ).read_text(encoding="utf-8")
        self.assertIn("COMPONENTE ESPERADO", source)
        self.assertIn("PRESENÇA ENCONTRADA", source)
        self.assertIn("REGIÃO AUSENTE", source)
        self.assertIn("missing_structure_loss", source)
        self.assertIn("missing_presence_retention", source)

    def test_panel_visibility_follows_active_engine(self):
        source = (
            ROOT / "src" / "ui" / "missing_component_panel.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"missing_expert.py" in active_engines', source)
        self.assertIn("frame_missing.setVisible(active)", source)
        self.assertIn("frame_missing.setVisible(False)", source)

    def test_decision_model_has_missing_engine_label(self):
        source = (
            ROOT / "src" / "ui" / "decision_model.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"missing": "Presença do componente"', source)


if __name__ == "__main__":
    unittest.main()
