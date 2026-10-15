import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MissingComponentUIContractTests(unittest.TestCase):
    def test_removed_visual_debugger_not_installed_on_startup(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("install_missing_component_panel", source)
        self.assertIn("install_decision_panel(panel)", source)
        self.assertIn("install_roi_visual_alignment(SilkExpert, MissingComponentExpert)", source)
        self.assertIn("install_anomaly_memory_integration(MoEOrchestrator)", source)

    def test_debugger_explains_patch_expectation_and_localized_difference(self):
        source = (
            ROOT / "src" / "ui" / "widgets" / "missing_debugger.py"
        ).read_text(encoding="utf-8")
        self.assertIn("EXPECTATIVA DO PATCH", source)
        self.assertIn("GABARITO • MESMA ROI DO LABORATÓRIO", source)
        self.assertIn("TESTE • MESMA ROI DO LABORATÓRIO", source)
        self.assertIn("DIVERGÊNCIA SOBRE A MESMA ROI", source)
        self.assertIn("roi_patch_expectation", source)
        self.assertIn("missing_patch_type", source)
        self.assertIn("missing_residual_mean", source)
        self.assertIn("missing_residual_p90", source)
        self.assertIn("missing_edge_mismatch", source)
        self.assertIn("missing_direct_similarity", source)
        self.assertIn("missing_best_similarity", source)
        self.assertIn("missing_dual_scale_triggered", source)
        self.assertIn("missing_context_score", source)
        self.assertIn("missing_local_global_area_ratio", source)
        self.assertIn("PRESENÇA FÍSICA LOCAL + CONTEXTO", source)

    def test_panel_visibility_follows_active_engine(self):
        source = (
            ROOT / "src" / "ui" / "missing_component_panel.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"missing_expert.py" in active_engines', source)
        self.assertIn('"physical_absence_guard.py" in active_engines', source)
        self.assertIn("frame_missing.setVisible(active)", source)
        self.assertIn("frame_missing.setVisible(False)", source)
        self.assertIn("EXPECTATIVA DO PATCH", source)
        self.assertIn("PRESENÇA FÍSICA", source)

    def test_decision_model_keeps_missing_engine_label(self):
        source = (
            ROOT / "src" / "ui" / "decision_model.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"missing": "Presença do componente"', source)


if __name__ == "__main__":
    unittest.main()
