import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SemanticMemoryUIContractTests(unittest.TestCase):
    def test_semantic_roi_extension_is_installed_after_calibration(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("install_semantic_roi_extension", source)
        self.assertLess(
            source.index("install_semantic_calibration(SemanticExpert)"),
            source.index("install_semantic_roi_extension(SemanticExpert)"),
        )
        self.assertIn("install_semantic_roi_widget(SemanticDNAWidget)", source)

    def test_semantic_extension_forbids_full_image_fallback(self):
        source = (
            ROOT / "src" / "core" / "semantic_roi_extension.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"full_image_fallback"] = False', source)
        self.assertIn('"semantic_scope"] = "epicenter_roi"', source)
        self.assertIn("ROI do epicentro ausente ou inválida", source)
        self.assertIn("escopo=ROI DO EPICENTRO", source)

    def test_knn_extension_has_no_global_part_or_legacy_fallback(self):
        source = (
            ROOT / "src" / "core" / "strict_category_memory.py"
        ).read_text(encoding="utf-8")
        self.assertIn('record.get("mode") == "anomaly"', source)
        self.assertIn("canonical_memory_category", source)
        self.assertIn("memory_filter_strict", source)
        self.assertNotIn("legacy_image", source)
        self.assertNotIn("target_part", source)
        self.assertNotIn('"global"', source)

    def test_knn_ui_exposes_current_category(self):
        source = (
            ROOT / "src" / "ui" / "strict_category_memory_ui.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Sem JSON de anomalia para", source)
        self.assertIn("memory_category", source)
        self.assertIn("MUITO ADESIVO", source)
        main_source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn(
            "install_strict_category_memory_ui(KNNSpectrumWidget)",
            main_source,
        )


if __name__ == "__main__":
    unittest.main()
