import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DebuggerCanonicalROIContractTests(unittest.TestCase):
    def test_structural_widget_uses_ssim_crop_test_as_display_source(self):
        source = (
            ROOT / "src" / "ui" / "widgets" / "silk_debugger.py"
        ).read_text(encoding="utf-8")

        self.assertIn('canonical_test = self._copy(detail.get("crop_test"))', source)
        self.assertIn("self.test_view = canonical_test", source)
        self.assertIn(
            "self.difference_view = self._raw_reconstruction(canonical_test, detail)",
            source,
        )
        self.assertIn("MESMA ROI DO LABORATÓRIO", source)
        self.assertIn("RECONSTRUÇÃO SOBRE A MESMA ROI", source)

    def test_structural_widget_does_not_use_numpy_arrays_with_boolean_or(self):
        source = (
            ROOT / "src" / "ui" / "widgets" / "silk_debugger.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn(
            'detail.get("match_mask_raw_coordinates") or detail.get("match_mask")',
            source,
        )
        self.assertNotIn(
            'detail.get("extra_mask_raw_coordinates") or detail.get("extra_mask")',
            source,
        )
        self.assertIn("def _first_array", source)

    def test_missing_widget_uses_same_ssim_roi_for_test_and_reconstruction(self):
        source = (
            ROOT / "src" / "ui" / "widgets" / "missing_debugger.py"
        ).read_text(encoding="utf-8")

        self.assertIn('canonical_test = self._copy(detail.get("crop_test"))', source)
        self.assertIn("self.test_view = canonical_test", source)
        self.assertIn(
            "self.reconstruction_view = self._rebuild_on_canonical_test(",
            source,
        )
        self.assertIn("MESMA ROI DO LABORATÓRIO", source)
        self.assertIn("DIVERGÊNCIA SOBRE A MESMA ROI", source)


if __name__ == "__main__":
    unittest.main()
