import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class IconographyLayoutRegressionTests(unittest.TestCase):
    def test_qt_layout_take_at_receives_required_index(self):
        source = (ROOT / "src" / "ui" / "iconography.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("layout.takeAt(0)", source)
        self.assertNotIn("layout.takeAt()", source)


if __name__ == "__main__":
    unittest.main()
