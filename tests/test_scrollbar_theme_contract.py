import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ScrollbarThemeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "src" / "ui" / "theme.py").read_text(
            encoding="utf-8"
        )

    def test_vertical_handle_uses_same_accent_for_all_interactions(self):
        self.assertIn("QScrollBar::handle:vertical,", self.source)
        self.assertIn("QScrollBar::handle:vertical:hover,", self.source)
        self.assertIn("QScrollBar::handle:vertical:pressed", self.source)
        self.assertIn("background-color: {ACCENT};", self.source)

    def test_horizontal_handle_uses_same_accent_for_all_interactions(self):
        self.assertIn("QScrollBar::handle:horizontal,", self.source)
        self.assertIn("QScrollBar::handle:horizontal:hover,", self.source)
        self.assertIn("QScrollBar::handle:horizontal:pressed", self.source)

    def test_old_gray_handle_color_was_removed(self):
        self.assertNotIn("QScrollBar::handle:vertical { background: #3a3a3a", self.source)
        self.assertNotIn("QScrollBar::handle:horizontal { background: #3a3a3a", self.source)

    def test_tracks_remain_dark_and_scrollbar_arrows_are_removed(self):
        self.assertIn("background-color: #151515;", self.source)
        self.assertIn("QScrollBar::add-line:vertical", self.source)
        self.assertIn("QScrollBar::sub-line:horizontal", self.source)
        self.assertIn("height: 0;", self.source)
        self.assertIn("width: 0;", self.source)


if __name__ == "__main__":
    unittest.main()
