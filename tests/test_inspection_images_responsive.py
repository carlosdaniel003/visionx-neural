import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication

from src.ui.control_panel_ui import _ResponsivePixmapLabel


ROOT = Path(__file__).resolve().parents[1]


class ResponsiveInspectionImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_pixmap_is_rescaled_when_its_own_viewport_changes(self):
        label = _ResponsivePixmapLabel("Sem Sinal")
        label.resize(320, 180)
        label.show()

        source = QPixmap(640, 360)
        source.fill()
        label.setPixmap(source)
        self.app.processEvents()

        first = label.pixmap()
        self.assertIsNotNone(first)
        self.assertLessEqual(first.width(), label.contentsRect().width())
        self.assertLessEqual(first.height(), label.contentsRect().height())

        # Simula mover a divisória do QSplitter sem redimensionar a janela raiz.
        label.resize(180, 90)
        self.app.processEvents()

        resized = label.pixmap()
        self.assertIsNotNone(resized)
        self.assertLessEqual(resized.width(), label.contentsRect().width())
        self.assertLessEqual(resized.height(), label.contentsRect().height())

    def test_source_pixmap_is_preserved_for_future_resizes(self):
        label = _ResponsivePixmapLabel()
        source = QPixmap(800, 400)
        source.fill()

        label.resize(200, 100)
        label.setPixmap(source)
        label.resize(400, 200)
        self.app.processEvents()

        self.assertEqual(label._source_pixmap.size(), source.size())
        self.assertLessEqual(label.pixmap().width(), 400)
        self.assertLessEqual(label.pixmap().height(), 200)


class ResponsiveInspectionSourceContractTests(unittest.TestCase):
    def test_four_normal_inspection_views_use_responsive_labels(self):
        source = (
            ROOT / "src" / "ui" / "control_panel_ui.py"
        ).read_text(encoding="utf-8")

        for declaration in (
            'window.lbl_sample = _ResponsivePixmapLabel("Sem Sinal")',
            'window.lbl_sample_focus = _ResponsivePixmapLabel("Sem Foco")',
            'window.lbl_ng = _ResponsivePixmapLabel("Sem Sinal")',
            'window.lbl_ng_focus = _ResponsivePixmapLabel("Sem Foco")',
        ):
            self.assertIn(declaration, source)

    def test_main_stage_splitter_is_not_user_draggable(self):
        source = (
            ROOT / "src" / "ui" / "control_panel_ui.py"
        ).read_text(encoding="utf-8")

        self.assertIn("window.main_splitter.setHandleWidth(0)", source)
        self.assertIn("splitter_handle.setEnabled(False)", source)
        self.assertIn("window.images_section.setMinimumHeight(560)", source)
        self.assertIn("window.main_splitter.setMinimumHeight(1000)", source)

    def test_wide_image_section_uses_bounded_responsive_width(self):
        source = (
            ROOT / "src" / "ui" / "control_panel_ui.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "image_width = min(520, max(420, int(width * 0.28)))",
            source,
        )
        self.assertIn("window.main_splitter.setMinimumHeight(560)", source)

    def test_controller_passes_original_pixmaps_to_responsive_views(self):
        source = (
            ROOT / "src" / "ui" / "control_panel.py"
        ).read_text(encoding="utf-8")

        self.assertIn("self.lbl_sample.setPixmap(px_sample)", source)
        self.assertIn("self.lbl_ng.setPixmap(px_ng)", source)
        self.assertNotIn(
            "self.lbl_sample.setPixmap(px_sample.scaled(",
            source,
        )
        self.assertNotIn(
            "self.lbl_ng.setPixmap(px_ng.scaled(",
            source,
        )


if __name__ == "__main__":
    unittest.main()
