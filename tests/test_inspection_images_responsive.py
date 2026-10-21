import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QStackedWidget, QWidget

from src.ui.control_panel_ui import ControlPanelUI, _ResponsivePixmapLabel


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


class MultilightScrollStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_same_breakpoint_resize_does_not_rebuild_layout(self):
        builder = ControlPanelUI()
        builder._active_profile_name = "wide"
        # Falha se a função tentar acessar geometria ou grids do painel.
        builder.apply_layout_profile(object(), 1920)

    def test_repeated_light_updates_do_not_force_multilight_relayout(self):
        builder = ControlPanelUI()
        normal, multilight = QWidget(), QWidget()
        stack = QStackedWidget()
        stack.addWidget(normal)
        stack.addWidget(multilight)
        stack.setCurrentWidget(multilight)
        window = SimpleNamespace(
            _multilight_inspection_mode=True,
            _adhesive_inspection_mode=True,
            inspection_view_stack=stack,
            normal_inspection_view=normal,
            adhesive_multilight_view=multilight,
        )
        builder.apply_layout_profile = Mock()
        for _ in ("SIDE", "TOP", "MID"):
            builder.set_multilight_inspection_mode(window, True)
        builder.apply_layout_profile.assert_not_called()
        self.assertIs(stack.currentWidget(), multilight)

    def test_vertical_scroll_reserves_stable_viewport_width(self):
        source = (ROOT / "src" / "ui" / "control_panel_ui.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "window.root_scroll.setVerticalScrollBarPolicy("
            "Qt.ScrollBarPolicy.ScrollBarAlwaysOn)",
            source,
        )


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
        self.assertIn("window.main_splitter.addWidget(window.images_section)", source)
        self.assertNotIn("window.main_splitter.addWidget(window.telemetry_section)", source)

    def test_responsive_images_use_entire_available_width(self):
        source = (
            ROOT / "src" / "ui" / "control_panel_ui.py"
        ).read_text(encoding="utf-8")
        self.assertIn("window.images_section.setMaximumWidth(16777215)", source)
        self.assertIn(
            "window.images_section.setMinimumHeight(",
            source,
        )
        self.assertNotIn("image_width = min(520,", source)

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
