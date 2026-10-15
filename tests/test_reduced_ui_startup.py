"""Startup smoke for the reduced VisionX operator UI (PR #33 layout).

The panel must be constructible and the production footer installer must work
without the specialist carousel, its scroll_layout or its hidden widgets.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from types import MethodType
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QWidget
from src.ui.control_panel_ui import ControlPanelUI
from src.ui.decision_panel import install_decision_panel


ROOT = Path(__file__).resolve().parents[1]


class StartupWithoutSpecialistsTests(unittest.TestCase):
    def test_main_no_longer_invokes_or_imports_removed_visual_panels(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        parsed = ast.parse(source)
        removed = {"install_missing_component_panel", "install_inverted_face_panel"}
        imports = {
            alias.name
            for node in ast.walk(parsed)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        calls = {
            node.func.id
            for node in ast.walk(parsed)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertFalse(imports & removed, imports & removed)
        self.assertFalse(calls & removed, calls & removed)
        self.assertIn("install_decision_panel", calls)
        self.assertIn("install_inverted_face_integration", calls)
        self.assertIn("install_roi_visual_alignment", calls)

    def test_main_ui_and_decision_footer_show_without_removed_scroll_layout(self):
        app = QApplication.instance() or QApplication([])
        panel = QWidget()
        # O QWidget isolado não tem os slots operacionais do ControlPanel.
        # Eles ficam inertes: o smoke exercita a construção real do layout
        # sem iniciar receptor XP, CNN, discos ou comandos físicos.
        panel.start_monitoring = lambda: None
        panel.skip_image = lambda: None
        panel.save_label = lambda *args, **kwargs: None
        panel.set_ng_archive_enabled = lambda checked: None
        panel.set_ok_archive_enabled = lambda checked: None
        builder = ControlPanelUI()
        panel.ui_builder = builder
        try:
            builder.setup_ui(panel)
            self.assertFalse(hasattr(panel, "scroll_layout"))
            self.assertFalse(hasattr(panel, "telemetry_section"))
            self.assertEqual(len(builder.info_cards), 2)

            panel._update_reference_panel = MethodType(
                lambda self, analysis: None, panel
            )
            panel._reset_confidence_panel = MethodType(
                lambda self: None, panel
            )
            install_decision_panel(panel)
            panel.show()
            app.processEvents()
            self.assertTrue(panel.isVisible())
            self.assertIsNotNone(panel.lbl_verdict)
            self.assertIsNotNone(panel.frame_knn)
            self.assertEqual(panel.main_splitter.count(), 1)
        finally:
            panel.close()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
