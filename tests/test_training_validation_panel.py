"""Contrato visual do painel sem conexão ao XP nem execução real de AOI."""
from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QFrame

from src.ui.training_validation_panel import (
    format_model_line, install_training_validation_panel,
)


class TrainingPanelUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        from pathlib import Path
        self.root = Path(self.folder.name)
        self.window = QWidget()
        self.window.content_layout = QVBoxLayout(self.window)
        self.window.network_debug_frame = QFrame()
        self.window.content_layout.addWidget(self.window.network_debug_frame)

    def tearDown(self):
        self.window.close()

    def test_button_always_visible_even_without_operational_buttons(self):
        ui = install_training_validation_panel(self.window, root=self.root)
        self.assertIs(ui, install_training_validation_panel(self.window, root=self.root))
        self.assertIn("treinamento", self.window.btn_training.text().lower())
        self.assertEqual(self.window.content_layout.indexOf(
            self.window.archive_training_frame
        ), 0)
        self.assertEqual(self.window.content_layout.indexOf(
            self.window.network_debug_frame
        ), 1)
        self.assertFalse(self.window.btn_cancel_training.isEnabled())
        self.assertEqual(self.window.training_progress_bar.value(), 0)

    def test_progress_includes_both_engines_and_can_show_counts(self):
        ui = install_training_validation_panel(self.window, root=self.root)
        ui._handle({"event": "started", "images": 212, "total": 414})
        ui._handle({"event": "progress", "done": 1, "total": 414,
                    "model": "KNN_MEMORIA", "path": "public/ok_archive/a.png",
                    "status": "PASSOU"})
        ui._handle({"event": "progress", "done": 2, "total": 414,
                    "model": "CNN_FALTANDO_V2", "path": "public/ok_archive/a.png",
                    "status": "ERRO_CLASSIFICACAO"})
        self.assertEqual(self.window.training_progress_bar.maximum(), 414)
        self.assertEqual(self.window.training_progress_bar.value(), 2)
        self.assertIn("1/1 corretas", ui.knn_metric.text())
        self.assertIn("1 pendências/erros", ui.cnn_metric.text())
        self.assertIn("a.png", ui.status.text())

    def test_summary_independently_reports_ok_and_ng(self):
        text = format_model_line("CNN_FALTANDO_V2", {
            "by_label": {
                "OK": {"passed": 190, "total": 195, "accuracy": .97436},
                "NG": {"passed": 17, "total": 17, "accuracy": 1.0},
            },
            "incorrect": 5, "no_coverage": 0, "invalid": 0,
            "out_of_scope": 10,
        })
        self.assertIn("OK 190/195", text)
        self.assertIn("NG 17/17", text)
        self.assertIn("Fora da CNN 10", text)


if __name__ == "__main__":
    unittest.main()
