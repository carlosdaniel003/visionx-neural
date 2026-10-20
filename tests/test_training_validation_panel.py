"""Contrato visual do painel sem conexão ao XP nem execução real de AOI."""
from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QFrame, QLabel

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
        ok_chip = self.window.archive_training_frame.findChild(
            QLabel, "archiveTrainingLegendOK"
        )
        ng_chip = self.window.archive_training_frame.findChild(
            QLabel, "archiveTrainingLegendNG"
        )
        self.assertIsNotNone(ok_chip)
        self.assertIsNotNone(ng_chip)
        self.assertIn("#4ade80", self.window.archive_training_frame.styleSheet())
        self.assertIn("#ff7777", self.window.archive_training_frame.styleSheet())

    def test_correction_button_removed_similarity_test_available(self):
        from PyQt6.QtWidgets import QPushButton
        ui = install_training_validation_panel(self.window, root=self.root)
        buttons = self.window.archive_training_frame.findChildren(QPushButton)
        labels = [button.text() for button in buttons]
        self.assertNotIn("Corrigir julgamento", labels)
        self.assertFalse(hasattr(ui, "correction_button"))
        self.assertEqual(ui.similarity_button.text(), "Testar KNN por similaridade")
        self.assertTrue(ui.similarity_button.isEnabled())
        self.assertFalse(hasattr(ui, "open_correction_mode"))

    def test_similarity_summary_shows_ng_missed_and_capture_groups(self):
        ui = install_training_validation_panel(self.window, root=self.root)
        ui._run_type = "similarity"
        ui._handle({"event": "started", "images": 4, "total": 4})
        ui._handle({
            "event": "progress", "done": 2, "total": 4,
            "path": "public/ng_archive/test_SIDE.png",
        })
        self.assertEqual(self.window.training_progress_bar.value(), 2)
        ui._finish_similarity(0, {
            "complete": True,
            "canceled": False,
            "json_path": "reports/x.json",
            "txt_path": "reports/x.txt",
            "results": {
                "by_label": {
                    "OK": {"total": 2, "correct": 2},
                    "NG": {"total": 2, "correct": 1},
                },
                "NG_as_OK": 1,
                "target_98pct_met": False,
            },
            "by_capture_cohort": {
                "side_only": {
                    "capture_groups": 2,
                    "results": {"correct": 1, "total": 2},
                },
                "three_lights": {
                    "capture_groups": 1,
                    "results": {"correct": 2, "total": 3},
                },
            },
            "by_lighting": {
                "SIDE": {"correct": 2, "total": 3},
                "TOP": {"correct": 1, "total": 1},
            },
        })
        self.assertIn("NG 1/2", ui.similarity_metric.text())
        self.assertIn("NG→OK 1", ui.similarity_metric.text())
        self.assertIn("Somente SIDE", ui.details.text())
        self.assertIn("SIDE + TOP + MID", ui.details.text())
        self.assertIn("revisar erros", ui.status.text())

    def test_progress_includes_both_engines_and_can_show_counts(self):
        ui = install_training_validation_panel(self.window, root=self.root)
        ui._handle({"event": "started", "images": 212, "total": 414})
        ui._handle({"event": "progress", "done": 1, "total": 414,
                    "model": "KNN_MEMORIA", "path": "public/ok_archive/a.png",
                    "expected": "OK", "status": "PASSOU"})
        ui._handle({"event": "progress", "done": 2, "total": 414,
                    "model": "CNN_FALTANDO_V2", "path": "public/ng_archive/b.png",
                    "expected": "NG", "status": "ERRO_CLASSIFICACAO"})
        self.assertEqual(self.window.training_progress_bar.maximum(), 414)
        self.assertEqual(self.window.training_progress_bar.value(), 2)
        self.assertIn("OK 1/1", ui.knn_metric.text())
        self.assertIn("NG 0/1", ui.cnn_metric.text())
        self.assertIn("1 pendências/erros", ui.cnn_metric.text())
        self.assertIn("#4ade80", ui.knn_metric.text())
        self.assertIn("#ff7777", ui.cnn_metric.text())
        self.assertIn("b.png", ui.status.text())
        self.assertIn("color:#ff7777", ui.status.text())

    def test_path_escaping_and_reset_does_not_confuse_ng_color(self):
        ui = install_training_validation_panel(self.window, root=self.root)
        ui._handle({"event": "started", "images": 1, "total": 2})
        ui._handle({"event": "progress", "done": 1, "total": 2,
                    "model": "KNN_MEMORIA",
                    "path": "public/ng_archive/<bad>.png",
                    "expected": "NG", "status": "PASSOU"})
        self.assertIn("NG 1/1", ui.knn_metric.text())
        self.assertIn("&lt;bad&gt;.png", ui.status.text())
        self.assertNotIn("<bad>", ui.status.text())
        ui._by_label = ui._empty_label_counts()
        ui._errors = {"KNN_MEMORIA": 0, "CNN_FALTANDO_V2": 0}
        self.assertIn("NG 0/0", ui._live_metric_html("KNN_MEMORIA"))

    def test_visual_similarity_has_no_day_selector(self):
        ui = install_training_validation_panel(self.window, root=self.root)
        self.assertFalse(hasattr(ui, "similarity_protocol"))
        self.assertIn("Testar KNN por similaridade", ui.similarity_button.text())
        self.assertIn("epicentro menor", ui.similarity_button.toolTip())
        self.assertIn("Sem seleção por data", (
            "Sem seleção por data"
            if not hasattr(ui, "similarity_protocol") else ""
        ))

    def test_shadow_candidate_in_similarity_report_is_visible(self):
        ui = install_training_validation_panel(self.window, root=self.root)
        ui._finish_similarity(0, {
            "complete": True, "canceled": False,
            "json_path": "reports/result.json",
            "txt_path": "reports/result.txt",
            "results": {
                "by_label": {
                    "OK": {"total": 3, "correct": 2},
                    "NG": {"total": 2, "correct": 1},
                }, "NG_as_OK": 1, "target_98pct_met": False,
            },
            "shadow_candidate": {
                "by_label": {
                    "OK": {"total": 3, "SUGERE_OK": 1},
                    "NG": {"total": 2, "SUGERE_NG": 1},
                },
                "NG_suggested_OK": 0,
                "OK_suggested_NG": 0,
                "reviews": 3,
            },
        })
        self.assertIn("OK 1/3", ui.shadow_candidate_metric.text())
        self.assertIn("NG 1/2", ui.shadow_candidate_metric.text())
        self.assertIn("NG→OK 0", ui.shadow_candidate_metric.text())
        self.assertIn("Revisão 3", ui.shadow_candidate_metric.text())

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
        self.assertIn('<span style="color:#4ade80', text)
        self.assertIn('<span style="color:#ff7777', text)


if __name__ == "__main__":
    unittest.main()
