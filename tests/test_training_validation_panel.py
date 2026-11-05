"""Regressões da auditoria do acervo: UX clara, responsiva e sem XP."""
from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import (
    QApplication, QFrame, QLabel, QPushButton, QVBoxLayout, QWidget,
)

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
        self.window.deleteLater()

    def test_audit_is_above_debug_with_clear_label_and_semantics(self):
        ui = install_training_validation_panel(self.window, root=self.root)
        self.assertIs(ui, install_training_validation_panel(self.window, root=self.root))
        self.assertEqual(self.window.btn_training.text(), "Validar acervo OK/NG")
        self.assertIn("NÃO treina", self.window.btn_training.toolTip())
        self.assertIn("diagnóstico", ui.shadow_button.text())
        self.assertEqual(
            self.window.content_layout.indexOf(self.window.archive_training_frame), 0,
        )
        self.assertEqual(
            self.window.content_layout.indexOf(self.window.network_debug_frame), 1,
        )
        self.assertFalse(self.window.btn_cancel_training.isEnabled())
        self.assertTrue(ui.cancel_button.isHidden())
        self.assertEqual(self.window.training_progress_bar.value(), 0)
        self.assertEqual(ui.status.property("state"), "idle")
        self.assertIn("não verificado", ui.status.text())

    def test_unused_experimental_controls_removed_from_main_panel(self):
        ui = install_training_validation_panel(self.window, root=self.root)
        labels = [
            button.text()
            for button in self.window.archive_training_frame.findChildren(QPushButton)
        ]
        self.assertEqual(len(labels), 3)
        for unwanted in (
            "Testar KNN por similaridade",
            "Replay operacional por evento",
            "Comparar geometria de bordas",
            "Corrigir julgamento",
        ):
            self.assertFalse(any(unwanted in label for label in labels))
        for attr in (
            "geometry_experiment_box", "similarity_button",
            "event_replay_button", "correction_button",
        ):
            self.assertFalse(hasattr(ui, attr))
        self.assertFalse(hasattr(ui, "start_similarity"))
        self.assertFalse(hasattr(ui, "start_event_replay"))
        self.assertIn("CNN V3 permanece", self.window.archive_training_frame.findChildren(
            QLabel, "archiveTrainingHint"
        )[0].text())

    def test_action_grid_reflows_for_narrow_and_wide_without_duplicating_buttons(self):
        ui = install_training_validation_panel(self.window, root=self.root)
        self.assertEqual(ui.action_bar._columns, 1)
        self.assertEqual(ui.action_bar.grid.count(), 3)
        ui.action_bar._reflow(3)
        self.assertEqual(ui.action_bar._columns, 3)
        self.assertEqual(ui.action_bar.grid.count(), 3)
        for i, button in enumerate((
            ui.start_button, ui.shadow_button, ui.cancel_button,
        )):
            index = ui.action_bar.grid.indexOf(button)
            row, col, _rs, _cs = ui.action_bar.grid.getItemPosition(index)
            self.assertEqual((row, col), (0, i))
        ui.action_bar._reflow(1)
        for i, button in enumerate((
            ui.start_button, ui.shadow_button, ui.cancel_button,
        )):
            index = ui.action_bar.grid.indexOf(button)
            row, col, _rs, _cs = ui.action_bar.grid.getItemPosition(index)
            self.assertEqual((row, col), (i, 0))
        self.assertIn("archiveTrainingStart:pressed", ui.panel.archive_training_frame.styleSheet())
        self.assertIn("archiveTrainingKnnCnnShadow:focus", ui.panel.archive_training_frame.styleSheet())

    def test_history_progress_and_both_labels_remain_visible(self):
        ui = install_training_validation_panel(self.window, root=self.root)
        ui._handle({"event": "started", "images": 212, "total": 414})
        ui._handle({"event": "progress", "done": 1, "total": 414,
                    "model": "KNN_MEMORIA", "path": "public/ok_archive/a.png",
                    "expected": "OK", "status": "PASSOU"})
        ui._handle({"event": "progress", "done": 2, "total": 414,
                    "model": "CNN_FALTANDO_V2", "path": "public/ng_archive/b.png",
                    "expected": "NG", "status": "ERRO_CLASSIFICACAO"})
        self.assertEqual(ui.progress_bar.maximum(), 414)
        self.assertEqual(ui.progress_bar.value(), 2)
        self.assertIn("OK 1/1", ui.knn_metric.text())
        self.assertIn("NG 0/1", ui.cnn_metric.text())
        self.assertIn("1 pendências/erros", ui.cnn_metric.text())
        self.assertIn("b.png", ui.status.text())

    def test_html_safe_progress_and_model_summary(self):
        ui = install_training_validation_panel(self.window, root=self.root)
        ui._handle({"event": "started", "images": 1, "total": 2})
        ui._handle({"event": "progress", "done": 1, "total": 2,
                    "model": "KNN_MEMORIA", "path": "public/ok_archive/<x>.png",
                    "expected": "OK", "status": "PASSOU"})
        self.assertIn("&lt;x&gt;.png", ui.status.text())
        self.assertNotIn("<x>", ui.status.text())
        formatted = format_model_line("CNN_FALTANDO_V2", {
            "by_label": {
                "OK": {"passed": 190, "total": 195, "accuracy": .97436},
                "NG": {"passed": 17, "total": 17, "accuracy": 1.0},
            }, "incorrect": 5, "no_coverage": 0,
            "invalid": 0, "out_of_scope": 10,
        })
        self.assertIn("OK 190/195", formatted)
        self.assertIn("NG 17/17", formatted)
        self.assertIn("Fora da CNN 10", formatted)

    def test_completed_audit_does_not_claim_generalization_or_new_training(self):
        ui = install_training_validation_panel(self.window, root=self.root)
        ui._last_finished = {
            "complete": True,
            "canceled": False,
            "models": {
                "KNN_MEMORIA": {"target_98pct_per_label": True},
                "CNN_FALTANDO_V2": {"target_98pct_per_label": True},
            },
            "json_path": "reports/check.json",
            "txt_path": "reports/check.txt",
        }
        ui._finished(0, None)
        self.assertEqual(ui.status.property("state"), "complete")
        self.assertIn("histórica", ui.status.text())
        self.assertIn("Não comprova", ui.status.text())
        self.assertFalse(ui.cancel_button.isEnabled())
        self.assertTrue(ui.cancel_button.isHidden())

    def test_partial_or_unverified_report_never_claims_success(self):
        ui = install_training_validation_panel(self.window, root=self.root)
        ui._last_finished = {
            "complete": False, "canceled": False,
            "models": {
                "KNN_MEMORIA": {"target_98pct_per_label": True},
                "CNN_FALTANDO_V2": {"target_98pct_per_label": True},
            },
        }
        ui._finished(0, None)
        self.assertIn("incompleto", ui.status.text())
        self.assertEqual(ui.status.property("state"), "attention")

    def test_shadow_result_is_clearly_experimental(self):
        ui = install_training_validation_panel(self.window, root=self.root)
        ui._finish_shadow(0, {
            "results": {
                "cnn_isolated_baseline": {"ng_as_ok": 3, "ok_as_ng": 1},
                "guarded_conflict_to_review": {"ng_as_ok": 2, "ok_as_ng": 1, "review": 4},
            },
            "cnn_eligible_pngs": 80,
            "conflicts": {"total": 4},
        })
        self.assertEqual(ui.status.property("state"), "attention")
        self.assertIn("experimental", ui.status.text().lower())
        self.assertIn("NÃO autoriza produção", ui.details.text())


if __name__ == "__main__":
    unittest.main()
