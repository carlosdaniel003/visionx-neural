"""Regressão Windows: subprocesso Qt não pode herdar CP1252 do terminal."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication

from src.ui.archive_label_correction_ui import ArchiveLabelCorrectionDialog


class MaintenanceWorkerEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dialog = ArchiveLabelCorrectionDialog(root=Path(self.temp.name))
        self.addCleanup(self.dialog.deleteLater)

    def test_actual_qprocess_forces_utf8_despite_cp1252_parent(self):
        # Mesmo que o Python do terminal use cp1252, o processo Qt precisa
        # imprimir emoji e registrar todos os caracteres sem crash.
        with patch.dict(os.environ, {
            "PYTHONIOENCODING": "cp1252",
            "PYTHONUTF8": "0",
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        }):
            self.dialog._launch_worker(
                "tests.utf8_console_probe", [], workflow="encoding_probe"
            )
            worker = self.dialog.train_process
            self.assertIsNotNone(worker)
            self.assertEqual(worker.arguments()[:4], ["-X", "utf8", "-u", "-m"])
            process_environment = worker.processEnvironment()
            self.assertEqual(process_environment.value("PYTHONIOENCODING"), "utf-8")
            self.assertEqual(process_environment.value("PYTHONUTF8"), "1")

            deadline = time.monotonic() + 20.0
            while (self.dialog.train_process is not None
                   and time.monotonic() < deadline):
                self.app.processEvents()
                time.sleep(.02)
        self.assertIsNone(
            self.dialog.train_process,
            "Subprocesso de teste não encerrou em até 20 segundos"
        )
        log = self.dialog.log.toPlainText()
        self.assertIn("✅ Tesseract encontrado:", log)
        self.assertIn("ENCODING: utf-8", log)
        self.assertIn("COMPARAÇÃO DE TREINO CONCLUÍDA", log)
        self.assertFalse(
            (Path(self.temp.name) / "reports/neural_online/live_active.json").exists()
        )


if __name__ == "__main__":
    unittest.main()
